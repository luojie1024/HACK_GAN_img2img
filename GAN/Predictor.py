# -*- coding: utf-8 -*-
'''
# @Time    : 5/23/18 8:43 PM
# @Author  : luojie
# @File    : api_Server.py.py
# @Desc    : 
'''

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import time

from GAN.gan_tool import CROP_SIZE, create_generator, out_channels, deprocess, A_Aspect_Ratio, \
    save_simple_images, load_1_pic

base_model_path = os.getcwd() + '/model/'

Facades = base_model_path + 'facades_BtoA'
Shoes = base_model_path + 'edges2shoes_AtoB'
Handbags = base_model_path + 'edges2handbags_AtoB'
CitysA2B = base_model_path + 'cityscapes_AtoB'
CitysB2A = base_model_path + 'cityscapes_BtoA'
MapsA2B = base_model_path + 'maps_AtoB'
MapsB2A = base_model_path + 'maps_BtoA'

models_dict = {'Facades': Facades, 'Shoes': Shoes, 'Handbags': Handbags, 'CitysA2B': CitysA2B, 'MapsA2B': MapsA2B,
               'CitysB2A': CitysB2A, 'MapsB2A': MapsB2A}


class Predictor:
    def __init__(self, model):
        # 随机种子
        seed = random.randint(0, 2 ** 31 - 1)
        tf.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.mode = models_dict[model]
        print('########model pach=%s' % self.mode)
        self.image_name = model

    def Preediction(self, base64data):
        display_fetches = self.input_load(base64data)
        print('########Preediction display_fetches ########3')
        # model
        saver = tf.train.Saver(max_to_keep=1)
        sv = tf.train.Supervisor(save_summaries_secs=0, saver=None)

        # 模型载入
        with sv.managed_session() as sess:
            # 测试
            start = time.time()
            checkpoint = tf.train.latest_checkpoint(self.mode)
            saver.restore(sess, checkpoint)
            results = sess.run(display_fetches)
            save_simple_images(results)
            print("rate", (time.time() - start))
            return results['outputs'][0]

    # 卷积图片
    def convert(self, image):
        if A_Aspect_Ratio != 1.0:
            # upscale to correct aspect ratio
            size = [CROP_SIZE, int(round(CROP_SIZE * A_Aspect_Ratio))]
            image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)
        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    def input_load(self, base64data):
        print('#######load_base64_pic########')
        inputs = load_1_pic(self.image_name)
        print('#######inputs load_base64_pic########')
        print(inputs)
        # output
        with tf.variable_scope("generator"):
            outputs = create_generator(inputs, out_channels)

        outputs = deprocess(outputs)
        converted_outputs = self.convert(outputs)
        display_fetches = {
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
        }

        return display_fetches
