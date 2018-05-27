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

from gan_tool import CROP_SIZE, create_generator, out_channels, deprocess, A_Aspect_Ratio, \
    A_Facades_Train, save_simple_images, load_pic_examples


def main():
    # 随机种子ssd
    seed = random.randint(0, 2 ** 31 - 1)
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    inputs = load_pic_examples('54')

    # output
    with tf.variable_scope("generator"):
        outputs = create_generator(inputs, out_channels)

    # 卷积图片
    def convert(image):
        if A_Aspect_Ratio != 1.0:
            # upscale to correct aspect ratio
            size = [CROP_SIZE, int(round(CROP_SIZE * A_Aspect_Ratio))]
            image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)
        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)


    outputs = deprocess(outputs)
    converted_outputs = convert(outputs)

    display_fetches = {
        "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
    }

    # model
    saver = tf.train.Saver(max_to_keep=1)
    sv = tf.train.Supervisor(save_summaries_secs=0, saver=None)

    # 模型载入
    with sv.managed_session() as sess:
        # 测试
        start = time.time()
        checkpoint = tf.train.latest_checkpoint(A_Facades_Train)
        saver.restore(sess, checkpoint)
        # feed_dict={xs: inputs}
        results = sess.run(display_fetches)
        save_simple_images(results)
        print("rate", (time.time() - start))


main()
