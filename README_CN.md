# 使用GAN(生成对抗网络)进行图像生成应用开发以图生图应用开发
详情见服务端代码地址,感谢大佬star

## 华中“Hackathon”创客马拉松大赛

> 本次由湖南大学和微软亚洲研究院主办，由国家超级计算长沙中心、微软学生俱乐部等承办的华中“Hackathon”创客马拉松大赛，将邀请150名高校中创意、开发、设计、营销达人聚集在一起，在24小时连续不间断工作坊中大展身手，我们将全力以赴让创客精神在华中地区迸发魅力。

> 此次Hackathon由湖南大学和微软亚洲研究院联合主办，比赛地点为国家超级计算长沙中心。大赛以激励华中地区高校学生的创新精神、创造精神为目标，为有想法、有创意的学生们提供一个共同交流、互相切磋，并将想法付诸实践的大舞台。届时，华中地区150名富有热情的青年创客聚集在一起，共同享受24小时不间断协作开发和学习交流的知识盛宴，让创意的火花时刻迸发。


### HACKX项目地址:
[Project URL](https://www.hackx.org/projects/299)

### 队名:
取个名字真TM难

### 作者:
+ [XueWenLiao](https://github.com/XuewenLiao) 
+ ChengChen 
+ [LuoJie](https://github.com/luojie1024)
+ [LiuDong](https://github.com/1mrliu)  
+ [WuQin](https://www.zhihu.com/people/jessiewu-qin/activities) [`Surprise Scene`](https://changba.com/wap/index.php?s=M89dzpF8OTcFqbT_3Qhnyw)

### 平台:
- Android 7.0
- Tensorflow 1.4.1 
- Django 2.0.5
- djangorestframework 3.8.2  


### 推荐
- Linux with Tensorflow GPU edition + cuDNN

### 代码地址:
https://github.com/luojie1024/HACK_GAN_MB
|[Android客户端](https://github.com/XuewenLiao/PaintSole)|[Python服务端+GAN模型](https://github.com/luojie1024/HACK_GAN_MB)|
-|-

## 项目描述
###简介:
> 
通过使用Tensorflow,GAN,Django,Android等技术,实现快速造图,来提升沟通效率,用户只需手绘草图,AI将实时生成逼真的效果图.在原型展示,室内装修,服装设计,LOGO设计等领域有广泛的应用.即画即现,AI让沟通变得如此简单.


### 描述:
> 基于GAN技术,即画即现,快速的将自己的想法转化成图像,提升沟通效率.使用Tensorflow构建GAN模型,将训练好的模型封装,使用Django框架进行服务器的搭建,提供API供客户端调用,本项目使用Android客户端进行演示,将用户手绘草图,通过GAN神经网络转化成十分逼真的效果图.

### 想法:
> 现代社会人与人之间的交流变得很频繁，每个人都要和不同的对象进行交流，但是并不是每个人都可以很好的表达自己的想法，因此会造成沟通的障碍。比如：当设计原型图时，需要在讨论时及时生成一个可展示的对象，可以准确的展示用户的需求，降低沟通难度。我们的实际就来源于人与人之间沟通难度的存在。

### 解决的问题:
1. 降低沟通难度。让人们能够更加准确的表达自己的想法。

2. 降低沟通成本。让人们能够更加便捷的使用可以交互的沟通方式。

3. 触发使用者的灵感，可以更加全面挖掘使用者的灵感

### 核心技术描述

#### 客户端:
>技术难点为：
客户端与服务器交互图片数据问题

>现使用方案：
将用户所画草图以JPEG图片格式保存在本地，将图片用Base64（编码规范）编码后以json的形式post到服务器。服务器将编码过的图片传回到客户端，然后客户端解码为JPEG图片格式以供展示。由于这样的方法数据交互速度及慢，需要优化方案。

>未来展望:
采用Google提供的grpc框架实现图片传输，该框架采用HTTP2.0作为数据传输协议，传输图片速度碾压HTTP1.1.

#### 后台:
>技术难点为：
1.模型的训练
2.模型转化成API,提供服务.

> 现使用方案:
使用Tensorflow实现GAN模型,将训练好的不同模型封成对应的API,将模型部署到服务端,提供给多终端调用(手机,平板,WEB,PC).

>未来展望:
准备使用Tensorflow Serving简化并加速模型到生产的过程,保持服务器架构和API保持不变,安全安全地部署新模型并运行试验。

## 项目展示

### LOGO
<div align=center><img width="256" src="https://github.com/luojie1024/HACK_GAN_MB/raw/master/image/logo.png"/></div>

### 图片

<div align=center><img width="256" src="https://cdn.hackx.org/undefined_1527394668230.png"/></div>

<div align=center><img width="256" src="https://cdn.hackx.org/undefined_1527394497926.png"/></div>

<div align=center><img width="256" src="https://cdn.hackx.org/o_1cefh9srb1jf11ac312gs19m7pg210.jpg"/></div>

<div align=center><img width="256" src="https://cdn.hackx.org/undefined_1527394477615.png"/></div>

<div align=center><img width="256" src="https://cdn.hackx.org/o_1cefh1f1jikq1311js414i91c0vc.jpg"/></div>

<div align=center><img width="256" src="https://cdn.hackx.org/undefined_1527394558844.jpg"/></div>

<div align=center><img width="256" src="https://cdn.hackx.org/undefined_1527394508549.png"/></div>

<div align=center><img width="256" src="https://cdn.hackx.org/undefined_1527394558794.jpg"/></div>

<div align=center><img width="256" src="https://cdn.hackx.org/undefined_1527394559033.jpg"/></div>

<div align=center><img width="256" src="https://cdn.hackx.org/undefined_1527394559111.jpg"/></div>

### 功能描述

#### 画板:
1. 手绘：用户可随意创作，不加任何限制。

2. 橡皮擦：擦除画错的部分。

3. 直线：用户只能用直线创作。

4. 撤销：用户可撤销前一步操作。

5. 清空：用户可清空画板。

6. 生成图片：创作完成后可即刻生成真是图片。

#### 模型:

1. 建筑模型：用户可使用五个标签组件：墙、门、窗户、屋檐和房柱来协助创作。

2. 街景模型：用户可使用五个标签组件：公路、草坪、汽车、树木和路灯来协助创作。

3. 包模型：用户可创作包。

4. 鞋模型：用户可创作鞋。.

### 视频:
[DEMO URL](https://www.bilibili.com/video/av23990741/)


### PPT展示:
<div align=center><img width="512" src="https://github.com/luojie1024/HACK_GAN_MB/raw/master/image/1.png"/></div>

<div align=center><img width="512" src="https://github.com/luojie1024/HACK_GAN_MB/raw/master/image/2.png"/></div>
<div align=center><img width="512" src="https://github.com/luojie1024/HACK_GAN_MB/raw/master/image/3.png"/></div>
<div align=center><img width="512" src="https://github.com/luojie1024/HACK_GAN_MB/raw/master/image/4.png"/></div>g"/></div>

<div align=center><img width="512" src="https://github.com/luojie1024/HACK_GAN_MB/raw/master/image/5.png"/></div>

<div align=center><img width="512" src="https://github.com/luojie1024/HACK_GAN_MB/raw/master/image/6.png"/></div>

<div align=center><img width="512" src="https://github.com/luojie1024/HACK_GAN_MB/raw/master/image/7.png"/></div>

<div align=center><img width="512" src="https://github.com/luojie1024/HACK_GAN_MB/raw/master/image/8.png"/></div>

## 项目构建 

### 开始
```
# clone this repo
git clone git@github.com:luojie1024/HACK_GAN_MB.git
cd HACK_GAN_MB
```
[服务器搭建参考](https://blog.csdn.net/luojie140/article/details/76832749)

### 数据集

| dataset | example |
| --- | --- |
|<br> 400 images from [CMP Facades dataset](http://cmp.felk.cvut.cz/~tylecr1/facade/). (31MB) <br> Pre-trained: [BtoA](https://mega.nz/#!H0AmER7Y!pBHcH4M11eiHBmJEWvGr-E_jxK4jluKBUlbfyLSKgpY)  | <img src="https://github.com/luojie1024/HACK_GAN_MB/raw/master/docs/facades.jpg" width="256px"/> |
|<br> 2975 images from the [Cityscapes training set](https://www.cityscapes-dataset.com/). (113M) <br> Pre-trained: [AtoB](https://mega.nz/#!K1hXlbJA!rrZuEnL3nqOcRhjb-AnSkK0Ggf9NibhDymLOkhzwuQk) [BtoA](https://mega.nz/#!y1YxxB5D!1817IXQFcydjDdhk_ILbCourhA6WSYRttKLrGE97q7k) | <img src="https://github.com/luojie1024/HACK_GAN_MB/raw/master/docs/cityscapes.jpg" width="256px"/> |
| <br> 1096 training images scraped from Google Maps (246M) <br> Pre-trained: [AtoB](https://mega.nz/#!7oxklCzZ!8fRZoF3jMRS_rylCfw2RNBeewp4DFPVE_tSCjCKr-TI) [BtoA](https://mega.nz/#!S4AGzQJD!UH7B5SV7DJSTqKvtbFKqFkjdAh60kpdhTk9WerI-Q1I) | <img src="https://github.com/luojie1024/HACK_GAN_MB/raw/master/docs/maps.jpg" width="256px"/> |
|<br> 50k training images from [UT Zappos50K dataset](http://vision.cs.utexas.edu/projects/finegrained/utzap50k/). Edges are computed by [HED](https://github.com/s9xie/hed) edge detector + post-processing. (2.2GB) <br> Pre-trained: [AtoB](https://mega.nz/#!u9pnmC4Q!2uHCZvHsCkHBJhHZ7xo5wI-mfekTwOK8hFPy0uBOrb4) | <img src="https://github.com/luojie1024/HACK_GAN_MB/raw/master/docs/edges2shoes.jpg" width="256px"/>  |
| <br> 137K Amazon Handbag images from [iGAN project](https://github.com/junyanz/iGAN). Edges are computed by [HED](https://github.com/s9xie/hed) edge detector + post-processing. (8.6GB) <br> Pre-trained: [AtoB](https://mega.nz/#!G1xlDCIS!sFDN3ZXKLUWU1TX6Kqt7UG4Yp-eLcinmf6HVRuSHjrM) | <img src="https://github.com/luojie1024/HACK_GAN_MB/raw/master/docs/edges2handbags.jpg" width="256px"/> |

The `facades` dataset is the smallest and easiest to get started with.

## 展望
1. 我们的产品未来可以应用到的领域有：装修装潢，建筑设计、服装设计，品牌LOGO、城镇规划、文物复原、动画动漫设计等。

2. 给不同行业用户提供更专业的工具包 

3. 未来当AI模型足够精准，为所有开发者和企业提供api sdk使用



### 参考资料

#### 代码
Based on [pix2pix](https://phillipi.github.io/pix2pix/) by Isola et al.

#### 论文
[Image-to-Image Translation with Conditional Adversarial Nets](https://arxiv.org/abs/1611.07004) [CVPR 2017]
