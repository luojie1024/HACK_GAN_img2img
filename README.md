# HACK_GAN_MB

Based on [pix2pix](https://phillipi.github.io/pix2pix/) by Isola et al.

## Setup

### Prerequisites
- Tensorflow 1.4.1 
- Django 2.0.5
- djangorestframework 3.8.2  

### Recommended
- Linux with Tensorflow GPU edition + cuDNN

### Getting Started

```
# clone this repo
git clone git@github.com:luojie1024/HACK_GAN_MB.git
cd HACK_GAN_MB
```


| dataset | example |
| --- | --- |
| `python tools/download-dataset.py facades` <br> 400 images from [CMP Facades dataset](http://cmp.felk.cvut.cz/~tylecr1/facade/). (31MB) <br> Pre-trained: [BtoA](https://mega.nz/#!H0AmER7Y!pBHcH4M11eiHBmJEWvGr-E_jxK4jluKBUlbfyLSKgpY)  | <img src="docs/facades.jpg" width="256px"/> |
| `python tools/download-dataset.py cityscapes` <br> 2975 images from the [Cityscapes training set](https://www.cityscapes-dataset.com/). (113M) <br> Pre-trained: [AtoB](https://mega.nz/#!K1hXlbJA!rrZuEnL3nqOcRhjb-AnSkK0Ggf9NibhDymLOkhzwuQk) [BtoA](https://mega.nz/#!y1YxxB5D!1817IXQFcydjDdhk_ILbCourhA6WSYRttKLrGE97q7k) | <img src="docs/cityscapes.jpg" width="256px"/> |
| `python tools/download-dataset.py maps` <br> 1096 training images scraped from Google Maps (246M) <br> Pre-trained: [AtoB](https://mega.nz/#!7oxklCzZ!8fRZoF3jMRS_rylCfw2RNBeewp4DFPVE_tSCjCKr-TI) [BtoA](https://mega.nz/#!S4AGzQJD!UH7B5SV7DJSTqKvtbFKqFkjdAh60kpdhTk9WerI-Q1I) | <img src="docs/maps.jpg" width="256px"/> |
| `python tools/download-dataset.py edges2shoes` <br> 50k training images from [UT Zappos50K dataset](http://vision.cs.utexas.edu/projects/finegrained/utzap50k/). Edges are computed by [HED](https://github.com/s9xie/hed) edge detector + post-processing. (2.2GB) <br> Pre-trained: [AtoB](https://mega.nz/#!u9pnmC4Q!2uHCZvHsCkHBJhHZ7xo5wI-mfekTwOK8hFPy0uBOrb4) | <img src="docs/edges2shoes.jpg" width="256px"/>  |
| `python tools/download-dataset.py edges2handbags` <br> 137K Amazon Handbag images from [iGAN project](https://github.com/junyanz/iGAN). Edges are computed by [HED](https://github.com/s9xie/hed) edge detector + post-processing. (8.6GB) <br> Pre-trained: [AtoB](https://mega.nz/#!G1xlDCIS!sFDN3ZXKLUWU1TX6Kqt7UG4Yp-eLcinmf6HVRuSHjrM) | <img src="docs/edges2handbags.jpg" width="256px"/> |

The `facades` dataset is the smallest and easiest to get started with.
