# MFM
code for paper "Masked Feature Prediction for Self-Supervised Visual Pre-Training" (https://arxiv.org/abs/2112.09133)


## Platform

* pytorch 1.11.0
* torchvision 0.12.0
* dali 1.16.0


## Dataset
Prepare imagenet val set in same method as pytorch official classification [example](https://github.com/pytorch/examples/tree/main/imagenet), and then link them to the folder of this repo:
```
    $ mkdir -p imagenet
    $ ln -s /path/to/imagenet/train ./imagenet/train
    $ ln -s /path/to/imagenet/val ./imagenet/val
```
