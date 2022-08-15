# MFM
Unofficial code for paper "Masked Feature Prediction for Self-Supervised Visual Pre-Training" (https://arxiv.org/abs/2112.09133)


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

## Train
Pretraining and finetuning Command is [here](./dist_train.sh)


#### Sadly, I cannot reproduce the result in the paper now, and I am still doing experiments to try different settings. Maybe I will fail and give it up, until the original authors release their code.

#### Hope this codebase can inspire other people who is interested in this work, and hope there is someone who can point out my mistake or tell me how to make it work.
