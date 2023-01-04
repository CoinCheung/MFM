# MFM
Unofficial code for paper "Masked Feature Prediction for Self-Supervised Visual Pre-Training" (https://arxiv.org/pdf/2206.07706.pdf)


Though better result is achieved, it seems that the baseline is also much higher than in paper.  

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">top-1 acc</th>
<th valign="bottom">pretrain</th>
<th valign="bottom">finetune</th>
<!-- TABLE BODY -->

<tr>
<td align="center">paper scratch</td>
<td align="center">78.1</td>
<td align="center"></td>
<td align="center"></td>
</tr>

<tr>
<td align="center">paper mfm pretrain</td>
<td align="center">78.5</td>
<td align="center"></td>
<td align="center"></td>
</tr>

<tr>
<td align="center">scratch</td>
<td align="center">78.542</td>
<td align="center">-</td>
<td align="center">link</td>
</tr>

<tr>
<td align="center">supervised pretrain</td>
<td align="center">78.942</td>
<td align="center"></td>
<td align="center"><a href="https://github.com/CoinCheung/MFM/releases/download/0.0.0/saved_model_bce_0_pretrain.pth">link</a></td>
</tr>

<tr>
<td align="center">mfm pretrain</td>
<td align="center">78.826</td>
<td align="center"><a href="https://github.com/CoinCheung/MFM/releases/download/0.0.0/saved_model_15.pth">link</a></td>
<td align="center"><a href="https://github.com/CoinCheung/MFM/releases/download/0.0.0/saved_model_bce_15.pth">link</a></td>
</tr>

</tbody></table>

&#8195;&#8195;Note: Supervised pretrain means finetune from torchvision resnet weights (by setting `pretrained=True`). It seems that supervised pretrain is better.  



## Platform

* pytorch 1.11.0
* torchvision 0.12.0
* dali 1.16.0
* V100 GPU(32G) x 8
* driver: 450.172.01


## Dataset
Prepare imagenet val set in same method as pytorch official classification [example](https://github.com/pytorch/examples/tree/main/imagenet), and then link them to the folder of this repo:
```
    $ mkdir -p imagenet
    $ ln -s /path/to/imagenet/train ./imagenet/train
    $ ln -s /path/to/imagenet/val ./imagenet/val
```

## Train
Pretraining and finetuning Command is [here](./dist_train.sh)


