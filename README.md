# MFM
Unofficial code for paper "Masked Feature Prediction for Self-Supervised Visual Pre-Training" (https://arxiv.org/pdf/2206.07706.pdf)

Below are experiments with resnet50. Though better result is achieved, it seems that the baseline is also much higher than in paper.  

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
<td align="center">-</td>
<td align="center">-</td>
</tr>

<tr>
<td align="center">paper mfm pretrain</td>
<td align="center">78.5</td>
<td align="center">-</td>
<td align="center">-</td>
</tr>

<tr>
<td align="center">scratch</td>
<td align="center">78.542</td>
<td align="center">-</td>
<td align="center"><a href="https://github.com/CoinCheung/MFM/releases/download/0.0.0/saved_model_bce_0_nopretrain.pth">link</a></td>
</tr>

<tr>
<td align="center">supervised pretrain</td>
<td align="center">78.942</td>
<td align="center">-</td>
<td align="center"><a href="https://github.com/CoinCheung/MFM/releases/download/0.0.0/saved_model_bce_0_pretrain.pth">link</a></td>
</tr>

<tr>
<td align="center">mfm pretrain</td>
<td align="center">78.826</td>
<td align="center"><a href="https://github.com/CoinCheung/MFM/releases/download/0.0.0/saved_model_15.pth">link</a></td>
<td align="center"><a href="https://github.com/CoinCheung/MFM/releases/download/0.0.0/saved_model_bce_15.pth">link</a></td>
</tr>

</tbody></table>

**Note**: Supervised pretrain means finetune from torchvision resnet weights (by setting `pretrained=True`). It seems that supervised pretrain is better.  



## Platform

* pytorch 1.13.1
* torchvision 0.14.1
* dali 1.21.0
* cuda 11.6
* V100 GPU(32G) x 8
* driver: 470.82.01


## Dataset
Prepare imagenet val set in same method as pytorch official classification [example](https://github.com/pytorch/examples/tree/main/imagenet), and then link them to the folder of this repo:
```
    $ mkdir -p imagenet
    $ ln -s /path/to/imagenet/train ./imagenet/train
    $ ln -s /path/to/imagenet/val ./imagenet/val
```

## Train
Pretraining and finetuning Command is [here](./dist_train.sh)


## More ablations  
Here are some points that affects the results: 

1. finetune `--val-resize-size`
When we eval the model after finetuning, we always resize the short side of the image to a fixed value before a center crop operation. Here I find sometimes the fixed short side size affects the acc by a noticeable margin. Take the "supervised pretrain" as example: 
<table><tbody>
<th valign="bottom">val-resize-size</th>
<th valign="bottom">234</th>
<th valign="bottom">235</th>
<th valign="bottom">236</th>

<tr>
<td align="center">top-1 acc</td>
<td align="center">78.856</td>
<td align="center">78.942</td>
<td align="center">78.794</td>
</tr>

</tbody></table>

2. finetune with bce loss is important  
We can see the difference of this by finetuning from scratch with CE(cross entropy) loss and BCE(binary cross entropy) loss, and the result is: 
<table><tbody>
<th valign="bottom">loss</th>
<th valign="bottom">CE</th>
<th valign="bottom">BCE</th>

<tr>
<td align="center">top-1 acc</td>
<td align="center">78.542</td>
<td align="center">77.952</td>
</tr>

</tbody></table>

3. pretrain random crop area
We usually crop a part of the image with certain area ratio from the original image, and the default value of this ratio is `0.08-1.0` with torchvision `RandomResizedCrop`. Different self-supervised learning methods tend to prefer different random area ratios. For example, MAE uses `0.2-1.0`, MAE3d uses `0.5-1.0`, and SimMIM uses `0.67-1.0`. Here I find a smaller lower bound of `0.2-1.0` is better:  
<table><tbody>
<th valign="bottom">random area ratio</th>
<th valign="bottom">0.67-1.0</th>
<th valign="bottom">0.2-1.0</th>
<th valign="bottom">0.1-1.0</th>

<tr>
<td align="center">top-1 acc</td>
<td align="center">78.770</td>
<td align="center">77.826</td>
<td align="center">77.842</td>
</tr>

</tbody></table>

Though here `0.1-1.0` is better than `0.2-1.0`, I still use the latter, since with `0.1-1.0`, the finetuning eval results is more affacted by `val-resize-size`: 
<table><tbody>
<th valign="bottom">val-resize-size</th>
<th valign="bottom">234</th>
<th valign="bottom">235</th>
<th valign="bottom">236</th>

<tr>
<td align="center">acc `0.2-1.0`</td>
<td align="center">78.816</td>
<td align="center">78.826</td>
<td align="center">78.796</td>
</tr>

<tr>
<td align="center">acc `0.1-1.0`</td>
<td align="center">78.730</td>
<td align="center">78.842</td>
<td align="center">78.738</td>
</tr>
</tbody></table>
