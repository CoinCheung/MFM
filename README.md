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

**Note**: Supervised pretrain means finetune from torchvision resnet weights (by setting `pretrained=True`). It seems that supervised pretrain is better than the proposed mfm pretrain.  



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
Pretraining and finetuning Command is [here](./dist_train.sh).


## More ablations  
Here are some points that affects the results: 

1. finetune `--val-resize-size`  
When we eval the model after finetuning, we always resize the short side of the image to a fixed value before a center crop operation. Here I find sometimes the value of fixed short side size affects the acc by a noticeable margin. Take the "supervised pretrain" as example: 
    <table><tbody>
    <th align="center">val-resize-size</th>
    <td align="center">234</td>
    <td align="center">235</td>
    <td align="center">236</td>

    <tr>
    <th align="center">top-1 acc</th>
    <td align="center">78.856</td>
    <td align="center">78.942</td>
    <td align="center">78.794</td>
    </tr>

    </tbody></table>

2. finetune with bce loss is important  
We can see this by finetuning from scratch with CE(cross entropy) loss and BCE(binary cross entropy) loss, the result is: 
    <table><tbody>
    <th align="center">loss</th>
    <td align="center">CE</td>
    <td align="center">BCE</td>

    <tr>
    <th align="center">top-1 acc</th>
    <td align="center">78.542</td>
    <td align="center">78.952</td>
    </tr>

    </tbody></table>

3. pretrain random crop area  
We usually crop a part of the image with certain area ratio from the original image, and the default value of this ratio is `0.08-1.0` with torchvision `RandomResizedCrop`. Different self-supervised learning methods tend to prefer different random area ratios. For example, MAE uses `0.2-1.0`, MAE3d uses `0.5-1.0`, and SimMIM uses `0.67-1.0`. Here I find a smaller lower bound of `0.2-1.0` is better:  
    <table><tbody>
    <th align="center">random area ratio</th>
    <td align="center">0.67-1.0</td>
    <td align="center">0.2-1.0</td>
    <td align="center">0.1-1.0</td>

    <tr>
    <th align="center">top-1 acc</th>
    <td align="center">78.770</td>
    <td align="center">78.826</td>
    <td align="center">78.842</td>
    </tr>

    </tbody></table>

    Though here `0.1-1.0` is better than `0.2-1.0`, I still use the latter, since, with `0.1-1.0`, the finetuning eval result is more affacted by `val-resize-size`: 
    <table><tbody>
    <th align="center">val-resize-size</th>
    <td align="center">234</td>
    <td align="center">235</td>
    <td align="center">236</td>

    <tr>
    <th align="center">0.2-1.0</th>
    <td align="center">78.816</td>
    <td align="center">78.826</td>
    <td align="center">78.796</td>
    </tr>

    <tr>
    <th align="center">0.1-1.0</th>
    <td align="center">78.730</td>
    <td align="center">78.842</td>
    <td align="center">78.738</td>
    </tr>
    </tbody></table>

4. model variance  
Here I pretrain the model for 4 times(2 on 8 v100 gpu, and 2 on 8 p40 gpu) with identical configuration. Then I finetune 3 times for each of the pretrained model(with 8 p40). Results are listed below. We can see that the results varies between a big margin. Maybe the above good results are brought by a good luck. Hence, I cannot say that I have certainly reproduced the results in the paper now.
    <table><tbody>
    <th align="center">pretrain</th>
    <th align="center">finetune</th>
    <th align="center">acc1(235)</th>
    <th align="center" colspan='2'>mean/std</th>

    <tr>
    <td align="center" rowspan="3">round 1</td>
    <td align="center">round 1</td>
    <td align="center">78.654</td>
    <td align="center" rowspan="3">78.644/0.024</td>
    <td align="center" rowspan="12">78.621/0.08</td>
    </tr>

    <tr>
    <td align="center">round 2</td>
    <td align="center">78.61</td>
    </tr>

    <tr>
    <td align="center">round 3</td>
    <td align="center">78.668</td>
    </tr>

    <tr>
    <td align="center" rowspan="3">round 2</td>
    <td align="center">round 1</td>
    <td align="center">78.646</td>
    <td align="center" rowspan="3">78.642/0.122</td>
    </tr>

    <tr>
    <td align="center">round 2</td>
    <td align="center">78.79</td>
    </tr>

    <tr>
    <td align="center">round 3</td>
    <td align="center">78.49</td>
    </tr>

    <tr>
    <td align="center" rowspan="3">round 3</td>
    <td align="center">round 1</td>
    <td align="center">78.516</td>
    <td align="center" rowspan="3">78.612/0.073</td>
    </tr>

    <tr>
    <td align="center">round 2</td>
    <td align="center">78.626</td>
    </tr>

    <tr>
    <td align="center">round 3</td>
    <td align="center">78.694</td>
    </tr>

    <tr>
    <td align="center" rowspan="3">round 4</td>
    <td align="center">round 1</td>
    <td align="center">78.608</td>
    <td align="center" rowspan="3">78.584/0.080</td>
    </tr>

    <tr>
    <td align="center">round 2</td>
    <td align="center">78.668</td>
    </tr>

    <tr>
    <td align="center">round 3</td>
    <td align="center">78.476</td>
    </tr>
    </tbody></table>


