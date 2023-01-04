
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# NGPUS=4
# LR=0.006
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NGPUS=8
LR=0.012

## bs=2048, lr=1.2e-2

torchrun --nproc_per_node=$NGPUS --master_port 33212 train_finetune.py \
    --data-path ./imagenet/ \
    --model resnet50 \
    --batch-size 128 \
    --epochs 100 \
    --opt adamw \
    --lr $LR \
    --wd 0.02 \
    --resume res_finetune/saved_model_bce_15.pth \
    --test-only \
    --label-smoothing 0.1 \
    --mixup-alpha 0.1 \
    --cutmix-alpha 1.0 \
    --lr-scheduler cosineannealinglr \
    --lr-warmup-epochs 5 \
    --lr-warmup-method linear \
    --output-dir ./res_finetune \
    --auto-augment ra_6_10 \
    --amp \
    --val-resize-size 235 \
    --train-crop-size 165

    # --resume res_finetune/model_100.pth \
