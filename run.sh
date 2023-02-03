
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# export CUDA_VISIBLE_DEVICES=0,1,6,7
# NGPUS=4
# LR=0.0006
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
NGPUS=6
LR=0.00135

## bs=2048, lr=1.2e-3

# mfm pretraining
# torchrun --nproc_per_node=$NGPUS train_mfm.py \
#     --data-path ./imagenet/ \
#     --model resnet50 \
#     --epochs 300 \
#     --opt adamw \
#     --batch-size 384 \
#     --lr $LR \
#     --wd 0.05 \
#     --lr-scheduler cosineannealinglr \
#     --lr-warmup-epochs 20 \
#     --clip-grad-norm 3.0 \
#     --lr-warmup-method linear \
#     --output-dir ./res_pretrain \
#     --amp \
#     --use-dali \
#     --train-crop-size 224
#
#     # --resume res_pretrain/model_60.pth \
#
#
# sleep 10

# finetune 100ep
PORT=45276
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# export CUDA_VISIBLE_DEVICES=0,1,6,7
# NGPUS=4
# LR=0.012
# BS=512
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NGPUS=8
LR=0.012
BS=256

## bs=2048, lr=1.2e-2
#

export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
NGPUS=6
LR=0.0135
BS=384
torchrun --nproc_per_node=$NGPUS --master_port $PORT train_finetune.py \
    --data-path ./imagenet/ \
    --model resnet50 \
    --batch-size $BS \
    --epochs 100 \
    --opt adamw \
    --lr $LR \
    --wd 0.02 \
    --label-smoothing 0.1 \
    --mixup-alpha 0.1 \
    --cutmix-alpha 1.0 \
    --lr-scheduler cosineannealinglr \
    --lr-warmup-epochs 5 \
    --lr-warmup-method linear \
    --output-dir ./res_finetune \
    --auto-augment ra_6_10 \
    --weights ./res_pretrain/saved_model_24.pth \
    --amp \
    --val-resize-size 235 \
    --train-crop-size 160


# ckpts=(saved_model_19.pth saved_model_20.pth saved_model_21.pth saved_model_22.pth)
# for ckpt in ${ckpts[@]};
# do
#     echo $ckpt
#     for i in $(seq 1 1 3);
#     do
#         torchrun --nproc_per_node=$NGPUS --master_port $PORT train_finetune.py \
#             --data-path ./imagenet/ \
#             --model resnet50 \
#             --batch-size $BS \
#             --epochs 100 \
#             --opt adamw \
#             --lr $LR \
#             --wd 0.02 \
#             --label-smoothing 0.1 \
#             --mixup-alpha 0.1 \
#             --cutmix-alpha 1.0 \
#             --lr-scheduler cosineannealinglr \
#             --lr-warmup-epochs 5 \
#             --lr-warmup-method linear \
#             --output-dir ./res_finetune \
#             --auto-augment ra_6_10 \
#             --weights ./res_pretrain/$ckpt \
#             --amp \
#             --val-resize-size 235 \
#             --train-crop-size 160 2>&1 | tee log_${ckpt}_${i}.txt
#         cp -riv ./res_finetune/model_100.pth ./res_finetune/${ckpt}_$i
#         sleep 60
#     done
# done
