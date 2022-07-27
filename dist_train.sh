

# python main_mfm.py \
#     -a resnet50  \
#     -b 256 \
#     --epoch 1 \
#     --dist-url 'tcp://127.0.0.1:10023' \
#     --use-mixed-precision \
#     --dist-backend 'nccl' \
#     --multiprocessing-distributed \
#     --world-size 1 --rank 0 \
#     --save-path ./mfm \
#     ./imagenet/

export CUDA_VISIBLE_DEVICES=0,1,2,3
NGPUS=4

torchrun --nproc_per_node=$NGPUS train_mfm.py \
    --data-path ./imagenet/ \
    --model resnet50 \
    --batch-size 256 \
    --epochs 300 \
    --opt adamw \
    --lr 0.0012 \
    --wd 0.05 \
    --lr-scheduler cosineannealinglr \
    --lr-warmup-epochs 20 \
    --clip-grad-norm 3.0 \
    --lr-warmup-method linear \
    --output-dir ./res_pretrain \
    --amp \
    --train-crop-size 224


# finetune 100ep
# torchrun --nproc_per_node=$NGPUS train_finetune.py \
#     --data-path ./imagenet/ \
#     --model resnet50 \
#     --batch-size 256 \
#     --epochs 100 \
#     --opt adamw \
#     --lr 0.012 \
#     --wd 0.02 \
#     --label-smoothing 0.1 \
#     --mixup-alpha 0.1 \
#     --cutmix-alpha 1.0 \
#     --lr-scheduler cosineannealinglr \
#     --lr-warmup-epochs 5 \
#     --lr-warmup-method linear \
#     --output-dir ./res_finetune \
#     --auto-augment ra_6_10 \
#     --amp \
#     --val-resize-size 236 \
#     --train-crop-size 160
#
#
#     # --weights None # pretrain加在这里
