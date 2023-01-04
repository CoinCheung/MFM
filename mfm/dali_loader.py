
import os
import os.path as osp
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
from nvidia.dali.pipeline import Pipeline
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types





@pipeline_def
def create_dali_pipeline(data_dir, crop, size, shard_id, num_shards, dali_cpu=False, is_training=True):
    images, labels = fn.readers.file(
            file_root=data_dir,
            shard_id=shard_id,
            num_shards=num_shards,
            random_shuffle=is_training,
            pad_last_batch=True,
            name="Reader")
    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'

    # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
    preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
    preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0
    if is_training:
        images = fn.decoders.image_random_crop(
                images,
                device=decoder_device, output_type=types.RGB,
                device_memory_padding=device_memory_padding,
                host_memory_padding=host_memory_padding,
                preallocate_width_hint=preallocate_width_hint,
                preallocate_height_hint=preallocate_height_hint,
                random_aspect_ratio=[0.8, 1.25],
                #  random_aspect_ratio=[0.75, 1.34],
                #  random_area=[0.67, 1.0],
                random_area=[0.2, 1.0],
                num_attempts=100)
        images = fn.resize(images,
                device=dali_device,
                resize_x=crop,
                resize_y=crop,
                interp_type=types.INTERP_TRIANGULAR)
        mirror = fn.random.coin_flip(probability=0.5)
    else:
        images = fn.decoders.image(
                images,
                device=decoder_device,
                output_type=types.RGB)
        images = fn.resize(images,
                device=dali_device,
                size=size,
                mode="not_smaller",
                interp_type=types.INTERP_TRIANGULAR)
        mirror = False

    mean = [0.4, 0.4 ,0.4]
    std  = [0.2, 0.2 ,0.2]
    images = fn.crop_mirror_normalize(images.gpu(),
            dtype=types.FLOAT,
            output_layout="CHW",
            crop=(crop, crop),
            mean=[el * 255 for el in mean],
            std=[el * 255 for el in std],
            mirror=mirror)
    labels = labels.gpu()
    return images, labels


def create_dali_dataloader_train(args, dali_cpu=False):
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    traindir = osp.join(args.data_path, 'train')

    pipe = create_dali_pipeline(
            batch_size=args.batch_size,
            num_threads=args.workers,
            device_id=local_rank,
            seed=12 + local_rank,
            data_dir=traindir,
            crop=args.train_crop_size,
            size=None, # useless
            dali_cpu=dali_cpu,
            shard_id=local_rank,
            num_shards=world_size,
            is_training=True)
    pipe.build()
    train_loader = DALIClassificationIterator(
            pipe, reader_name="Reader",
            last_batch_policy=LastBatchPolicy.PARTIAL)

    return train_loader


def create_dali_dataloader_test(args, dali_cpu=False):
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    valdir = osp.join(args.data_path, 'val')

    pipe = create_dali_pipeline(
            batch_size=args.batch_size,
            num_threads=args.workers,
            device_id=local_rank,
            seed=12 + local_rank,
            data_dir=valdir,
            crop=args.val_crop_size,
            size=args.val_resize_size,
            dali_cpu=dali_cpu,
            shard_id=local_rank,
            num_shards=world_size,
            is_training=False)
    pipe.build()
    val_loader = DALIClassificationIterator(
            pipe, reader_name="Reader",
            last_batch_policy=LastBatchPolicy.PARTIAL)

    return val_loader
