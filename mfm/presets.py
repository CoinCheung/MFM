import torch
from torchvision.transforms import autoaugment, transforms
from torchvision.transforms.functional import InterpolationMode


class ClassificationPresetTrain:

    def __init__(
        self,
        *,
        crop_size,
        mean=(0.4, 0.4, 0.4),
        std=(0.2, 0.2, 0.2),
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        auto_augment_policy=None,
        random_erase_prob=0.0,
    ):
        trans = [transforms.RandomResizedCrop(crop_size, interpolation=interpolation)]
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            if auto_augment_policy.startswith('ra'):
                tokens = auto_augment_policy.split('_')
                mag, n_mag_bins = int(tokens[1]), int(tokens[2])
                trans.append(autoaugment.RandAugment(
                    interpolation=interpolation,
                    magnitude=mag,
                    num_magnitude_bins=n_mag_bins))
            elif auto_augment_policy == "ta_wide":
                trans.append(autoaugment.TrivialAugmentWide(interpolation=interpolation))
            elif auto_augment_policy == "augmix":
                trans.append(autoaugment.AugMix(interpolation=interpolation))
            else:
                aa_policy = autoaugment.AutoAugmentPolicy(auto_augment_policy)
                trans.append(autoaugment.AutoAugment(policy=aa_policy, interpolation=interpolation))
        trans.extend(
            [
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        if random_erase_prob > 0:
            trans.append(transforms.RandomErasing(p=random_erase_prob))

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)


class ClassificationPresetEval:

    def __init__(
        self,
        *,
        crop_size,
        resize_size=256,
        mean=(0.4, 0.4, 0.4),
        std=(0.2, 0.2, 0.2),
        interpolation=InterpolationMode.BILINEAR,
    ):

        self.transforms = transforms.Compose(
            [
                transforms.Resize(resize_size, interpolation=interpolation),
                transforms.CenterCrop(crop_size),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def __call__(self, img):
        return self.transforms(img)


class MFMPresetTrain:

    def __init__(
        self,
        *,
        crop_size,
        mean=(0.4, 0.4, 0.4),
        std=(0.2, 0.2, 0.2),
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
    ):
        trans = [transforms.RandomResizedCrop(crop_size, interpolation=interpolation)]
        if hflip_prob > 0:
            trans.append(transforms.RandomHorizontalFlip(hflip_prob))
        trans.extend(
            [
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        self.transforms = transforms.Compose(trans)

    def __call__(self, img):
        return self.transforms(img)
