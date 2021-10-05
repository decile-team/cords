import torch
import torchvision.transforms as tt

from . import augmentation_pool as aug_pool
from .rand_augment import RandAugment


class ReduceChannelwithNormalize:
    """ Reduce alpha channel of RGBA """
    def __init__(self, mean, scale, zca):
        self.mean = mean
        self.scale = scale
        self.zca = zca

    def __call__(self, tch_img):
        rgb = tch_img[:3]
        i1, i2 = torch.where(tch_img[3] == 0)
        if self.zca:
            rgb = aug_pool.GCN()(tch_img)
            rgb = aug_pool.ZCA(self.mean, self.scale)(rgb)
        else:
            rgb = tt.functional.normalize(rgb, self.mean, self.scale, True)
        rgb[:, i1, i2] = 0
        return rgb

    def __repr__(self):
        return f"ReduceChannelwithNormalize(mean={self.mean}, scale={self.scale})"


class RGB2RGBA:
    def __call__(self, x):
        return x.convert("RGBA")

    def __repr__(self):
        return "RGB2RGBA()"


class StrongAugmentation:
    """
    Strong augmentation class
    including RandAugment and Cutout
    """
    def __init__(
        self,
        img_size: int,
        mean: list,
        scale: list,
        flip: bool,
        crop: bool,
        alg: str = "fixmatch",
        zca: bool = False,
        cutout: bool = True,
    ):
        augmentations = [tt.ToPILImage()]
        
        if flip:
            augmentations += [tt.RandomHorizontalFlip(p=0.5)]
        if crop:
            augmentations += [tt.RandomCrop(img_size, int(img_size*0.125), padding_mode="reflect")]

        augmentations += [
            RGB2RGBA(),
            RandAugment(alg=alg),
            tt.ToTensor(),
            ReduceChannelwithNormalize(mean, scale, zca)
        ]
        if cutout:
            augmentations += [aug_pool.TorchCutout(16)]

        self.augmentations = tt.Compose(augmentations)

    def __call__(self, img):
        return self.augmentations(img)

    def __repr__(self):
        return repr(self.augmentations)


class WeakAugmentation:
    """
    Weak augmentation class
    including horizontal flip, random crop, and gaussian noise
    """
    def __init__(
        self,
        img_size: int,
        mean: list,
        scale: list,
        flip=True,
        crop=True,
        noise=True,
        zca=False
    ):
        augmentations = [tt.ToPILImage()]
        if flip:
            augmentations.append(tt.RandomHorizontalFlip())
        if crop:
            augmentations.append(tt.RandomCrop(img_size, int(img_size*0.125), padding_mode="reflect"))
        augmentations += [tt.ToTensor()]
        if zca:
            augmentations += [aug_pool.GCN(), aug_pool.ZCA(mean, scale)]
        else:
            augmentations += [tt.Normalize(mean, scale, True)]
        if noise:
            augmentations.append(aug_pool.GaussianNoise())
        self.augmentations = tt.Compose(augmentations)

    def __call__(self, img):
        return self.augmentations(img)

    def __repr__(self):
        return repr(self.augmentations)
