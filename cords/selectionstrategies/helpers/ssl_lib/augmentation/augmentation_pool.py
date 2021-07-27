import random
import torch
import torch.nn.functional as F
import numpy as np

from PIL import ImageOps, ImageEnhance, ImageFilter, Image


"""
For PIL.Image
"""

def autocontrast(x, *args, **kwargs):
    return ImageOps.autocontrast(x.convert("RGB")).convert("RGBA")


def brightness(x, level, magnitude=10, max_level=1.8, *args, **kwargs):
    level = (level / magnitude) * max_level + 0.1
    return ImageEnhance.Brightness(x).enhance(level)


def color(x, level, magnitude=10, max_level=1.8, *args, **kwargs):
    level = (level / magnitude) * max_level + 0.1
    return ImageEnhance.Color(x).enhance(level)


def contrast(x, level, magnitude=10, max_level=1.8, *args, **kwargs):
    level = (level / magnitude) * max_level + 0.1
    return ImageEnhance.Contrast(x).enhance(level)


def equalize(x, *args, **kwargs):
    return ImageOps.equalize(x.convert("RGB")).convert("RGBA")


def identity(x, *args, **kwargs):
    return x


def invert(x, *args, **kwargs):
    return ImageOps.invert(x.convert("RGB")).convert("RGBA")


def posterize(x, level, magnitude=10, max_level=4, *args, **kwargs):
    level = int((level / magnitude) * max_level)
    return ImageOps.posterize(x.convert("RGB"), 4 - level).convert("RGBA")


def rotate(x, level, magnitude=10, max_level=30, *args, **kwargs):
    degree = int((level / magnitude) * max_level)
    if random.random() > 0.5:
        degree = -degree
    return x.rotate(degree)


def sharpness(x, level, magnitude=10, max_level=1.8, *args, **kwargs):
    level = (level / magnitude) * max_level + 0.1
    return ImageEnhance.Sharpness(x).enhance(level)


def shear_x(x, level, magnitude=10, max_level=0.3, *args, **kwargs):
    level = (level / magnitude) * max_level
    if random.random() > 0.5:
        level = -level
    return x.transform(x.size, Image.AFFINE, (1, level, 0, 0, 1, 0))


def shear_y(x, level, magnitude=10, max_level=0.3, *args, **kwargs):
    level = (level / magnitude) * max_level
    if random.random() > 0.5:
        level = -level
    return x.transform(x.size, Image.AFFINE, (1, 0, 0, level, 1, 0))


def solarize(x, level, magnitude=10, max_level=256, *args, **kwargs):
    level = int((level / magnitude) * max_level)
    return ImageOps.solarize(x.convert("RGB"), 256 - level).convert("RGBA")


def translate_x(x, level, magnitude=10, max_level=10, *args, **kwargs):
    level = int((level / magnitude) * max_level)
    if random.random() > 0.5:
        level = -level
    return x.transform(x.size, Image.AFFINE, (1, 0, level, 0, 1, 0))


def translate_y(x, level, magnitude=10, max_level=10, *args, **kwargs):
    level = int((level / magnitude) * max_level)
    if random.random() > 0.5:
        level = -level
    return x.transform(x.size, Image.AFFINE, (1, 0, 0, 0, 1, level))


def cutout(x, level, magnitude=10, max_level=20, *args, **kwargs):
    size = int((level / magnitude) * max_level)
    if size <= 0:
        return x
    w, h = x.size
    upper_coord, lower_coord = _gen_cutout_coord(h, w, size)

    pixels = x.load()
    for i in range(upper_coord[0], lower_coord[0]):
        for j in range(upper_coord[1], lower_coord[1]):
            pixels[i, j] = (127, 127, 127, 0)
    return x


def _gen_cutout_coord(height, width, size):
    height_loc = random.randint(0, height - 1)
    width_loc = random.randint(0, width - 1)

    upper_coord = (max(0, height_loc - size // 2),
                    max(0, width_loc - size // 2))
    lower_coord = (min(height, height_loc + size // 2),
                    min(width, width_loc + size // 2))

    return upper_coord, lower_coord

"""
For torch.Tensor
"""

class TorchCutout:
    def __init__(self, size=16):
        self.size = size

    def __call__(self, img):
        h, w = img.shape[-2:]
        upper_coord, lower_coord  = _gen_cutout_coord(h, w, self.size)

        mask_height = lower_coord[0] - upper_coord[0]
        mask_width = lower_coord[1] - upper_coord[1]
        assert mask_height > 0
        assert mask_width > 0

        mask = torch.ones_like(img)
        zeros = torch.zeros((img.shape[0], mask_height, mask_width))
        mask[:, upper_coord[0]:lower_coord[0], upper_coord[1]:lower_coord[1]] = zeros
        return img * mask

    def __repr__(self):
        return f"TorchCutout(size={self.size})"


class GaussianNoise:
    def __init__(self, std=0.15):
        self.std = std

    def __call__(self, x):
        with torch.no_grad():
            return x + torch.randn_like(x) * self.std

    def __repr__(self):
        return f"GaussianNoise(std={self.std})"


class BatchRandomFlip:
    def __init__(self, flip_prob=0.5):
        self.p = flip_prob

    def __call__(self, x):
        with torch.no_grad():
            return torch.stack([
                torch.flip(img, (-1,))
                if random.random() > self.p
                else img
                for img in x
            ], 0)

    def __repr__(self):
        return f"BatchRandomFlip(flip_prob={self.p})"


class RandomFlip:
    def __init__(self, flip_prob=0.5):
        self.p = flip_prob

    def __call__(self, x):
        if random.random() > self.p:
            return torch.flip(x, (-1,))
        return x

    def __repr__(self):
        return f"RandomFlip(flip_prob={self.p})"


class BatchRandomCrop:
    def __init__(self, padding=4):
        self.pad = padding

    def __call__(self, x):
        with torch.no_grad():
            b, _, h, w = x.shape
            x = F.pad(x, [self.pad for _ in range(4)], mode="reflect")
            left, top = torch.randint(0, 1+self.pad*2, (b,)), torch.randint(0, 1+self.pad*2, (b,))
            return torch.stack([
                img[..., t:t+h, l:l+w]
                for img, t, l in zip(x, left, top)
            ], 0)

    def __repr__(self):
        return f"BatchRandomCrop(padding={self.pad})"


class RandomCrop:
    def __init__(self, padding=4):
        self.pad = padding

    def __call__(self, x):
        with torch.no_grad():
            _, h, w = x.shape
            x = F.pad(x[None], [self.pad for _ in range(4)], mode="reflect")
            left, top = random.randint(0, self.pad*2), random.randint(0, self.pad*2)
            return x[0, :, top:top+h, left:left+w]

    def __repr__(self):
        return f"RandomCrop(padding={self.pad})"


class ZCA:
    def __init__(self, mean, scale):
        self.mean = torch.from_numpy(mean).float()
        self.scale = torch.from_numpy(scale).float()

    def __call__(self, x):
        c, h, w = x.shape
        x = x.reshape(-1)
        x = (x - self.mean) @ self.scale
        return x.reshape(c, h, w)

    def __repr__(self):
        return f"ZCA()"


class GCN:
    """global contrast normalization"""
    def __init__(self, multiplier=55, eps=1e-10):
        self.multiplier = multiplier
        self.eps = eps

    def __call__(self, x):
        x -= x.mean()
        norm = x.norm(2)
        norm[norm < self.eps] = 1
        return self.multiplier * x / norm

    def __repr__(self):
        return f"GCN(multiplier={self.multiplier}, eps={self.eps})"


"""
For numpy.array
"""
def numpy_batch_gcn(images, multiplier=55, eps=1e-10):
    # global contrast normalization
    images = images.astype(np.float)
    images -= images.mean(axis=(1,2,3), keepdims=True)
    per_image_norm = np.sqrt(np.square(images).sum((1,2,3), keepdims=True))
    per_image_norm[per_image_norm < eps] = 1
    return multiplier * images / per_image_norm
