import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import param_init


class BaseModel(nn.Module):
    def forward(self, x, last=False, freeze=False):
        if freeze:
            with torch.no_grad():
                f = self.feature_extractor(x)
                f = f.mean((2, 3))
        else:
            f = self.feature_extractor(x)
            f = f.mean((2, 3))
        if last:
            return self.classifier(f), f
        else:
            return self.classifier(f)

    def logits_with_feature(self, x):
        f = self.feature_extractor(x)
        c = self.classifier(f.mean((2, 3)))
        return c, f

    def update_batch_stats(self, flag):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.update_batch_stats = flag


def conv3x3(i_c, o_c, stride=1, bias=False):
    return nn.Conv2d(i_c, o_c, 3, stride, 1, bias=bias)


class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, channels, momentum=1e-3, eps=1e-3):
        super().__init__(channels)
        self.update_batch_stats = True

    def forward(self, x):
        if self.update_batch_stats or not self.training:
            return super().forward(x)
        else:
            return nn.functional.batch_norm(
                x, None, None, self.weight, self.bias, True, self.momentum, self.eps
            )


def leaky_relu():
    return nn.LeakyReLU(0.1)


class _ShakeShake(nn.Module):
    def __init__(self, branch1, branch2):
        super().__init__()
        self.branch1 = branch1
        self.branch2 = branch2

    def forward(self, x):
        a = self.branch1(x)
        b = self.branch2(x)
        if not self.training:
            return 0.5 * (a + b)
        mu = a.new([a.shape[0]] + [1] * (len(a.shape) - 1)).uniform_()
        mixf = a + mu * (b - a)
        mixb = a + mu[::1] * (b - a)
        return (mixf - mixb).detach() + mixb


class _SkipBranch(nn.Module):
    def __init__(self, branch1, branch2, bn):
        super().__init__()
        self.branch1 = branch1
        self.branch2 = branch2
        self.bn = bn

    def forward(self, x):
        a = self.branch1(x[..., ::2, ::2])
        b = self.branch2(x[..., 1::2, 1::2])
        x = torch.cat([a, b], 1)
        return self.bn(x)


def _branch(filters, channels, stride=1):
    return nn.Sequential(
        nn.ReLU(),
        conv3x3(channels, filters, stride),
        BatchNorm2d(filters),
        nn.ReLU(),
        conv3x3(filters, filters),
        BatchNorm2d(filters)
    )


class _Residual(nn.Module):
    def __init__(self, channels, filters, stride=1):
        super().__init__()
        self.branch = _ShakeShake(
            _branch(channels, filters, stride),
            _branch(channels, filters, stride)
        )

        if stride == 2:
            branch1 = nn.Sequential(nn.ReLU(), nn.Conv2d(channels//2, filters >> 1, 1, bias=False))
            branch2 = nn.Sequential(nn.ReLU(), nn.Conv2d(channels//2, filters >> 1, 1, bias=False))
            bn = BatchNorm2d(filters)
            self.skip = _SkipBranch(branch1, branch2, bn)
        elif channels != filters:
            self.skip = nn.Sequential(
                nn.Conv2d(channels, filters, 1, bias=False),
                BatchNorm2d(filters)
            )

    def forward(self, x):
        return self.branch(x) + self.skip(x)


class ShakeNet(BaseModel):
    """
    Shake-Shake model

    Parameters
    --------
    num_classes: int
        number of classes
    filters: int
        number of filters
    scales: int
        number of scales
    repeat: int
        number of residual blocks per scale
    dropout: float
        dropout ratio (None indicates dropout is unused)
    """
    def __init__(self, num_classes, filters, scales, repeat, dropout=None, *args, **kwargs):
        super().__init__()

        feature_extractor = [conv3x3(3, 16)]
        channels = 16

        for scale, i in itertools.product(range(scales), range(repeat)):
            if i == 0:
                feature_extractor.append(_Residual(channels, filters << scale, stride = 2 if scale else 1))
            else:
                feature_extractor.append(_Residual(channels, filters << scale))

            channels = filters << scale

        self.feature_extractor = nn.Sequential(*feature_extractor)

        classifier = []
        if dropout is not None:
            classifier.append(nn.Dropout(dropout))
        classifier.append(nn.Linear(channels, num_classes))

        param_init(self.modules())
