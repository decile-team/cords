import torch.nn as nn
import torch
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


class CNN13(BaseModel):
    """
    13-layer CNN

    Parameters
    --------
    num_classes: int
        number of classes
    filters: int
        number of filters
    """
    def __init__(self, num_classes, filters, *args, **kwargs):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            conv3x3(3, filters, bias=True),
            leaky_relu(),
            BatchNorm2d(filters),
            conv3x3(filters, filters, bias=True),
            leaky_relu(),
            BatchNorm2d(filters),
            conv3x3(filters, filters, bias=True),
            leaky_relu(),
            BatchNorm2d(filters),
            nn.MaxPool2d(2, 2),
            conv3x3(filters, 2*filters, bias=True),
            leaky_relu(),
            BatchNorm2d(2*filters),
            conv3x3(2*filters, 2*filters, bias=True),
            leaky_relu(),
            BatchNorm2d(2*filters),
            conv3x3(2*filters, 2*filters, bias=True),
            leaky_relu(),
            BatchNorm2d(2*filters),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(2*filters, 4*filters, 3),
            leaky_relu(),
            BatchNorm2d(4*filters),
            nn.Conv2d(4*filters, 2*filters, 1, bias=False),
            leaky_relu(),
            BatchNorm2d(2*filters),
            nn.Conv2d(2*filters, filters, 1, bias=False),
            leaky_relu(),
            BatchNorm2d(filters)
        )

        self.classifier = nn.Linear(filters, num_classes)
        param_init(self.modules())

        # for m in self.modules():
        #     if isinstance(m, (nn.Conv2d, nn.Linear)):
        #         nn.init.xavier_normal_(m.weight)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)

