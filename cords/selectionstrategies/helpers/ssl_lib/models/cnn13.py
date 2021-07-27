import torch.nn as nn
from .utils import leaky_relu, conv3x3, BatchNorm2d, BaseModel


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

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
