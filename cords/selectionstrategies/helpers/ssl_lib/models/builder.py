import numpy as np
from .resnet import ResNet
from .shakenet import ShakeNet
from .cnn13 import CNN13
from .cnn import CNN

def gen_model(name, num_classes, img_size):
    scale =  int(np.ceil(np.log2(img_size)))
    if name == "wrn":
        return ResNet(num_classes, 32, scale, 4)
    elif name == "shake":
        return ShakeNet(num_classes, 32, scale, 4)
    elif name == "cnn13":
        return CNN13(num_classes, 32)
    elif name == 'cnn':
        return CNN(num_classes)
    else:
        raise NotImplementedError