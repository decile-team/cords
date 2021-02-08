import torch
import torch.nn as nn
import math

## the model definition
# see HeKaiming's implementation using torch:
# https://github.com/KaimingHe/resnet-1k-layers/blob/master/README.md
class Bottleneck(nn.Module):
    expansion = 4  # # output cahnnels / # input channels

    def __init__(self, inplanes, outplanes, stride=1):
        assert outplanes % self.expansion == 0
        super(Bottleneck, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.bottleneck_planes = outplanes / self.expansion
        self.stride = stride

        self._make_layer()

    def _make_layer(self):
        # conv 1x1
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.conv1 = nn.Conv2d(self.inplanes, self.bottleneck_planes,
                               kernel_size=1, stride=self.stride, bias=False)
        # conv 3x3
        self.bn2 = nn.BatchNorm2d(self.bottleneck_planes)
        self.conv2 = nn.Conv2d(self.bottleneck_planes, self.bottleneck_planes,
                               kernel_size=3, stride=1, padding=1, bias=False)
        # conv 1x1
        self.bn3 = nn.BatchNorm2d(self.bottleneck_planes)
        self.conv3 = nn.Conv2d(self.bottleneck_planes, self.outplanes, kernel_size=1,
                               stride=1)
        if self.inplanes != self.outplanes:
            self.shortcut = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=1,
                                      stride=self.stride, bias=False)
        else:
            self.shortcut = None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        # we do pre-activation
        out = self.relu(self.bn1(x))
        out = self.conv1(out)

        out = self.relu(self.bn2(out))
        out = self.conv2(out)

        out = self.relu(self.bn3(out))
        out = self.conv3(out)

        if self.shortcut is not None:
            residual = self.shortcut(residual)

        out += residual
        return out


class ResNet(nn.Module):
    def __init__(self, block, depth, output_classes=1000):
        assert (depth - 2) % 9 == 0  # 164 or 1001
        super(ResNet, self).__init__()
        n = (depth - 2) / 9
        nstages = [16, 64, 128, 256]
        # one conv at the beginning (spatial size: 32x32)
        self.conv1 = nn.Conv2d(3, nstages[0], kernel_size=3, stride=1,
                               padding=1, bias=False)

        # use `block` as unit to construct res-net
        # Stage 0 (spatial size: 32x32)
        self.layer1 = self._make_layer(block, nstages[0], nstages[1], n)
        # Stage 1 (spatial size: 32x32)
        self.layer2 = self._make_layer(block, nstages[1], nstages[2], n, stride=2)
        # Stage 2 (spatial size: 16x16)
        self.layer3 = self._make_layer(block, nstages[2], nstages[3], n, stride=2)
        # Stage 3 (spatial size: 8x8)
        self.bn = nn.BatchNorm2d(nstages[3])
        self.relu = nn.ReLU(inplace=True)
        # classifier
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(nstages[3], output_classes)

        # weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, inplanes, outplanes, nstage, stride=1):
        layers = []
        layers.append(block(inplanes, outplanes, stride))
        for i in range(1, nstage):
            layers.append(block(outplanes, outplanes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x, last=False):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(self.bn(x))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        if last:
            return out, x
        else:
            return out


def ResNet164(output_classes):
    model = ResNet(Bottleneck, 164, output_classes)
    return model