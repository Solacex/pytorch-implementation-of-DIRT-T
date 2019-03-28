from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, keep_fc=False):
        super(ResNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.keep_fc = keep_fc

        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth in resnet:", depth)

        self.base = ResNet.__factory[depth](pretrained=pretrained)

        if not self.pretrained:
            self.reset_params()

    def forward(self, x):

        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)

        if self.cut_at_pooling:
            return x

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        return x

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)


def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)


def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)
