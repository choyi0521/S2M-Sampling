import torch
import torch.nn as nn
from models.mobilenetv2 import MobileNetV2_32, MobileNetV2_64, ClassifierWrapper


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


def get_cls(dataset_name, d_cond_mtd):
    if dataset_name == 'cifar7to3':
        num_classes=3
        net = MobileNetV2_32(num_classes=num_classes)
    elif dataset_name == 'celeba7to3':
        num_classes=3
        net = MobileNetV2_64(num_classes=num_classes)
    else:
        raise NotImplementedError
    
    net.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
    net.bn2 = nn.BatchNorm2d(1280)
    net.linear = Identity()
    net = ClassifierWrapper(net, 1280, num_classes, d_cond_mtd)

    return net
