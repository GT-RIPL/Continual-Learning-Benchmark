import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, droprate=0):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.drop = nn.Dropout(p=droprate) if droprate>0 else None
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        if self.drop is not None:
            out = self.drop(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, droprate=None):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, in_channels=3):
        super(PreActResNet, self).__init__()
        self.in_planes = 64
        last_planes = 512*block.expansion

        self.conv1 = conv3x3(in_channels, 64)
        self.stage1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.stage2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.stage3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.stage4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn_last = nn.BatchNorm2d(last_planes)
        self.last = nn.Linear(last_planes, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x):
        out = self.conv1(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        return out

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = F.relu(self.bn_last(x))
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.logits(x.view(x.size(0), -1))
        return x


class PreActResNet_cifar(nn.Module):
    def __init__(self, block, num_blocks, filters, num_classes=10, droprate=0):
        super(PreActResNet_cifar, self).__init__()
        self.in_planes = 16
        last_planes = filters[2]*block.expansion

        self.conv1 = conv3x3(3, self.in_planes)
        self.stage1 = self._make_layer(block, filters[0], num_blocks[0], stride=1, droprate=droprate)
        self.stage2 = self._make_layer(block, filters[1], num_blocks[1], stride=2, droprate=droprate)
        self.stage3 = self._make_layer(block, filters[2], num_blocks[2], stride=2, droprate=droprate)
        self.bn_last = nn.BatchNorm2d(last_planes)
        self.last = nn.Linear(last_planes, num_classes)

        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()
        """

    def _make_layer(self, block, planes, num_blocks, stride, droprate):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, droprate))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def features(self, x):
        out = self.conv1(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        return out

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        out = self.features(x)
        out = F.relu(self.bn_last(out))
        out = F.avg_pool2d(out, 8)
        out = self.logits(out.view(out.size(0), -1))
        return out


# ResNet for Cifar10/100 or the dataset with image size 32x32

def ResNet20_cifar(out_dim=10):
    return PreActResNet_cifar(PreActBlock, [3 , 3 , 3 ], [16, 32, 64], num_classes=out_dim)

def ResNet56_cifar(out_dim=10):
    return PreActResNet_cifar(PreActBlock, [9 , 9 , 9 ], [16, 32, 64], num_classes=out_dim)

def ResNet110_cifar(out_dim=10):
    return PreActResNet_cifar(PreActBlock, [18, 18, 18], [16, 32, 64], num_classes=out_dim)

def ResNet29_cifar(out_dim=10):
    return PreActResNet_cifar(PreActBottleneck, [3 , 3 , 3 ], [16, 32, 64], num_classes=out_dim)

def ResNet164_cifar(out_dim=10):
    return PreActResNet_cifar(PreActBottleneck, [18, 18, 18], [16, 32, 64], num_classes=out_dim)

def WideResNet_28_2_cifar(out_dim=10):
    return PreActResNet_cifar(PreActBlock, [4, 4, 4], [32, 64, 128], num_classes=out_dim)

def WideResNet_28_2_drop_cifar(out_dim=10):
    return PreActResNet_cifar(PreActBlock, [4, 4, 4], [32, 64, 128], num_classes=out_dim, droprate=0.3)

def WideResNet_28_10_cifar(out_dim=10):
    return PreActResNet_cifar(PreActBlock, [4, 4, 4], [160, 320, 640], num_classes=out_dim)

# ResNet for general purpose. Ex:ImageNet

def ResNet10(out_dim=10):
    return PreActResNet(PreActBlock, [1,1,1,1], num_classes=out_dim)

def ResNet18S(out_dim=10):
    return PreActResNet(PreActBlock, [2,2,2,2], num_classes=out_dim, in_channels=1)

def ResNet18(out_dim=10):
    return PreActResNet(PreActBlock, [2,2,2,2], num_classes=out_dim)

def ResNet34(out_dim=10):
    return PreActResNet(PreActBlock, [3,4,6,3], num_classes=out_dim)

def ResNet50(out_dim=10):
    return PreActResNet(PreActBottleneck, [3,4,6,3], num_classes=out_dim)

def ResNet101(out_dim=10):
    return PreActResNet(PreActBottleneck, [3,4,23,3], num_classes=out_dim)

def ResNet152(out_dim=10):
    return PreActResNet(PreActBottleneck, [3,8,36,3], num_classes=out_dim)