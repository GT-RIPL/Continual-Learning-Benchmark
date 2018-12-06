import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import conv3x3, PreActResNet, PreActResNet_cifar


class SE_PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(SE_PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

        # SE layers
        self.fc1 = nn.Conv2d(planes, planes // 16, kernel_size=1)
        self.fc2 = nn.Conv2d(planes // 16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        # Excitation
        out = out * w  # New broadcasting feature from v0.2!
        out += shortcut
        return out


class SE_PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(SE_PreActBottleneck, self).__init__()
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

        # SE layers
        self.fc1 = nn.Conv2d(self.expansion*planes, self.expansion*planes // 16, kernel_size=1)
        self.fc2 = nn.Conv2d(self.expansion*planes // 16, self.expansion*planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        # Excitation
        out = out * w
        out += shortcut
        return out


# ResNet for Cifar10/100 or the dataset with image size 32x32

def SE_ResNet20_cifar(out_dim=10):
    return PreActResNet_cifar(SE_PreActBlock, [3 , 3 , 3 ], [16, 32, 64], num_classes=out_dim)

def SE_ResNet56_cifar(out_dim=10):
    return PreActResNet_cifar(SE_PreActBlock, [9 , 9 , 9 ], [16, 32, 64], num_classes=out_dim)

def ResNet110_cifar(out_dim=10):
    return PreActResNet_cifar(SE_PreActBlock, [18, 18, 18], [16, 32, 64], num_classes=out_dim)

def SE_ResNet29_cifar(out_dim=10):
    return PreActResNet_cifar(SE_PreActBottleneck, [3 , 3 , 3 ], [16, 32, 64], num_classes=out_dim)

def SE_ResNet164_cifar(out_dim=10):
    return PreActResNet_cifar(SE_PreActBottleneck, [18, 18, 18], [16, 32, 64], num_classes=out_dim)

def SE_WideResNet_28_2_cifar(out_dim=10):
    return PreActResNet_cifar(SE_PreActBlock, [4, 4, 4], [32, 64, 128], num_classes=out_dim)

def SE_WideResNet_28_10_cifar(out_dim=10):
    return PreActResNet_cifar(SE_PreActBlock, [4, 4, 4], [160, 320, 640], num_classes=out_dim)

# ResNet for general purpose. Ex:ImageNet

def SE_ResNet10(out_dim=10):
    return PreActResNet(SE_PreActBlock, [1,1,1,1], num_classes=out_dim)

def SE_ResNet18S(out_dim=10):
    return PreActResNet(SE_PreActBlock, [2,2,2,2], num_classes=out_dim, in_channels=1)

def SE_ResNet18(out_dim=10):
    return PreActResNet(SE_PreActBlock, [2,2,2,2], num_classes=out_dim)

def SE_ResNet34(out_dim=10):
    return PreActResNet(SE_PreActBlock, [3,4,6,3], num_classes=out_dim)

def SE_ResNet50(out_dim=10):
    return PreActResNet(SE_PreActBottleneck, [3,4,6,3], num_classes=out_dim)

def SE_ResNet101(out_dim=10):
    return PreActResNet(SE_PreActBottleneck, [3,4,23,3], num_classes=out_dim)

def SE_ResNet152(out_dim=10):
    return PreActResNet(SE_PreActBottleneck, [3,8,36,3], num_classes=out_dim)

