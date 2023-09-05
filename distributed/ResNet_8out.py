'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=1)
        self.layer5 = self._make_layer(block, 256, num_blocks[4], stride=1)
        self.layer6 = self._make_layer(block, 256, num_blocks[5], stride=1)
        self.layer7 = self._make_layer(block, 256, num_blocks[6], stride=1)
        self.layer8 = self._make_layer(block, 512, num_blocks[7], stride=2)
        self.fc1 = nn.Linear(64 * block.expansion, num_classes)
        self.fc2 = nn.Linear(128 * block.expansion, num_classes)
        self.fc3 = nn.Linear(256 * block.expansion, num_classes)
        self.fc4 = nn.Linear(256 * block.expansion, num_classes)
        self.fc5 = nn.Linear(256 * block.expansion, num_classes)
        self.fc6 = nn.Linear(256 * block.expansion, num_classes)
        self.fc7 = nn.Linear(256 * block.expansion, num_classes)
        self.fc8 = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        ex1 = out
        ex1 = F.avg_pool2d(ex1, 20)
        ex1 = ex1.view(ex1.size(0), -1)
        ex1 = self.fc1(ex1)

        out = self.layer2(out)
        ex2 = out
        ex2 = F.avg_pool2d(ex2, 10)
        ex2 = ex2.view(ex2.size(0), -1)
        ex2 = self.fc2(ex2)

        out = self.layer3(out)
        ex3 = out
        ex3 = F.avg_pool2d(ex3, 8)
        ex3 = ex3.view(ex3.size(0), -1)
        ex3 = self.fc3(ex3)

        out = self.layer4(out)
        ex4 = out
        ex4 = F.avg_pool2d(ex4, 8)
        ex4 = ex4.view(ex4.size(0), -1)
        ex4 = self.fc4(ex4)

        out = self.layer5(out)
        ex5 = out
        ex5 = F.avg_pool2d(ex5, 8)
        ex5 = ex5.view(ex5.size(0), -1)
        ex5 = self.fc5(ex5)

        out = self.layer6(out)
        ex6 = out
        ex6 = F.avg_pool2d(ex6, 8)
        ex6 = ex6.view(ex6.size(0), -1)
        ex6 = self.fc6(ex6)

        out = self.layer7(out)
        ex7 = out
        ex7 = F.avg_pool2d(ex7, 8)
        ex7 = ex7.view(ex7.size(0), -1)
        ex7 = self.fc7(ex7)

        out = self.layer8(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc8(out)

        return out, ex1, ex2, ex3, ex4, ex5, ex6, ex7


# def ResNet18(num_classes=1000):
#     return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
#
#
# def ResNet34(num_classes=1000):
#     return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)
#
#
# def ResNet50(num_classes=1000):
#     return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def ResNet101(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 5, 4, 5, 4, 5, 3], num_classes)


def ResNet152(num_classes=1000):
    return ResNet(Bottleneck, [3, 8, 7, 7, 7, 7, 8, 3], num_classes)


def test():
    net = ResNet101()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()
