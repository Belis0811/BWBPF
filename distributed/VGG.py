import torch
import torch.nn as nn

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super(VGGBlock, self).__init__()
        # print(out_channels/4)
        self.block = self._make_layer(in_channels=in_channels, out_channels=out_channels, num_layers=num_layers)
        self.classifier = nn.Linear(out_channels, 10)
        kernel_size = 0
        if out_channels == 64:
            kernel_size = 16
        elif out_channels == 128:
            kernel_size = 8
        elif out_channels == 256:
            kernel_size = 4
        else:
            kernel_size = 2

        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=1)  # Add AvgPool2d layer

    def _make_layer(self, in_channels, out_channels, num_layers):
        layers = []
        for _ in range(num_layers):
            layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                       nn.BatchNorm2d(out_channels),
                       nn.ReLU(inplace=True)]
            in_channels = out_channels
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if x.shape[1] == 512:
            out = out.view(out.size(0), -1)
            out = self.classifier(out)
            return out, out
        identity = out
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        # print(identity.shape, out.shape)
        return out, identity


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.layer1 = VGGBlock(3, 64, 2)
        self.layer2 = VGGBlock(64, 128, 2)
        self.layer3 = VGGBlock(128, 256, 4)
        self.layer4 = VGGBlock(256, 512, 4)
        self.layer5 = VGGBlock(512, 512, 4)

    def forward(self, x):
        out1, out = self.layer1(x)

        out2, out = self.layer2(out)

        out3, out = self.layer3(out)
        out4, out = self.layer4(out)
        out5, _ = self.layer5(out)
        return out5, out1, out2, out3, out4


def test():
    # Example usage
    vgg = VGG('VGG16')
    print(vgg)
