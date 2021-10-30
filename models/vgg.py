'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, batchnorm=False, num_classes=10):
        super(VGG, self).__init__()
        self.layer_list = self._make_layers(cfg[vgg_name], bn=batchnorm)
        self.layer_list.append(nn.Flatten())
        self.layer_list.append(nn.Linear(512, 4096, bias=False))
        self.layer_list.append(nn.ReLU(inplace=True))
        self.layer_list.append(nn.Linear(4096, 4096, bias=False))
        self.layer_list.append(nn.ReLU(inplace=True))
        self.layer_list.append(nn.Linear(4096, num_classes, bias=False))
        self.layer_list.append(nn.LogSoftmax(dim=-1))

    def forward(self, x):
        for layer in self.layer_list:
            x = layer(x)
        return x

    def _make_layers(self, cfg, bn):
        layers = nn.ModuleList()
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False))
                layers.append(nn.BatchNorm2d(x) if bn else nn.Identity())
                layers.append(nn.ReLU(inplace=True))
                in_channels = x
        return layers
