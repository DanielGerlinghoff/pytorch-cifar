'''CNN by Fang et al. in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

class Fang(nn.Module):
    def __init__(self, num_classes=10):
        super(Fang, self).__init__()
        self.layer_list = nn.ModuleList()

        self.layer_list.append(nn.Conv2d(1, 32, 3, bias=False))
        self.layer_list.append(nn.ReLU())
        self.layer_list.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer_list.append(nn.Conv2d(32, 32, 3, bias=False))
        self.layer_list.append(nn.ReLU())
        self.layer_list.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer_list.append(nn.Flatten())

        self.layer_list.append(nn.Linear(800, 256, bias=False))
        self.layer_list.append(nn.ReLU())

        self.layer_list.append(nn.Linear(256, num_classes, bias=False))
        self.layer_list.append(nn.LogSoftmax(dim=-1))

    def forward(self, x):
        for layer in self.layer_list:
            x = layer(x)
        return x
