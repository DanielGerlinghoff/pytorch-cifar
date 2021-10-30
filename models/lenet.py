'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.layer_list = nn.ModuleList()

        self.layer_list.append(nn.Conv2d(1, 6, 5, bias=False))
        self.layer_list.append(nn.ReLU())
        self.layer_list.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer_list.append(nn.Conv2d(6, 16, 5, bias=False))
        self.layer_list.append(nn.ReLU())
        self.layer_list.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer_list.append(nn.Conv2d(16, 120, 5, bias=False))
        self.layer_list.append(nn.ReLU())

        self.layer_list.append(nn.Flatten())

        self.layer_list.append(nn.Linear(120, 84, bias=False))
        self.layer_list.append(nn.ReLU())

        self.layer_list.append(nn.Linear(84, num_classes, bias=False))
        self.layer_list.append(nn.LogSoftmax(dim=-1))

    def forward(self, x):
        for layer in self.layer_list:
            x = layer(x)
        return x
