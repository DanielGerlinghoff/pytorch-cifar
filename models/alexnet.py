import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=10, batchnorm=False):
        super(AlexNet, self).__init__()
        self.layer_list = nn.ModuleList()
        self.layer_list.append(nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False))
        self.layer_list.append(nn.BatchNorm2d(64) if batchnorm else nn.Identity())
        self.layer_list.append(nn.ReLU(inplace=True))
        self.layer_list.append(nn.MaxPool2d(kernel_size=2))
        self.layer_list.append(nn.Conv2d(64, 192, kernel_size=3, padding=1, bias=False))
        self.layer_list.append(nn.BatchNorm2d(192) if batchnorm else nn.Identity())
        self.layer_list.append(nn.ReLU(inplace=True))
        self.layer_list.append(nn.MaxPool2d(kernel_size=2))
        self.layer_list.append(nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False))
        self.layer_list.append(nn.BatchNorm2d(384) if batchnorm else nn.Identity())
        self.layer_list.append(nn.ReLU(inplace=True))
        self.layer_list.append(nn.Conv2d(384, 256, kernel_size=3, padding=1, bias=False))
        self.layer_list.append(nn.BatchNorm2d(256) if batchnorm else nn.Identity())
        self.layer_list.append(nn.ReLU(inplace=True))
        self.layer_list.append(nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False))
        self.layer_list.append(nn.BatchNorm2d(256) if batchnorm else nn.Identity())
        self.layer_list.append(nn.ReLU(inplace=True))
        self.layer_list.append(nn.MaxPool2d(kernel_size=2))
        self.layer_list.append(nn.Flatten())
        self.layer_list.append(nn.Linear(256 * 2 * 2, 4096))
        self.layer_list.append(nn.ReLU(inplace=True))
        self.layer_list.append(nn.Linear(4096, 4096))
        self.layer_list.append(nn.ReLU(inplace=True))
        self.layer_list.append(nn.Linear(4096, num_classes))
        self.layer_list.append(nn.LogSoftmax(dim=-1))

    def forward(self, x):
        for layer in self.layer_list:
            x = layer(x)
        return x
