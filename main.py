import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import os

from models import *
from utils import progress_bar


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc    = 0
start_epoch = 0

# Data
print('==> Preparing data..')
transform_mnist28 = transforms.Compose([
    transforms.ToTensor()])
transform_mnist32 = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()])
transform_cifar_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()])
transform_cifar_test = transforms.Compose([
    transforms.ToTensor()])

# trainset = torchvision.datasets.mnist.MNIST('../data', train=True, download=True, transform=transform_mnist28)
# testset  = torchvision.datasets.mnist.MNIST('../data', train=False, download=True, transform=transform_mnist28)
# trainset = torchvision.datasets.mnist.MNIST('../data', train=True, download=True, transform=transform_mnist32)
# testset  = torchvision.datasets.mnist.MNIST('../data', train=False, download=True, transform=transform_mnist32)
# trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_cifar_train)
# testset  = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_cifar_test)
trainset = torchvision.datasets.CIFAR100(root='../data', train=True, download=True, transform=transform_cifar_train)
testset  = torchvision.datasets.CIFAR100(root='../data', train=False, download=True, transform=transform_cifar_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader  = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Model
print('==> Building model..')
# net = LeNet(num_classes=10)
# net = Fang(num_classes=10)
# net = AlexNet(batchnorm=True, num_classes=10)
net = VGG('VGG11', batchnorm=True, num_classes=100)

net = net.to(device)
if device == 'cuda':
    cudnn.benchmark = True

for mod in net.modules():
    if type(mod) is nn.Conv2d:
        nn.init.xavier_uniform_(mod.weight)

# Training
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(net.state_dict(), './checkpoint/ckpt.pth')
        best_acc = acc


# Execution
for epoch in range(start_epoch, start_epoch+200):
    train(epoch)
    test(epoch)
    scheduler.step()

print("Best val. accuracy: ", best_acc)
