import torch, math, copy
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import warnings
from deepvi.arch import CNN
warnings.filterwarnings("ignore")
global use_cuda
use_cuda = torch.cuda.is_available()

def train(epochs, model, criterion, optimizer, train_loader, test_loader, scheduler=None):
    train_errs = []
    test_errs = []
    for epoch in range(epochs):
        train_err = train_epoch(model, criterion, optimizer, train_loader)
        test_err = test(model, test_loader)
        print('Epoch {:03d}/{:03d}, Error: {} || {}'.format(epoch, epochs, train_err, test_err))
        train_errs.append(train_err)
        test_errs.append(test_err)
        if scheduler is not None: scheduler.step()
    return train_errs, test_errs
    
def train_epoch(model, criterion, optimizer, loader):
    total_correct = 0.
    total_samples = 0.
    
    for batch_idx, (data, target) in enumerate(loader):
        
        if use_cuda:
          data, target = data.cuda(), target.cuda()

        # insert code to get the model outputs and compute the loss (criterion)
        optimizer.zero_grad()
        outputs = model.forward(data)
        loss = criterion(outputs, target).backward()

        _, predicted = torch.max(outputs.data, 1)

        total_correct += (predicted == target).sum().item()
        total_samples += target.size(0)
        
        # insert code to update the parameters using optimizer
        optimizer.step()

    return 1 - total_correct/total_samples
    
def test(model, loader):
    total_correct = 0.
    total_samples = 0.

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):

          if use_cuda:
            data, target = data.cuda(), target.cuda()

            # insert code to get the model outputs
            outputs = model.forward(data)
            
            _, predicted = torch.max(outputs.data, 1)

            total_correct += (predicted == target).sum().item()
            total_samples += target.size(0)

    return 1 - total_correct/total_samples

def plot_err(train_errs, test_errs):
    plt.xlabel("epochs")
    plt.ylabel("error (%)")
    plt.plot(np.arange(len(train_errs)), train_errs, color='red')
    plt.plot(np.arange(len(test_errs)), test_errs, color='blue')
    plt.legend(['train error', 'test error'], loc='upper right')
    plt.tight_layout()
    plt.show()
    plt.clf()


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
sampler=torch.utils.data.sampler.SubsetRandomSampler(range(256))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, sampler=sampler)

test_dataset = datasets.MNIST("data", train=False, download=True, transform=transform)
sampler=torch.utils.data.sampler.SubsetRandomSampler(range(2048))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, sampler=sampler)


criterion = torch.nn.CrossEntropyLoss()
network1 = CNN(128)
if use_cuda:
  network1.cuda()
optimizer = torch.optim.SGD(network1.parameters(), lr=0.0002, weight_decay=1e-2, momentum=0.5, nesterov=True)
train_errs, test_errs = train(40, network1, criterion, optimizer, train_loader, test_loader)
plot_err(train_errs, test_errs)

network2 = CNN(128)
if use_cuda:
  network2.cuda()
optimizer = torch.optim.SGD(network2.parameters(), lr=1.0, weight_decay=1e-2, momentum=0.5, nesterov=True)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 15, gamma=0.1)
train_errs, test_errs = train(40, network2, criterion, optimizer, train_loader, test_loader, scheduler)
plot_err(train_errs, test_errs)
