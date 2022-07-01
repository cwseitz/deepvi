import torch, math, copy
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
global use_cuda
use_cuda = torch.cuda.is_available()

class CNN(nn.Module):
    def __init__(self, w):
        super(CNN, self).__init__()
        
        self.w=w
        self.conv1=nn.Conv2d(1,w,5,2,2)
        self.conv2=nn.Conv2d(w,2*w,5,2,2)
        self.conv3=nn.Conv2d(2*w,4*w,5,2,2)
        self.conv4=nn.Conv2d(4*w,8*w,5,2,2)
        
        self.bn1=nn.BatchNorm2d(w)
        self.bn2=nn.BatchNorm2d(2*w)
        self.bn3=nn.BatchNorm2d(4*w)
        self.bn4=nn.BatchNorm2d(8*w)
        self.bn5=nn.BatchNorm1d(2048)
        
        self.fc1 = nn.Linear(4096,2048)
        self.fc2 = nn.Linear(2048,10)
        
    def forward(self, input):
        
        output = F.relu(self.bn1(self.conv1(input)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = F.relu(self.bn3(self.conv3(output)))
        output = F.relu(self.bn4(self.conv4(output)))
        output = output.view(-1, 8*self.w*2*2)
        output = F.relu(self.bn5(self.fc1(output)))
        output = self.fc2(output)
    
        return output


