import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten

class ThinningNet(Module):
    def __init__(self):        
        
        super(ThinningNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, stride=1, padding=0)        
        self.relu1 = ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = ReLU(inplace=True)      
      
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = ReLU(inplace=True)
        
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu4 = ReLU(inplace=True)
        
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu5 = ReLU(inplace=True)      
      
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu6 = ReLU(inplace=True)
        
        self.conv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu7 = ReLU(inplace=True)
        
        self.conv8 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu8 = ReLU(inplace=True)      
      
        self.conv9 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.relu9 = ReLU(inplace=True)
        
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, x):
        rep = nn.ReplicationPad2d(4)
        x = rep(x)
        
        x = self.conv1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.relu3(x) 

        x = self.conv4(x)
        x = self.relu4(x)
        
        x = self.conv5(x)
        x = self.relu5(x)
        
        x = self.conv6(x)
        x = self.relu6(x)

        x = self.conv7(x)
        x = self.relu7(x)
        
        x = self.conv8(x)
        x = self.relu8(x)
        
        x = self.conv9(x)
        x = self.relu9(x)
                          
        return self.sigmoid(x-0.7)