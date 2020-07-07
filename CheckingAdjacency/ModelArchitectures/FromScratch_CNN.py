#!/usr/bin/env python
# coding: utf-8

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import os

import pprint
import itertools
from collections import defaultdict

# generate random integer values
from random import seed
from random import randint
import numpy as np
from random import sample
import math

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torchvision import transforms, utils
from torch import nn, optim
from torchvision import datasets, transforms
#from torchvision.utils import make_grid


#import csv
from time import time


import sys
import Checking_adjacency_dataset as cad
get_ipython().run_line_magic('matplotlib', 'inline')




class BlockUnit(nn.Module):
    def __init__(self):
        super(BlockUnit, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels = 3, out_channels = 3, kernel_size = (5,5), stride = (1,1),
            padding = (2,2), dilation = 1, groups = 1, bias = False, padding_mode = 'zeros')
        self.conv2 = nn.Conv2d(
            in_channels = 3, out_channels = 3, kernel_size = (5,5), stride = (1,1),
            padding = (2,2), dilation = 1, groups = 1, bias = False, padding_mode = 'zeros')
        self.pool = torch.nn.MaxPool2d(
            kernel_size = (2,2), stride=(2,2), padding=0, dilation=1,
            return_indices=False, ceil_mode=False)
        self.unit = nn.Sequential(
            self.conv1, nn.ReLU(), nn.BatchNorm2d(3),
            self.conv2,  nn.ReLU(), nn.BatchNorm2d(3),
            self.pool)

    def forward(self, x):
        # batchsize x c x h x w > batchsize x c x (h//2) x (w//2)
        return self.unit(x)
        



class FromScratch(nn.Module):
    def __init__(self):
        super(FromScratch, self).__init__()
        units = []
        for i in [1,2,3,4,5,6]:
            units.append(BlockUnit())
        #   h > h/2 > h/4 > h/8 > h/16 > h/32 > h/64
        #   w > w/2 > w/4 > w/8 > w/16 > w/32 > w/64
        # input > 1  > 2  > 3  > 4 > 5 > 6
        self.bigunit = nn.Sequential(*units)

        
        # 27 > 9
        self.fc1 = nn.Linear(27,9)
        torch.nn.init.zeros_(self.fc1.bias)
        self.bn1 =  nn.BatchNorm1d(9)


        # 9 > 2
        self.fc2 = nn.Linear(9,2) 
        torch.nn.init.zeros_(self.fc2.bias)
        self.bn2 =  nn.BatchNorm1d(2)

    def forward(self, x):
        
        #x dim is 3 x 224 x 224
        
        
        #Passing through 6 BlockUnits
        #  224  > 112 > 56 > 28 > 14 > 7 > 3
        # input > 1   > 2  > 3  > 4  > 5 > 6
        x = self.bigunit(x)
        # x dim now (batch_size, 3, 3, 3)
        
        
        #Flattening x
        x = x.view(x.shape[0], -1)
        # x dim now (batch_size, 27)
        
        #Passing through FC1
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.bn1(x)
        # x dim now (batch_size, 9)
        
        #Passing through FC1
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x= self.bn2(x)
        # x dim (batch_size, 2)

        x = nn.LogSoftmax(dim = 1)(x)
        # x dim (batch_size, 2)
        
        return x






