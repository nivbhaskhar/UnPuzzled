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
from torchvision import transforms, utils, models
from torch import nn, optim
from torchvision import datasets, transforms
#from torchvision.utils import make_grid


#import csv
from time import time


import sys
import Checking_adjacency_dataset as cad
get_ipython().run_line_magic('matplotlib', 'inline')




def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def reshape_resnet(no_of_classes, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0
    model_ft = models.resnet18(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    no_of_features = model_ft.fc.in_features
    model_ft.fc = nn.Linear(no_of_features, no_of_classes)    
    return model_ft


def parameters_to_update(model_name, model, feature_extract=False):
    params = model.parameters()
    if model_name=="ResNetFT":
        if feature_extract:
            print("Feature extracting from ResNet - Expect less number of parameters to learn!")
            params = []
            for name,param in model.named_parameters():
                if param.requires_grad == True:
                    params.append(param)
                    print("\t",name)
        else:
            print("Fine tuning ResNet - Expect more number of parameters to learn!")
            for name,param in model.named_parameters():
                if param.requires_grad == True:
                    print("\t",name)
    elif model_name=="FromScratch":
        print("Using FromScratch - Expect more number of parameters to learn!")
        for name,param in model.named_parameters():
                if param.requires_grad == True:
                    print("\t",name)
        
    print(f"No_of_parameters to learn : {len(list(model.parameters()))}")
    return params

