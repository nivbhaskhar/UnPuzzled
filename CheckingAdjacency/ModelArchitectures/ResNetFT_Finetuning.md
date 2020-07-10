# FineTuning ResNet

Recall that we call a tuple of puzzle pieces (P, Q) (order matters!) to be __left-right adjacent__ if when P is placed to the left of Q, P's right edge is adjacent to Q's left edge.

Instead of building a model from scratch, we _finetune_ an exisiting model ResNet18. That is, we take ResNet18's architecture and reshape its last fully connected layer so as to give outputs in the shapes we require. 


We will then train this modified model __ResNetFT__ to solve our _checking_left_right_adjacency_problem_









```python
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
%matplotlib inline





```

# The training data

We use our earlier defined custom  _AdjacencyDataset_. Recall that we generate our data set from the CUB-200 dataset. 

__Input for AdjacencyDataset__
<ul>
    <li>root_dir : the root directory where the CUB-200 images are stored </li>
<li> sq_puzzle_piece_dim : the dimension of the square puzzle piece (recall we cut the original image into uniform square puzzle pieces) </li>
    <li> size_of_buffer : the buffer size for our shuffle_iterator</li>
    <li> model_dim : input size for the model</li>
 </ul>
 
__Output of AdjacencyDataset__
<ul>
    <li> juxtaposed_pieces_torchtensor : cropped (from the middle of the juxtaposed pieces) square rescaled piece with width, height = model_dim </li>
    <li> label : 1 if left-right adjacent, 0 if not</li>
</ul>

Each data point therefore looks like (juxtaposed_pieces_torchtensor, label) the torchtensor has dimensions 3 x model_dim x model_dim (3 because RGB image, so 3 channels)
The label is 1 if the pieces are left-right adjacent else 0



## Loading the dataset and dataloader


```python
my_root_dir = os.getenv("MY_ROOT_DIR")
my_sq_puzzle_piece_dim = 100
my_size_of_buffer = 1000
my_model_dim = 224
my_batch_size = 5
```


```python
train_resnetft_adj_dataset = cad.AdjacencyDataset(my_root_dir, 
                                                      my_sq_puzzle_piece_dim, 
                                                      my_size_of_buffer, my_model_dim)



train_resnetft_adj_dataloader = DataLoader(train_resnetft_adj_dataset, 
                                               my_batch_size)



```

## Sample data point 


```python
juxtaposed_pieces_torchtensor, label = next(iter(train_resnetft_adj_dataloader))
print(juxtaposed_pieces_torchtensor.shape, label.shape)
print(label)
print(torch.min(juxtaposed_pieces_torchtensor), torch.max(juxtaposed_pieces_torchtensor))
print(torch.mean(juxtaposed_pieces_torchtensor.view(-1)))

```

    torch.Size([5, 3, 224, 224]) torch.Size([5])
    tensor([1, 1, 1, 1, 0])
    tensor(0.) tensor(1.)
    tensor(0.4812)


# Reshaping ResNet


```python
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
```


```python
def reshape_resnet(no_of_classes, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0
    model_ft = models.resnet18(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    no_of_features = model_ft.fc.in_features
    model_ft.fc = nn.Linear(no_of_features, no_of_classes)    
    return model_ft

```


```python
def parameters_to_update(model_name, model, feature_extract=False):
    params = list(model.parameters())
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
        
    print(f"No_of_parameters to learn : {len(params)}")
    return params
```

# A test ResNetFT




```python
no_of_classes = 2
#If finetuning, feature_extract = False, else True
feature_extract = False
```


```python
test_model = reshape_resnet(no_of_classes, feature_extract, use_pretrained=True)
```


```python
parameters_to_update("ResNetFT", test_model, feature_extract)
```

    Fine tuning ResNet - Expect more number of parameters to learn!
    	 conv1.weight
    	 bn1.weight
    	 bn1.bias
    	 layer1.0.conv1.weight
    	 layer1.0.bn1.weight
    	 layer1.0.bn1.bias
    	 layer1.0.conv2.weight
    	 layer1.0.bn2.weight
    	 layer1.0.bn2.bias
    	 layer1.1.conv1.weight
    	 layer1.1.bn1.weight
    	 layer1.1.bn1.bias
    	 layer1.1.conv2.weight
    	 layer1.1.bn2.weight
    	 layer1.1.bn2.bias
    	 layer2.0.conv1.weight
    	 layer2.0.bn1.weight
    	 layer2.0.bn1.bias
    	 layer2.0.conv2.weight
    	 layer2.0.bn2.weight
    	 layer2.0.bn2.bias
    	 layer2.0.downsample.0.weight
    	 layer2.0.downsample.1.weight
    	 layer2.0.downsample.1.bias
    	 layer2.1.conv1.weight
    	 layer2.1.bn1.weight
    	 layer2.1.bn1.bias
    	 layer2.1.conv2.weight
    	 layer2.1.bn2.weight
    	 layer2.1.bn2.bias
    	 layer3.0.conv1.weight
    	 layer3.0.bn1.weight
    	 layer3.0.bn1.bias
    	 layer3.0.conv2.weight
    	 layer3.0.bn2.weight
    	 layer3.0.bn2.bias
    	 layer3.0.downsample.0.weight
    	 layer3.0.downsample.1.weight
    	 layer3.0.downsample.1.bias
    	 layer3.1.conv1.weight
    	 layer3.1.bn1.weight
    	 layer3.1.bn1.bias
    	 layer3.1.conv2.weight
    	 layer3.1.bn2.weight
    	 layer3.1.bn2.bias
    	 layer4.0.conv1.weight
    	 layer4.0.bn1.weight
    	 layer4.0.bn1.bias
    	 layer4.0.conv2.weight
    	 layer4.0.bn2.weight
    	 layer4.0.bn2.bias
    	 layer4.0.downsample.0.weight
    	 layer4.0.downsample.1.weight
    	 layer4.0.downsample.1.bias
    	 layer4.1.conv1.weight
    	 layer4.1.bn1.weight
    	 layer4.1.bn1.bias
    	 layer4.1.conv2.weight
    	 layer4.1.bn2.weight
    	 layer4.1.bn2.bias
    	 fc.weight
    	 fc.bias
    No_of_parameters to learn : 62





    <generator object Module.parameters at 0x12527bdd0>




```python
print(test_model.fc.weight)
print(test_model.fc.bias)

```

    Parameter containing:
    tensor([[-0.0189, -0.0148,  0.0311,  ...,  0.0053,  0.0298, -0.0299],
            [-0.0359, -0.0122, -0.0362,  ..., -0.0415, -0.0339, -0.0433]],
           requires_grad=True)
    Parameter containing:
    tensor([ 0.0251, -0.0067], requires_grad=True)



```python
juxtaposed_pieces_torchtensor, label = next(iter(train_resnetft_adj_dataloader))
test_output_for_resnetft = test_model(juxtaposed_pieces_torchtensor)
```


```python
test_output_for_resnetft
```




    tensor([[-0.2232,  0.3393],
            [ 0.4366, -0.1868],
            [-0.4116, -0.6638],
            [ 0.5400,  0.6125],
            [ 0.2978, -0.6460]], grad_fn=<AddmmBackward>)




```python
scores, predictions = torch.max(test_output_for_resnetft, axis = 1)   
```


```python
scores
```




    tensor([ 0.3393,  0.4366, -0.4116,  0.6125,  0.2978], grad_fn=<MaxBackward0>)




```python
predictions
```




    tensor([1, 0, 0, 1, 0])




```python
label
```




    tensor([0, 0, 0, 0, 0])



# Summary

We have reshaped "ResNet18", whose outputs are vectors of the shape (a_0, a_1). We interpret $a_i$ to be the _score_ given by the model for the label being i. We interpret higher scores to mean higher chance of the label in truth being i.

