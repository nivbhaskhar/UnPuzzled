# A Convolutional Neural Network from scratch


Recall that we call a tuple of puzzle pieces (P, Q) (order matters!) to be __left-right adjacent__ if when P is placed to the left of Q, P's right edge is adjacent to Q's left edge.
One idea is to use convolutional neural networks (CNNs). These try to retain the spatial structure of the inputs, which makes them work well on problems with images as inputs. 

We build a CNN network called _FromScratch_ to solve our __checking_left_right_adjacency_problem__.






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
from torchvision import transforms, utils
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




```python
train_from_scratch_adj_dataset = cad.AdjacencyDataset(my_root_dir, 
                                                      my_sq_puzzle_piece_dim, 
                                                      my_size_of_buffer, my_model_dim)



train_from_scratch_adj_dataloader = DataLoader(train_from_scratch_adj_dataset, 
                                               my_batch_size)


```

## Sample data point 



```python
juxtaposed_pieces_torchtensor, label = next(iter(train_from_scratch_adj_dataloader))

```


```python
juxtaposed_pieces_torchtensor.shape, label.shape
```




    (torch.Size([5, 3, 224, 224]), torch.Size([5]))




```python
label
```




    tensor([0, 1, 1, 0, 1])




```python
torch.min(juxtaposed_pieces_torchtensor), torch.max(juxtaposed_pieces_torchtensor)

```




    (tensor(0.), tensor(1.))




```python
torch.mean(juxtaposed_pieces_torchtensor.view(-1))
```




    tensor(0.3728)



# Model architecture of "FromScratch"

## Defining the basic layers 




* __Convolutional filter (CF)__ : 

  A convolutional filter with kernel_size = (5,5) and stride = (1,1) and padding (2,2) and   no dilation. Such a filter turns input
  __(in_channels x input_height x input_width)__ 
  into output __(1 x input_height x input_width)__. 
  That is, this convolutional filter doesn't change the height and width of input 
    
    
    
* __Convolutional layer of shape C__ : 

    A convolutional layer with convolutional filters (CF) and further with no_of_filters  = in_channels. Thus after a convolutional layer of shape C , output dimension = input dimension.
    

* __Maxpool filter (MF)__ : 

  A maxpool filter with kernel_size = (2,2) and stride = (2,2) and padding 0 and no dilation. Such a filter turns input
  __(in_channels x input_height x input_width)__ 
  into output __(in_channels x input_height//2 x input_width//2)__. 
    
    
* __Maxpool layer of shape M__ : 

    A maxpool layer with one maxpool filter MF. 
    
    
* __Batchnorm (B)__ :

  _BatchNorm2d(in_channels)_ turns input tensor __(batchsize x in_channels x input_height x input_width)__  into output __(batchsize x in_channels x input_height x input_width)__. Batchnorms will be applied to the output of each layer so that the outputs are rescaled to fit into a nice (normal) distribution. 
  
  


_A word about dilation_ : Dilation = n makes a pixel (1x1) of the kernel to be n x n, where the original kernel pixel is at the top left, and the rest pixels are empty (or filled with 0). Thus dilation=1 is equivalent to the standard convolution with no dilation.

## Creating test layers

### A test Convolution layer of Shape C 
From PyTorch documentation, the following creates a convolution layer

_torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=True, padding_mode='zeros')_

where 

   * kernel and filter mean the same thing
   * out_channels = number of filters, that is, depth of your output 
   * in_channels = depth of your input 
   
     _Note that each filter has dimension in_channels x filter_height x filter_width. Thus when you run a filter over your input image, the filter's depth is the same as your image depth. So filter can move only across the width and height dimensions (2-D convolution)_
     
     
  * The parameters kernel_size, stride, padding, dilation can either be a single int (in which case the same value is used for the height and width dimension) or a tuple of two ints (in which case, the first int is used for the height dimension, and the second int for the width dimension)


```python
shape_C_layer = nn.Conv2d(
    in_channels = 3,
    out_channels = 3,
    kernel_size = (5,5),
    stride = (1,1),
    padding = (2,2),
    dilation = 1,
    groups = 1,
    bias = False,
    padding_mode = 'zeros')
```

__Input to shape C layer__ : A (batchsize x in_channels x input_height x input_width) tensor


```python
test_input_to_shape_C_layer = torch.rand(my_batch_size,3,5,6)
```


```python
test_input_to_shape_C_layer.shape
```




    torch.Size([5, 3, 5, 6])



__Output of shape C layer__ : A (batchsize x out_channels x output_height x output_width) tensor where output_height=input_height and output_width=input_width


```python
test_output_for_shape_C_layer = shape_C_layer(test_input_to_shape_C_layer)
```


```python
test_output_for_shape_C_layer.shape
```




    torch.Size([5, 3, 5, 6])



### A test Maxpool layer of Shape M


From PyTorch documentation, the following creates a maxpool layer

_torch.nn.MaxPool2d(kernel_size, stride, padding, dilation=1, return_indices=False, ceil_mode=False)_

where 

   * kernel and filter mean the same thing
   
     _Note that each maxpool filter has dimension 1 x filter_height x filter_width. Thus when you run a filter over your input image, it will run across each input depth layer once_
     
     
  * The parameters kernel_size, stride, padding, dilation can either be a single int (in which case the same value is used for the height and width dimension) or a tuple of two ints (in which case, the first int is used for the height dimension, and the second int for the width dimension)



```python
shape_M_layer = torch.nn.MaxPool2d(
    kernel_size = (2,2),
    stride=(2,2),
    padding=0,
    dilation=1, 
    return_indices=False,
    ceil_mode=False)
```

__Input to shape M layer__ : A (batchsize x in_channels x input_height x input_width) tensor


```python
test_input_to_shape_M_layer = torch.rand(my_batch_size,3,5,10)
```


```python
test_input_to_shape_M_layer.shape
```




    torch.Size([5, 3, 5, 10])



__Output of shape M layer__ : A (batchsize x out_channels x output_height x output_width) tensor where out_channels = in_channels, output_height=input_height//2 and output_width=input_width//2


```python
test_output_for_shape_M_layer = shape_M_layer(test_input_to_shape_M_layer)
```


```python
test_output_for_shape_M_layer.shape
```




    torch.Size([5, 3, 2, 5])



## Defining a BlockUnit

A BlockUnit U is built as follows:

$$C_1\to ReLU \to B_1 \to C_2 \to ReLU \to B_2 \to M_1$$

Here the $C_i$ are convolutional layers of shape C, the $B_i$ are BatchNorm layers and $M_1$ is a maxpool layer of shape M


```python
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
        


```

### A test BlockUnit

__Input to BlockUnit__ : A (batchsize x in_channels x input_height x input_width) tensor



```python
test_input_to_blockunit = torch.rand(my_batch_size,3,5,10)
test_input_to_blockunit.shape
```




    torch.Size([5, 3, 5, 10])




__Output of BlockUnit__ : A (batchsize x out_channels x output_height x output_width) tensor where out_channels = in_channels, output_height=input_height//2 and output_width=input_width//2




```python
test_output_for_blockunit = BlockUnit()(test_input_to_blockunit)
test_output_for_blockunit.shape
```




    torch.Size([5, 3, 2, 5])



## FromScratch: The final model 


Our final model consists of some BlockUnits followed by two fully connected layers. More precisely, it is built as follows:


$$I\to U_1\to U_2\to U_3 \to U_4 \to U_5 \to U_6 \to FC_1 \to ReLU \to B_1 \to FC_2 \to B_2\to ReLU\to SoftMax$$


Here the $U_i$ are the BlockUnits, the $B_i$ are BatchNorm layers and the $FC_i$ are the fully connected layers. 


_In the code, we’ll actually apply and output the logsoftmax instead of the softmax as it is easier to handle sum of log of small numbers than products of small numbers_

__Input-output dimensions__

Let b =  batchsize, c = channels, h = height, w = width

* Input from dataloader will be of dimension b,c,h,w = (b,3,224,224)
* After passing through U_is, it will become (b, 3, 3, 3) 




```python
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
```

### A test model



```python
juxtaposed_pieces_torchtensor, label = next(iter(train_from_scratch_adj_dataloader))
test_output_for_fromscratch = FromScratch()(juxtaposed_pieces_torchtensor)
```


```python
#log softmax outputs from the model
test_output_for_fromscratch
```




    tensor([[-0.5749, -0.8273],
            [-0.5749, -0.8273],
            [-0.0619, -2.8138],
            [-2.3423, -0.1010],
            [-1.3246, -0.3091]], grad_fn=<LogSoftmaxBackward>)




```python
predicted_probabilities=torch.exp(test_output_for_fromscratch)
```


```python
predicted_probabilities
```




    tensor([[0.5628, 0.4372],
            [0.5628, 0.4372],
            [0.9400, 0.0600],
            [0.0961, 0.9039],
            [0.2659, 0.7341]], grad_fn=<ExpBackward>)




```python
probabilities, predictions = torch.max(predicted_probabilities, axis = 1)   


```


```python
probabilities
```




    tensor([0.5628, 0.5628, 0.9400, 0.9039, 0.7341], grad_fn=<MaxBackward0>)




```python
predictions
```




    tensor([0, 0, 0, 1, 1])




```python
label
```




    tensor([0, 0, 0, 1, 1])



# Summary

We have built "FromScratch" our CNN model, whose outputs are vectors of the shape (a_0, a_1). We interpret $a_i = log p_i$, where $p_i$ is the model-predicted probability of the label being i (i=0 or 1)




```python
print(FromScratch())
```

    FromScratch(
      (bigunit): Sequential(
        (0): BlockUnit(
          (conv1): Conv2d(3, 3, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
          (conv2): Conv2d(3, 3, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
          (pool): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
          (unit): Sequential(
            (0): Conv2d(3, 3, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
            (1): ReLU()
            (2): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (3): Conv2d(3, 3, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
            (4): ReLU()
            (5): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (6): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
          )
        )
        (1): BlockUnit(
          (conv1): Conv2d(3, 3, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
          (conv2): Conv2d(3, 3, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
          (pool): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
          (unit): Sequential(
            (0): Conv2d(3, 3, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
            (1): ReLU()
            (2): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (3): Conv2d(3, 3, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
            (4): ReLU()
            (5): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (6): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
          )
        )
        (2): BlockUnit(
          (conv1): Conv2d(3, 3, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
          (conv2): Conv2d(3, 3, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
          (pool): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
          (unit): Sequential(
            (0): Conv2d(3, 3, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
            (1): ReLU()
            (2): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (3): Conv2d(3, 3, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
            (4): ReLU()
            (5): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (6): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
          )
        )
        (3): BlockUnit(
          (conv1): Conv2d(3, 3, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
          (conv2): Conv2d(3, 3, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
          (pool): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
          (unit): Sequential(
            (0): Conv2d(3, 3, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
            (1): ReLU()
            (2): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (3): Conv2d(3, 3, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
            (4): ReLU()
            (5): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (6): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
          )
        )
        (4): BlockUnit(
          (conv1): Conv2d(3, 3, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
          (conv2): Conv2d(3, 3, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
          (pool): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
          (unit): Sequential(
            (0): Conv2d(3, 3, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
            (1): ReLU()
            (2): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (3): Conv2d(3, 3, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
            (4): ReLU()
            (5): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (6): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
          )
        )
        (5): BlockUnit(
          (conv1): Conv2d(3, 3, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
          (conv2): Conv2d(3, 3, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
          (pool): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
          (unit): Sequential(
            (0): Conv2d(3, 3, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
            (1): ReLU()
            (2): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (3): Conv2d(3, 3, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
            (4): ReLU()
            (5): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (6): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, dilation=1, ceil_mode=False)
          )
        )
      )
      (fc1): Linear(in_features=27, out_features=9, bias=True)
      (bn1): BatchNorm1d(9, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (fc2): Linear(in_features=9, out_features=2, bias=True)
      (bn2): BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )



```python

```
