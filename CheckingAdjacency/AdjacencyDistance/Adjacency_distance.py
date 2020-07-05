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
#from pylab import array
from random import sample
import math

#pytorch modules
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torchvision import transforms, utils

import sys






def display_edge_extraction(juxtaposed_pieces_torchtensor, width):
    #batchsize x channel x height x width
    check = width % 2
    assert (check==0), "Model dim is not even"
    
    #Get the first piece in a batch
    piece = juxtaposed_pieces_torchtensor[0, :, :, :]
    
    #Extract left and right edges around middle of width 10px
    thickened_right_edge = piece[:,:,(width//2)-10:(width//2)]
    thickened_left_edge = piece[:,:,(width//2):(width//2)+10]

    #Display the extracted edges
    my_dpi = 100
    fig = plt.figure(dpi = my_dpi)
    
    print(f"Piece of size {piece.size()}")
    piece_image = transforms.ToPILImage()(piece)
    ax=fig.add_subplot(222)
    ax.imshow(piece_image)
    ax.title.set_text('Piece')
    plt.axis('off')

    print(f"Thickened right edge of size {thickened_right_edge.size()}")
    ax=fig.add_subplot(223)
    right_edge_image = transforms.ToPILImage()(thickened_right_edge)
    ax.imshow(right_edge_image)
    ax.title.set_text('Right edge')
    plt.axis('off')


    print(f"Thickened left edge of size {thickened_left_edge.size()}")
    ax=fig.add_subplot(224)
    left_edge_image = transforms.ToPILImage()(thickened_left_edge)
    ax.imshow(left_edge_image)
    ax.title.set_text('Left edge')
    plt.axis('off')
    
    fig.tight_layout()
    plt.show()
    print("*****************")
    
    



def adjacency_dist(juxtaposed_pieces_torchtensor, width):
    #juxtaposed_pieces_torchtensor = batchsize x channel x height x width
    check = width % 2
    assert (check==0), "Model dim is not even"
    right_edges = juxtaposed_pieces_torchtensor[:, :, :, (width//2)-1]
    left_edges = juxtaposed_pieces_torchtensor[:, :, :, (width//2)]
    differences = left_edges-right_edges
    distances = torch.norm(differences, p='fro', dim=(1,2))
    return distances
  




