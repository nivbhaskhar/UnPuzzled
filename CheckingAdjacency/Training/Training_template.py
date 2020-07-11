#!/usr/bin/env python
# coding: utf-8

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import os

import pprint
import itertools
from collections import defaultdict
#from collections import OrderedDict

# generate random integer values
from random import seed
from random import randint
import numpy as np
#from pylab import array
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

from Checking_adjacency_dataset import *
from FromScratch_CNN import *
from ResNetFT_Finetuning import *

from torch.utils.tensorboard import SummaryWriter
import sys


def get_dataset_input():
    sq_puzzle_piece_dim = int(input("Enter sq_puzzle_piece_dim "))
    size_of_buffer = int(input("Enter size_of_shuffle_buffer "))
    model_dim = int(input("Enter model_dim "))
    batch_size = int(input("Enter batch_size "))
    return sq_puzzle_piece_dim, size_of_buffer, model_dim, batch_size


def set_dataset_input(default=True):
    if default:
        sq_puzzle_piece_dim = 100
        size_of_buffer = 1000
        model_dim = 224
        batch_size = 20
        
    else:
        sq_puzzle_piece_dim, size_of_buffer, model_dim, batch_size = get_dataset_input()
    return sq_puzzle_piece_dim, size_of_buffer, model_dim, batch_size


def create_dataloaders(root_dir,val_dir, sq_puzzle_piece_dim,
                       size_of_buffer, model_dim,batch_size):
                        
        
    train_adjacency_dataset = AdjacencyDataset(root_dir,sq_puzzle_piece_dim, size_of_buffer, model_dim)
    train_adjacency_dataloader = DataLoader(train_adjacency_dataset, batch_size)
    val_adjacency_dataset = AdjacencyDataset(val_dir, sq_puzzle_piece_dim, size_of_buffer, model_dim)
    val_adjacency_dataloader = DataLoader(val_adjacency_dataset, batch_size)
    dataloaders = {'Training':train_adjacency_dataloader , 'Validation': val_adjacency_dataloader}
    return dataloaders


def save_checkpoint(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: boolean which tells you if this is the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    # if it is the best model till that point (min validation loss till that point)
    if is_best:
        torch.save(state, best_model_path)
    else:
        torch.save(state, checkpoint_path)
   


def load_checkpoint(checkpoint_path, model, optimizer):
    """
    checkpoint_path: path where checkpoint was saved
    model: model into which we want to load our checkpoint     
    optimizer: optimizer into which we want to load our checkpoint
    """

    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    min_validation_loss = checkpoint['min_validation_loss']
    
    return model, optimizer, checkpoint['epoch'], min_validation_loss

def load_checkpoint_gpu(checkpoint_path, model, optimizer, GpuAvailable):
    """
    checkpoint_path: path where checkpoint was saved
    model: model into which we want to load our checkpoint     
    optimizer: optimizer into which we want to load our checkpoint
    """
    if GpuAvailable:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,map_location=torch.device("cpu"))
    
    model.load_state_dict(checkpoint['state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    min_validation_loss = checkpoint['min_validation_loss']
    
    return model, optimizer, checkpoint['epoch'], min_validation_loss


def make_model(model_name,feature_extract,no_of_classes):
    if model_name=="FromScratch":
        model=FromScratch()
    elif model_name=="ResNetFT":
        model=reshape_resnet(no_of_classes, feature_extract, use_pretrained=True)   
    return model


def make_loss_criterion(model_name):
    if model_name=="FromScratch":
        loss_criterion = nn.NLLLoss()
    elif model_name=="ResNetFT":
        loss_criterion = nn.CrossEntropyLoss()
    return loss_criterion


def make_optimizer(given_model_parameters, learning_rate, momentum):
    optimizer = optim.SGD(given_model_parameters, lr = learning_rate, momentum = momentum)
    return optimizer


def make_model_lc_optimizer(model_name,learning_rate, momentum,
                            feature_extract=False,no_of_classes=2):
    model = make_model(model_name,feature_extract, no_of_classes)
    params_to_update = parameters_to_update(model_name, model, feature_extract)
    loss_criterion = make_loss_criterion(model_name)
    optimizer = make_optimizer(params_to_update, learning_rate, momentum)
    return model, loss_criterion, optimizer
    


model_names = ["FromScratch", "ResNetFT"]


def get_model_details():
    i = int(input("Press 0 for FromScratch and 1 for ResNetFT "))
    my_model_name = model_names[i]
    if i==1:
        j = int(input("Press 0 for FineTuning and 1 for FeatureExtracting "))
        feature_extracting=(j==1)
    else:
        feature_extracting=False
    print("************")
    print(f"Using {my_model_name}")
    print(f"feature_extracting : {feature_extracting}")
    return my_model_name, feature_extracting


def get_hyperparameters(default=True):
    if default:
        learning_rate=1e-3
        momentum = 0.9
    else:
        learning_rate = float(input("Enter learning rate "))
        momentum = float(input("Enter momentum "))
    return learning_rate, momentum


def train_it_short(no_of_epochs, starting_epoch, 
              model_name,model,loss_criterion, optimizer,
              batch_size, dataloaders,batches_per_epoch=100,
              is_best=False,min_validation_loss=math.inf):

    last_checkpoint_path = f"./last_checkpoint_for_{model_name}.pt"
    best_model_path=f"./best_model_for_{model_name}.pt"
    tensorboard_dir=f"Training_{model_name}"
    board_writer = SummaryWriter(tensorboard_dir) 
    

    for epoch in range(starting_epoch,starting_epoch+no_of_epochs):
        print(f"Epoch : {epoch}")
        start_time = time()
        
        for phase in ['Training', 'Validation']:
            print(f"{phase}")
            if phase == 'Training':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            loss_in_this_epoch = 0
            no_of_batches_in_this_epoch = 0
            correct_in_this_epoch = 0
            for batch_data, batch_labels in dataloaders[phase]:
                no_of_batches_in_this_epoch+= 1
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'Training'):
                    batch_outputs = model(batch_data)
                    #Compute loss for this batch
                    batch_loss = loss_criterion(batch_outputs, batch_labels)
                    loss_in_this_batch = batch_loss.item()
                    loss_in_this_epoch += loss_in_this_batch
                if phase=='Training':    
                    batch_loss.backward()
                    optimizer.step()
                with torch.no_grad():
                    score, predictions = torch.max(batch_outputs, axis = 1)   
                    correct_in_this_batch = torch.sum(predictions == batch_labels.data).item()
                    correct_in_this_epoch += correct_in_this_batch
                if (no_of_batches_in_this_epoch % (batches_per_epoch//10)) == 0:
                    print(f"{phase} #{no_of_batches_in_this_epoch} Batch Acc : {correct_in_this_batch}/{batch_size}, Batch Loss: {loss_in_this_batch}")
                if no_of_batches_in_this_epoch == batches_per_epoch:
                    print(f"Epoch : {epoch}, {phase}Batch: {no_of_batches_in_this_epoch}")
                    break
            board_writer.add_scalar(f'{phase}/Loss/Average', loss_in_this_epoch/no_of_batches_in_this_epoch, epoch)
            board_writer.add_scalar(f'{phase}/Accuracy/Average', correct_in_this_epoch/(no_of_batches_in_this_epoch*batch_size), epoch)
            board_writer.add_scalar(f'{phase}/TimeTakenInMinutes', (time()-start_time)/60, epoch)
            board_writer.flush()
            print(f"{phase} average accuracy : {correct_in_this_epoch/(no_of_batches_in_this_epoch*batch_size)}")
            print(f"{phase} average loss : {loss_in_this_epoch/no_of_batches_in_this_epoch}")

            
            if phase=='Validation':
                if  min_validation_loss >= loss_in_this_epoch:
                    is_best = True
                    min_validation_loss = min(min_validation_loss,loss_in_this_epoch)
                    checkpoint = {
                        'epoch': epoch + 1,
                        'min_validation_loss': min_validation_loss,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                    save_checkpoint(checkpoint, is_best, last_checkpoint_path, best_model_path)
                    print(f"In epoch number {epoch}, average validation loss decreased to {loss_in_this_epoch/no_of_batches_in_this_epoch}")
            last_checkpoint = {
                    'epoch': epoch + 1,
                    'min_validation_loss': min_validation_loss,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                     }
            save_checkpoint(last_checkpoint, False, last_checkpoint_path, best_model_path)
    board_writer.close()



def train_it_without_gpu(no_of_epochs, starting_epoch, 
              model_name,model,loss_criterion, optimizer,
              batch_size, dataloaders,batches_per_epoch=100,
              is_best=False,min_validation_loss=math.inf):

    last_checkpoint_path = f"./last_checkpoint_for_{model_name}.pt"
    best_model_path=f"./best_model_for_{model_name}.pt"
    tensorboard_dir=f"Training_{model_name}"
    board_writer = SummaryWriter(tensorboard_dir) 
    

    for epoch in range(starting_epoch,starting_epoch+no_of_epochs):
        print(f"Epoch : {epoch}")
        start_time = time()

        model.train()
        print("Training")
        train_loss_in_this_epoch = 0
        no_of_batches_in_this_epoch = 0
        train_correct_in_this_epoch = 0
        for train_batch_data, train_batch_labels in dataloaders["Training"]:
                no_of_batches_in_this_epoch+= 1
                optimizer.zero_grad()
                train_batch_outputs = model(train_batch_data)
                #Compute loss for this batch
                train_batch_loss = loss_criterion(train_batch_outputs, train_batch_labels)
                train_loss_in_this_batch = train_batch_loss.item()
                train_loss_in_this_epoch += train_loss_in_this_batch 
                train_batch_loss.backward()
                optimizer.step()
                with torch.no_grad():
                    train_score, train_predictions = torch.max(train_batch_outputs, axis = 1)   
                    train_correct_in_this_batch = torch.sum(train_predictions == train_batch_labels.data).item()
                    train_correct_in_this_epoch += train_correct_in_this_batch
                if (no_of_batches_in_this_epoch % (batches_per_epoch//10)) == 0:
                    print(f"Training #{no_of_batches_in_this_epoch} Batch Acc : {train_correct_in_this_batch}/{batch_size}, Batch Loss: {train_loss_in_this_batch}")
                if no_of_batches_in_this_epoch == batches_per_epoch:
                    print(f"Epoch : {epoch}, Training Batch: {no_of_batches_in_this_epoch}")
                    break
        board_writer.add_scalar(f'Training/Loss/Average', train_loss_in_this_epoch/no_of_batches_in_this_epoch, epoch)
        board_writer.add_scalar(f'Training/Accuracy/Average', train_correct_in_this_epoch/(no_of_batches_in_this_epoch*batch_size), epoch)
        board_writer.add_scalar(f'Training/TimeTakenInMinutes', (time()-start_time)/60, epoch)
        board_writer.flush()
        print(f"Training average accuracy : {train_correct_in_this_epoch/(no_of_batches_in_this_epoch*batch_size)}")
        print(f"Training average loss : {train_loss_in_this_epoch/no_of_batches_in_this_epoch}")
            

        model.eval()
        print("Validation")
        val_loss_in_this_epoch = 0
        no_of_batches_in_this_epoch = 0
        val_correct_in_this_epoch = 0
        with torch.no_grad():
            for val_batch_data, val_batch_labels in dataloaders["Validation"]:
                no_of_batches_in_this_epoch+= 1
                val_batch_outputs = model(val_batch_data)
                #Compute loss for this batch
                val_batch_loss = loss_criterion(val_batch_outputs, val_batch_labels)
                val_loss_in_this_batch = val_batch_loss.item()
                val_loss_in_this_epoch += val_loss_in_this_batch 
                val_score, val_predictions = torch.max(val_batch_outputs, axis = 1)   
                val_correct_in_this_batch = torch.sum(val_predictions == val_batch_labels.data).item()
                val_correct_in_this_epoch += val_correct_in_this_batch
                if (no_of_batches_in_this_epoch % (batches_per_epoch//10)) == 0:
                    print(f"Validation #{no_of_batches_in_this_epoch} Batch Acc : {val_correct_in_this_batch}/{batch_size}, Batch Loss: {val_loss_in_this_batch}")
                if no_of_batches_in_this_epoch == batches_per_epoch:
                    print(f"Epoch : {epoch}, Validation Batch: {no_of_batches_in_this_epoch}")
                    break
            board_writer.add_scalar(f'Validation/Loss/Average', val_loss_in_this_epoch/no_of_batches_in_this_epoch, epoch)
            board_writer.add_scalar(f'Validation/Accuracy/Average', val_correct_in_this_epoch/(no_of_batches_in_this_epoch*batch_size), epoch)
            board_writer.add_scalar(f'Validation/TimeTakenInMinutes', (time()-start_time)/60, epoch)
            board_writer.flush()
            print(f"Validation average accuracy : {val_correct_in_this_epoch/(no_of_batches_in_this_epoch*batch_size)}")
            print(f"Validation average loss : {val_loss_in_this_epoch/no_of_batches_in_this_epoch}")
            if  min_validation_loss >= val_loss_in_this_epoch:
                    is_best = True
                    min_validation_loss = min(min_validation_loss,val_loss_in_this_epoch)
                    checkpoint = {
                        'epoch': epoch + 1,
                        'min_validation_loss': min_validation_loss,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                    save_checkpoint(checkpoint, is_best, last_checkpoint_path, best_model_path)
                    print(f"In epoch number {epoch}, average validation loss decreased to {val_loss_in_this_epoch/no_of_batches_in_this_epoch}")
            last_checkpoint = {
                    'epoch': epoch + 1,
                    'min_validation_loss': min_validation_loss,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                     }
            save_checkpoint(last_checkpoint, False, last_checkpoint_path, best_model_path)
    board_writer.close()



    
def train_it(no_of_epochs, starting_epoch, 
              model_name,model,loss_criterion, optimizer,
              batch_size, dataloaders,board_writer,device,batches_per_epoch=100,
              is_best=False,min_validation_loss=math.inf):

    last_checkpoint_path = f"./last_checkpoint_for_{model_name}.pt"
    best_model_path=f"./best_model_for_{model_name}.pt"    
    

    for epoch in range(starting_epoch,starting_epoch+no_of_epochs):
        print(f"Epoch : {epoch}")
        start_time = time()

        model.train()
        print("Training")
        train_loss_in_this_epoch = 0
        no_of_batches_in_this_epoch = 0
        train_correct_in_this_epoch = 0
        for train_batch_data, train_batch_labels in dataloaders["Training"]:
                train_batch_data, train_batch_labels = train_batch_data.to(device), train_batch_labels.to(device)
                no_of_batches_in_this_epoch+= 1
                optimizer.zero_grad()
                train_batch_outputs = model(train_batch_data)
                #Compute loss for this batch
                train_batch_loss = loss_criterion(train_batch_outputs, train_batch_labels)
                train_loss_in_this_batch = train_batch_loss.item()
                train_loss_in_this_epoch += train_loss_in_this_batch 
                train_batch_loss.backward()
                optimizer.step()
                with torch.no_grad():
                    train_score, train_predictions = torch.max(train_batch_outputs, axis = 1)   
                    train_correct_in_this_batch = torch.sum(train_predictions == train_batch_labels.data).item()
                    train_correct_in_this_epoch += train_correct_in_this_batch
                if (no_of_batches_in_this_epoch % (batches_per_epoch//10)) == 0:
                    print(f"Training #{no_of_batches_in_this_epoch} Batch Acc : {train_correct_in_this_batch}/{batch_size}, Batch Loss: {train_loss_in_this_batch}")
                if no_of_batches_in_this_epoch == batches_per_epoch:
                    print(f"Epoch : {epoch}, Training Batch: {no_of_batches_in_this_epoch}")
                    break
        board_writer.add_scalar(f'Training/Loss/Average', train_loss_in_this_epoch/no_of_batches_in_this_epoch, epoch)
        board_writer.add_scalar(f'Training/Accuracy/Average', train_correct_in_this_epoch/(no_of_batches_in_this_epoch*batch_size), epoch)
        board_writer.add_scalar(f'Training/TimeTakenInMinutes', (time()-start_time)/60, epoch)
        board_writer.flush()
        print(f"Training average accuracy : {train_correct_in_this_epoch/(no_of_batches_in_this_epoch*batch_size)}")
        print(f"Training average loss : {train_loss_in_this_epoch/no_of_batches_in_this_epoch}")
            

        model.eval()
        print("Validation")
        val_loss_in_this_epoch = 0
        no_of_batches_in_this_epoch = 0
        val_correct_in_this_epoch = 0
        with torch.no_grad():
            for val_batch_data, val_batch_labels in dataloaders["Validation"]:
                val_batch_data, val_batch_labels = val_batch_data.to(device), val_batch_labels.to(device)
                no_of_batches_in_this_epoch+= 1
                val_batch_outputs = model(val_batch_data)
                #Compute loss for this batch
                val_batch_loss = loss_criterion(val_batch_outputs, val_batch_labels)
                val_loss_in_this_batch = val_batch_loss.item()
                val_loss_in_this_epoch += val_loss_in_this_batch 
                val_score, val_predictions = torch.max(val_batch_outputs, axis = 1)   
                val_correct_in_this_batch = torch.sum(val_predictions == val_batch_labels.data).item()
                val_correct_in_this_epoch += val_correct_in_this_batch
                if (no_of_batches_in_this_epoch % (batches_per_epoch//10)) == 0:
                    print(f"Validation #{no_of_batches_in_this_epoch} Batch Acc : {val_correct_in_this_batch}/{batch_size}, Batch Loss: {val_loss_in_this_batch}")
                if no_of_batches_in_this_epoch == batches_per_epoch:
                    print(f"Epoch : {epoch}, Validation Batch: {no_of_batches_in_this_epoch}")
                    break
            board_writer.add_scalar(f'Validation/Loss/Average', val_loss_in_this_epoch/no_of_batches_in_this_epoch, epoch)
            board_writer.add_scalar(f'Validation/Accuracy/Average', val_correct_in_this_epoch/(no_of_batches_in_this_epoch*batch_size), epoch)
            board_writer.add_scalar(f'Validation/TimeTakenInMinutes', (time()-start_time)/60, epoch)
            board_writer.flush()
            print(f"Validation average accuracy : {val_correct_in_this_epoch/(no_of_batches_in_this_epoch*batch_size)}")
            print(f"Validation average loss : {val_loss_in_this_epoch/no_of_batches_in_this_epoch}")
            if  min_validation_loss >= val_loss_in_this_epoch:
                    is_best = True
                    min_validation_loss = min(min_validation_loss,val_loss_in_this_epoch)
                    checkpoint = {
                        'epoch': epoch + 1,
                        'min_validation_loss': min_validation_loss,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                    }
                    save_checkpoint(checkpoint, is_best, last_checkpoint_path, best_model_path)
                    print(f"In epoch number {epoch}, average validation loss decreased to {val_loss_in_this_epoch/no_of_batches_in_this_epoch}")
            last_checkpoint = {
                    'epoch': epoch + 1,
                    'min_validation_loss': min_validation_loss,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                     }
            save_checkpoint(last_checkpoint, False, last_checkpoint_path, best_model_path)
    board_writer.close()
