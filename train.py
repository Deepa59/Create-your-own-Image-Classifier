import argparse

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
import torch.nn.functional as F
from PIL import Image

from collections import OrderedDict

import time

import numpy as np
import matplotlib.pyplot as plt
from fc_model import build_classifier,train_model,test_model,save_model

from utility import load_data ,process_image

parser = argparse.ArgumentParser(description="Train neural network")
parser.add_argument('--data_dir', action='store')
parser.add_argument('--arch', dest='arch', default='vgg13',choices=['vgg16','densenet121'])
parser.add_argument('--learning_rate', dest='learning_rate', default='0.001',type=float)
parser.add_argument('--hidden_units', dest='hidden_units', default='512',type=int)
parser.add_argument('--epochs', dest='epochs', default='3',type=int)
parser.add_argument('--gpu', action='store', default='gpu', help = 'Turn GPU on')
parser.add_argument('--save_dir', dest="save_dir", action="store", default="checkpoint.pth", help = 'Enter location to save checkpoint')
    
result=parser.parse_args()
data_dir=result.data_dir
arch=result.arch
lr=result.learning_rate
hidden_units=result.hidden_units
epochs=result.epochs
gpu_mode=result.gpu
save_dir=result.save_dir


# Load and preprocess data 
trainloader, testloader, validloader, train_data, test_data, valid_data = load_data(data_dir)

model = getattr(models,arch)(pretrained=True)

# Build and attach new classifier
input_units = model.classifier[0].in_features
dropout=0.2
build_classifier(model, input_units, hidden_units, dropout)

#use NLLLoss
criterion = nn.NLLLoss()
# Using Adam optimiser
optimizer = optim.Adam(model.classifier.parameters(), lr)

# Train model
model, optimizer = train_model(model, epochs,trainloader, validloader, criterion, optimizer)

# Test model
test_model(model, testloader,criterion)

# Save model
save_model(model, train_data, optimizer, save_dir, epochs,arch)




