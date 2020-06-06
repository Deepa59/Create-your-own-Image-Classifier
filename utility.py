import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from PIL import Image

from torchvision import datasets, transforms, models
import json

# Function to load and preprocess the data
def load_data(data_dir):
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    val_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    #define transforms for the train,test,valid sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    
    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])]) 

    
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data = datasets.ImageFolder(val_dir, transform=test_transforms)
    
    # The trainloader will have shuffle=True so that the order of the images do not affect the model
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    
    return trainloader, testloader, validloader, train_data, test_data, valid_data

#function to process the image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    #resize the image with shorter side=256
    w, h = image.size

    if w == h:
        size = 256, 256
    elif w > h:
        ratio = w/h #aspect ratio
        size = 256*ratio, 256
    elif h > w:
        ratio = h/w
        size = 256, 256*ratio
        
        
    image.thumbnail(size)
    
    #the centre 224x224 portion of the image
    l = (256 - 224)/2
    t = (256 - 224)/2
    r = (256 + 224)/2
    b = (256 + 224)/2

    image = image.crop((l, t, r, b))
    
    #convert the color channel integer values of 0-255 to 0-1
    image = np.array(image)
    image = image/255.0 
    
    #normalize images
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image=(image-mean)/std
    
    
    #reorder the dimensions to have color channel as first dimension
    image = np.transpose(image, (2, 0, 1))
    
    
    return image
    
    

