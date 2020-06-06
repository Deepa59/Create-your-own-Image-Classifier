import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

# Function to build new classifier
def build_classifier(model, input_units, hidden_units, dropout):
    # Weights of pretrained model are frozen so we don't backprop through/update them
    for param in model.parameters():
        param.requires_grad = False
        
        
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_units, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('dropout1', nn.Dropout(dropout)),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    # Replacing the pretrained classifier with the one above
    model.classifier = classifier
    return model


def train_model(model, epochs,trainloader, validloader, criterion, optimizer):
    model.to('cuda')
    steps = 0
    running_loss = 0
    print_every = 10

    for epoch in range(epochs):
        
        for inputs, labels in trainloader:
            
            steps += 1
            # Move input and label tensors to cuda
        
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
        
            optimizer.zero_grad()
        
             #Forward pass
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
             # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to('cuda'), labels.to('cuda')
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()
                    
                    # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
             
                print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train Loss: {running_loss/print_every:.3f}.. "
                  f"Validation Loss: {valid_loss/len(validloader):.3f}.. "
                  f"Accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
    
    return model,optimizer



def  test_model(model, testloader, criterion):
    test_loss = 0
    accuracy = 0
    model.to('cuda')
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
                    
            test_loss += batch_loss.item()
                    
        # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
    print(f"'Test accuracy:{100*accuracy/len(testloader):.3f}%")


def save_model(model, train_data, optimizer, save_dir, epochs,arch):
    checkpoint = {'arch':arch,
                  'state_dict': model.state_dict(),
                  'classifier': model.classifier,
                  'class_to_idx': train_data.class_to_idx,
                  'opt_state': optimizer.state_dict,
                  'num_epochs': epochs}

    
    return torch.save(checkpoint, save_dir)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    print(checkpoint['arch'])
    model = getattr(models, checkpoint['arch'])(pretrained=True)
     # Freeze parameters so we don't backprop through them
    for param in model.parameters(): param.requires_grad = False
    
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['num_epochs']
    model.optimizer = checkpoint['opt_state']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model
    