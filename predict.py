import argparse
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import json
from collections import OrderedDict
import time
import numpy as np
import matplotlib.pyplot as plt
from fc_model import build_classifier,train_model,test_model,save_model,load_checkpoint
from utility import load_data ,process_image


parser = argparse.ArgumentParser(description="Predict image")
parser.add_argument('--filepath', dest='filepath', default='flowers/test/5/image_05159.jpg')
parser.add_argument('--checkpoint', action='store', dest='save_data', default='checkpoint.pth')
parser.add_argument('--top_k', dest='top_k', default='3')
parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json')
parser.add_argument('--gpu', action='store', default='gpu')

results=parser.parse_args()
image_path=results.filepath
checkpoint=results.save_data
top_k=results.top_k
cat_names=results.category_names
gpu_mode=results.gpu

#label mapping
with open(cat_names, 'r') as f:
    cat_to_name = json.load(f)

#load the checkpoint
model=load_checkpoint(checkpoint)


#func to predict class
def predict(image_path, model, topk,gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    if gpu == 'gpu':
        model = model.cuda()
    else:
        model = model.cpu()
        
    model.eval();
    img=Image.open(image_path)
    image=process_image(img)
    
    #convert numpy image to tensor
    img_tensor=torch.from_numpy(image).type(torch.FloatTensor)
    
    #expected dimension of tensor is batch x channel x w x h
    img_tensor=img_tensor.unsqueeze_(0)  
    #batch size is 1
    
    
    if gpu=='gpu':
        image_t=img_tensor.to('cuda')
        
    else:
        image_t=img_tensor
            
    # Set model to evaluate
    with torch.no_grad():
        output=model.forward(image_t)
        
    #Move tensor to cpu memory since numpy doesn't support gpu
    output=output.to('cpu')
        
    # Calculating probabilities
    probs = torch.exp(output)
    
    probs, indices = probs.topk(topk)
    
    #convert to numpy and then to list
    top_probs=probs.numpy()
    top_probs=top_probs.tolist()[0]
    
    top_ind=indices.numpy()
    top_ind=top_ind.tolist()[0]
    
    #invert dict class_to_idx
    class_to_idx=model.class_to_idx
    idx_to_class = {val:key for key,val in class_to_idx.items()}
    
    class_list=[]
    
    for i in top_ind:
        class_list+=[idx_to_class[i]]
        
        
    return top_probs,class_list
        

    
probs, classes = predict(image_path, model, int(top_k),gpu_mode)
#map index to class
flowers = [cat_to_name[i] for i in classes]

print("Image path :{}\n".format(image_path))

print("The top labels are:\n")

for j in range(len(flowers)):
    print("{} has probability of {}".format(flowers[j],probs[j]))
    
    
print("\n The image is most likely {} with probability {}".format(flowers[0],probs[0]))