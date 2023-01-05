# 2. Predict
# Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

# Basic usage: python predict.py /path/to/image checkpoint
# Options:
# Return top KK most likely classes: python predict.py input checkpoint --top_k 3
# Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
# Use GPU for inference: python predict.py input checkpoint --gpu

# Imports here
import argparse
import torch
import torchvision
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import OrderedDict
from PIL import Image
import numpy as np
import json

import matplotlib.pyplot as plt
import seaborn as sns

# Define command line arguments for predict.py
def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Input image')
    parser.add_argument('checkpoint', type=str, help='Checkpoint file')
    parser.add_argument('--top_k', type=int, default=3, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='', help='Mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    args = parser.parse_args()
    return args

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        ret
        urns an Numpy array
    '''
    pil_image = Image.open(image)
    pil_image.thumbnail((256, 256))
    pil_image = pil_image.crop((16, 16, 240, 240))
    np_image = np.array(pil_image)
    np_image = np_image/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    np_image = np_image.transpose((2, 0, 1))
    return np_image


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def test(path, model, cat_to_name):
    # TODO: Display an image along with the top 5 classes
    image_path =  path
    probs, classes, flowers = predict(image_path, model)
    image = process_image(image_path)
    plt.figure(figsize = (6,10))
    ax = plt.subplot(2,1,1)
    flower_num = image_path.split('/')[2]
    title_ = cat_to_name[flower_num]
    imshow(image, ax, title = title_);
    plt.subplot(2,1,2)
    sns.barplot(x=probs, y=flowers, color=sns.color_palette()[0])
    plt.show()

def predict():
    # Get input arguments
    args = get_input_args()
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint)
    # Load category names
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
    # Load model
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    # Process image
    image = Image.open(args.input)
    image = process_image(image)
    # Predict
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.unsqueeze_(0)
    # check whether the gpu is available or not and set parameters accordingly
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    print(f"Using device: {device}")

    if device == 'cuda':
        image = image.cuda()
        model = model.cuda()
    else:
        image = image.cpu()
        model = model.cpu()
        
    model.eval()
    with torch.no_grad():
        output = model.forward(image)
    ps = torch.exp(output)
    top_p, top_class = ps.topk(args.top_k, dim=1)
    top_p = top_p[0].tolist()
    top_class = top_class[0].tolist()
    if args.category_names:
        top_class = [cat_to_name[str(i+1)] for i in top_class]
        top_flowers = [cat_to_name[str(i+1)] for i in top_class]

    test(args.input, model, cat_to_name)

    return top_p, top_class, top_flowers

if __name__ == '__main__':
    top_p, top_class = predict()
    print(top_p)
    print(top_class)
