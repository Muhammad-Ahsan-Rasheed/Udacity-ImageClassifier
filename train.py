# 1. Train
# Train a new network on a data set with train.py

# Basic usage: python train.py data_directory
# Prints out training loss, validation loss, and validation accuracy as the network trains
# Options:
# Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
# Choose architecture: python train.py data_dir --arch "vgg13"
# Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
# Use GPU for training: python train.py data_dir --gpu

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import seaborn as sns
import argparse
from collections import OrderedDict

def argparser():
    parser = argparse.ArgumentParser(description='Train.py')
    parser.add_argument('data_dir', action='store', default='flowers')
    parser.add_argument('--save_dir', action='store', dest='save_dir', default='checkpoint.pth')
    parser.add_argument('--arch', action='store', dest='arch', default='vgg13')
    parser.add_argument('--learning_rate', action='store', dest='learning_rate', type=float, default=0.001)
    parser.add_argument('--hidden_units', action='store', dest='hidden_units', type=int, default=512)
    parser.add_argument('--epochs', action='store', dest='epochs', type=int, default=20)
    parser.add_argument('--gpu', action='store', dest='gpu', default='gpu')
    return parser.parse_args()

def train():

    args = argparser()
    
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms =  transforms.Compose([transforms.RandomRotation(30), transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    image_datasets = datasets.ImageFolder(train_dir, transform=data_transforms)

    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=64, shuffle=True)

    model = getattr(models, args.arch)(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    split = args.hidden_units

    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(25088, split)),
                            ('relu', nn.ReLU()),
                            ('dropout', nn.Dropout(p=0.5)),
                            ('fc2', nn.Linear(split, 102)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))

    model.classifier = classifier

    criterion = nn.NLLLoss()

    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu == 'gpu' else "cpu")
    print('Available device is : ', device)

    model.to(device)       

    epochs = args.epochs
    steps = 0
    running_loss = 0
    print_every = 5

    for epoch in range(epochs):
        for inputs, labels in dataloaders:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in dataloaders:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(dataloaders):.3f}.. "
                      f"Test accuracy: {accuracy/len(dataloaders):.3f}")
                running_loss = 0
                model.train()

    model.class_to_idx = image_datasets.class_to_idx

    checkpoint = {'input_size': 25088,
                    'output_size': 102,
                    'arch': args.arch,
                    'learning_rate': args.learning_rate,
                    'hidden_units': args.hidden_units,
                    'epochs': args.epochs,
                    'classifier': classifier,
                    'state_dict': model.state_dict(),
                    'class_to_idx': model.class_to_idx}
                    
    torch.save(checkpoint, args.save_dir)

    
if __name__ == '__main__':
    train()
