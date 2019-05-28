import argparse

import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

from data_utils import load_data
from model_utils import build_classifier, validate_model, train_model, test_model, save_model, load_checkpoint

parser = argparse.ArgumentParser(description='Training Neural Networks to classify Flowers.')

parser.add_argument('data_directory', action = 'store', 
                    help = 'Enter training data path.')

parser.add_argument('--arch', action='store',
                    dest = 'pretrained_model', default = 'vgg16',
                    help= 'This classifier can currently work with vgg16, densenet161, vgg19, alexnet. \n\nUse vgg16 for vgg16 \nUse vgg19 for vgg19 \nUse densenet for densenet161 \nUse alexnet for alexnet \nDefault is vgg16')

parser.add_argument('--save_dir', action = 'store',
                    dest = 'save_directory', default = 'checkpoint.pth',
                    help = 'Enter location to save checkpoint in.')

parser.add_argument('--learning_rate', action = 'store',
                    dest = 'lr', type=float, default = 0.001,
                    help = 'Enter learning rate for training the model, default is 0.001.')

parser.add_argument('--dropout', action = 'store',
                    dest='drpt', type=float, default = 0.2,
                    help = 'Enter dropout for training the model, default is 0.5.')

parser.add_argument('--hidden_units', action = 'store',
                    dest = 'units', type=int, default = 512,
                    help = 'Enter number of hidden units in classifier, default is 4096.')

parser.add_argument('--epochs', action = 'store',
                    dest = 'num_epochs', type = int, default = 10,
                    help = 'Enter number of epochs to use during training, default is 5.')

parser.add_argument('--gpu', action = "store_true", default = False,
                    help = 'Turn GPU mode on or off, default is off.')

results = parser.parse_args()

data_dir = results.data_directory
save_dir = results.save_directory
learning_rate = results.lr
dropout = results.drpt
hidden_units = results.units
epochs = results.num_epochs
gpu_mode = results.gpu

if gpu_mode == True:
    device = torch.device("cuda" if torch.cuda.is_available()
                           else "cpu")

#Loading Data
trainloader, testloader, validloader, train_data, test_data, valid_data = load_data(data_dir)

#loading model
pre_tr_model = results.pretrained_model
print("Selected model : "+pre_tr_model)
model = getattr(models,pre_tr_model)(pretrained=True)
model.class_to_idx = train_data.class_to_idx

#setting input units based on model given in input
if pre_tr_model == 'vgg16':
    input_units = model.classifier[0].in_features
    model.name = 'vgg16'
elif pre_tr_model == 'vgg19':
    input_units = model.classifier[0].in_features
    model.name = 'vgg19'
elif pre_tr_model == 'densenet':
    input_units = model.classifier.in_features
    model.name = 'densenet'
elif pre_tr_model == 'alexnet':
    input_units = model.classifier[1].in_features
    model.name = 'alexnet'

#building classifier of model
model = build_classifier(model, input_units, hidden_units, dropout)
print(model)

#Set criterion and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), learning_rate)
model.to(device)

# Training model
model = train_model(model, epochs, trainloader, validloader, criterion, optimizer, device)

# Testing model
test_model(model, testloader, device)

# Saving model
save_model(model, train_data, save_dir)