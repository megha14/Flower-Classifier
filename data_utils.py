import torch
import numpy as np
import PIL
from PIL import Image
from torchvision import datasets, transforms, models

# Function to load and preprocess the data
def load_data(data_dir):
    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = {"train_transform" : transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])]),
                       "test_transform" : transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])]),
                       "valid_transform" : transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])}

    # TODO: Load the datasets with ImageFolder
    image_datasets = { "train_data" : datasets.ImageFolder(data_dir + '/train', transform=data_transforms["train_transform"]),
                       "test_data" : datasets.ImageFolder(data_dir + '/test', transform=data_transforms["test_transform"]),
                       "valid_data" : datasets.ImageFolder(data_dir + '/valid', transform=data_transforms["valid_transform"])}

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = { "trainloader" : torch.utils.data.DataLoader(image_datasets["train_data"], batch_size=50, shuffle=True),
                    "testloader" : torch.utils.data.DataLoader(image_datasets["test_data"], batch_size=50, shuffle=True),
                    "validloader" : torch.utils.data.DataLoader(image_datasets["valid_data"], batch_size=50, shuffle=True)}

    return dataloaders["trainloader"], dataloaders["testloader"], dataloaders["validloader"], image_datasets["train_data"], image_datasets["test_data"], image_datasets["valid_data"]

# Function to load and preprocess test image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    img_pil = PIL.Image.open(image)
    
    img_loader = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    np_image = img_loader(img_pil)  
    
    return np_image
