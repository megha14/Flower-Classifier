import torch
import numpy as np
from torch import nn
from torchvision import models
from collections import OrderedDict
from data_utils import process_image


# Function to build new classifier
def build_classifier(model, inputs, hidden, dropout):
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
        
    #Network Architecture
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(inputs, hidden)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=dropout)),
                          ('fc2', nn.Linear(hidden, 100)),
                          ('relu2', nn.ReLU()),
                          ('dropout2', nn.Dropout(p=dropout)),
                          ('fc3', nn.Linear(100, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.classifier = classifier
    return model;

def validate_model(model, validloader, criterion, device):
    valid_loss = 0
    accuracy = 0
   
    with torch.no_grad():
        for inputs, labels in validloader:
            
            inputs, labels = inputs.to(device), labels.to(device)
                
            logps = model.forward(inputs)
            valid_loss += criterion(logps, labels).item()         
                    
            # Calculate accuracy
            ps = torch.exp(logps)
            equals = (labels.data == ps.max(dim = 1)[1])
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    return valid_loss, accuracy

def train_model(model, epochs, trainloader, validloader, criterion, optimizer, device):
    #print("in train")
    steps = 0
    running_loss = 0
    print_every = 30
                      
    for epoch in range(epochs):
        #print("in epoch")
        for inputs, labels in trainloader:
            
            steps += 1
           
            inputs, labels = inputs.to(device), labels.to(device)
       
            optimizer.zero_grad()
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            #print(running_loss)
                      
            if steps % print_every == 0:
                #print("in valid")
                # setting model to evaluation mode during validation
                model.eval()
                # Gradients are turned off as no longer in training
                with torch.no_grad():
                      valid_loss, accuracy = validate_model(model, validloader, criterion, device)
            
                print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.4f}.. "
                  f"Validation loss: {valid_loss/len(validloader):.4f}.. "
                  f"Validation accuracy: {accuracy/len(validloader):.4f}")
            running_loss = 0
            model.train()
                      
    return model

                      
def test_model(model, testloader, device):
    right = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
           
            inputs, labels = inputs.to(device), labels.to(device)
        
            output = model(inputs)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            right += (predicted == labels).sum().item()
        
    print('Accuracy achieved by the network on test images is: %d%%' % (100 * right / total))
    
def save_model(model, train_data, save_dir):
    # Saving: feature weights, new classifier, index-to-class mapping, optimiser state, and No. of epochs
    checkpoint = {'architecture': model.name,
             'classifier': model.classifier,
             'class_to_idx': model.class_to_idx,
             'state_dict': model.state_dict()}

    return torch.save(checkpoint, save_dir)

def load_checkpoint(save_dir):
    """
    Loads deep learning model checkpoint.
    """
    
    # Load the saved file
    
    checkpoint = torch.load(save_dir)
    
    # Download pretrained model
    if checkpoint['architecture'] == 'vgg16':
       model = models.vgg16(pretrained=True)
    if checkpoint['architecture'] == 'vgg19':
       model = models.vgg19(pretrained=True)
    elif checkpoint['architecture'] == 'densenet':
       model = models.densenet161(pretrained=True)
    elif checkpoint['architecture'] == 'alexnet':
       model = models.alexnet(pretrained=True)
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters(): 
        param.requires_grad = False
    
    # Load stuff from checkpoint
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])

    
    return model

def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.to(device)
    img = process_image(image_path).unsqueeze_(0).float()
    img = img.to(device)
   
    model.eval()
    with torch.no_grad():
        output = model.forward(img)
        ps = torch.exp(output)
        
    probabilities, indices = torch.topk(ps, topk)
    probs = np.array(probabilities.data[0]) 
    inds = np.array(indices.data[0])
    
    idx_to_class = {idx:c for c,idx in model.class_to_idx.items()}
    classes = [idx_to_class[i] for i in inds]
    
        
    return probs, classes