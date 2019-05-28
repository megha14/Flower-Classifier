import argparse
import torch
from torchvision import models
from PIL import Image
import json

from data_utils import load_data, process_image
from model_utils import load_checkpoint, predict


parser = argparse.ArgumentParser(description='Use neural network to make prediction on image.')

parser.add_argument('image_path', action='store',
                    default = '../aipnd-project/flowers/test/20/image_04910.jpg',
                    help='Enter path to image.')

parser.add_argument('--checkpoint', action='store',
                    dest='checkpoint', default = 'checkpoint.pth',
                    help='Enter location to save checkpoint in.')

parser.add_argument('--top_k', action='store',
                    dest='topk', type=int, default = 5,
                    help='Enter number of top most likely classes to view, default is 5.')

parser.add_argument('--cat_to_name', action='store',
                    dest='cat_name_dir', default = 'cat_to_name.json',
                    help='Enter path to image.')

parser.add_argument('--gpu', action="store_true", default=False,
                    help='Turn GPU mode on or off, default is off.')

results = parser.parse_args()

checkpoint = results.checkpoint
image = results.image_path
top_k = results.topk
gpu_mode = results.gpu
cat_names = results.cat_name_dir

device = torch.device("cpu")
if gpu_mode == True:
    device = torch.device("cuda" if torch.cuda.is_available()
                           else "cpu")
    
with open(cat_names, 'r') as f:
    cat_to_name = json.load(f)
    

# Load model
loaded_model = load_checkpoint(checkpoint)
loaded_model.to(device)


# Carry out prediction
probs, classes = predict(image, loaded_model, device, top_k)

# Print probabilities and predicted classes
labels = [cat_to_name[c] for c in classes]
for p, c, l in zip(probs, classes, labels):
    print("Probability is {0:2f} for class {1} with corresponding label {2}".format(p, c, l))
