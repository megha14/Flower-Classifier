### Flower-Classifier


Trained an image classifier to classify different species of flowers. It is a command line application that takes in input the location of dataset, the classifier architecture (VGG, Densenet, AlexNet), and other parameters to train a classifier on flower dataset. Project done as part of Udacity's AI Programming with Python Nanodegree.

## Scripts

# Training the classifier using **train.py**

Basic usage using default settings

``python train.py ./flowers``

To change the architecture

``python train.py ./flowers --arch "densenet"``

To change other parameters

``python train.py ./flowers --learning_rate 0.01 --hidden_units 512 --epochs 20 --dropout 0.5 --gpu --save_dir checkpint.pth``

# Prediction using **predict.py**

Basic usage using default settings using a test image sample

``python predict.py ./flowers/test/20/image_04910``

To change other parameters using a test image sample

``python predict.py ./flowers/test/20/image_04910 --category_names cat_to_name.json --top_k 10 --gpu``

**Didn't include the flowers dataset here as it has large size**
