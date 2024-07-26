# Face Landmarks Detection Model
This repository contains a PyTorch implementation for detecting facial landmarks using a deep neural network. The model is trained to predict 68 facial landmarks from grayscale images of faces. Additionally, there is a separate model trained to detect landmarks specifically around the eyes.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Transformations](#transformations)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)

## Introduction
Facial landmark detection is a crucial step in various computer vision applications, including facial recognition, emotion detection, and augmented reality. This project leverages a modified ResNet-18 architecture to perform landmark detection on grayscale images of faces. An additional model is also available for detecting eye landmarks only.

## Requirements
To run the code, you need the following dependencies:

- Python 3.7 or higher
- PyTorch
- torchvision
- OpenCV
- PIL (Python Imaging Library)
- numpy
- imutils
- matplotlib

You can install the required packages using:
``` bash
pip install torch torchvision opencv-python pillow numpy imutils matplotlib
```

## Dataset
The dataset used for training is the iBUG 300-W dataset, which contains annotated images with 68 facial landmarks. The dataset should be organized as follows:

``` bash
ibug_300W_large_face_landmark_dataset
├───afw
├───helen
│   ├───testset
│   └───trainset
├───ibug
├───lfpw
│   ├───testset
│   └───trainset
├───image_metadata_stylesheet.xsl
├───labels_ibug_300W.xml
├───labels_ibug_300W_test.xml
└───labels_ibug_300W_train.xml
```

## Transformations
The Transforms class applies various image augmentations to improve the robustness of the model. The augmentations include:

Face cropping
Resizing
Color jittering
Brightness adjustment
Adding Gaussian noise
Rotation

## Model Architecture
The model is based on the ResNet-18 architecture, modified to accept grayscale images as input and to output 136 values (face) or 24 values (eyes) (landmarks with x and y coordinates).

``` python
class Network(nn.Module):
    def __init__(self, num_classes= 136 / 28):
        super().__init__()
        self.model = models.resnet18()
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x   
```

## Training
The training process involves:

Splitting the dataset into training and validation sets.
Shuffling and batching the data.
Training the model using Mean Squared Error (MSE) loss and the Adam optimizer.
Saving the model checkpoint with the lowest validation loss.
Training is available for multiple platforms like CPU, cuda and rocm, please refer to the pages listed below.

### Full face landmarks
Training of the model is available with:
* [cpu](training/face/cpu)
* [cuda](training/face/cuda)
* [rocm](training/face/rocm) (Linux only) TODO

###  Eye landmarks
Training of the model is available with:
* [cpu](training/eyes/cpu)
* [cuda](training/eyes/cuda)
* [rocm](training/eyes/rocm) (Linux only) TODO

## Usage
To use the trained model for inference, load the model weights and pass a preprocessed image through the network. Example code for inference will be provided soon.

# Acknowledgements
* This project uses the iBUG 300-W dataset. Special thanks to the authors and contributors of the dataset.
* Reference taken from [this towards datascience article](https://towardsdatascience.com/face-landmarks-detection-with-pytorch-4b4852f5e9c4)
