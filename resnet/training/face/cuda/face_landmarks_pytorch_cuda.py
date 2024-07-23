import time
import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imutils
import matplotlib.image as mpimg
from math import *
import xml.etree.ElementTree as ET

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset


class Transforms():
    def __init__(self):
        pass

    def rotate(self, image, landmarks, angle):
        angle = random.uniform(-angle, +angle)

        transformation_matrix = torch.tensor([
            [+cos(radians(angle)), -sin(radians(angle))],
            [+sin(radians(angle)), +cos(radians(angle))]
        ])

        image = imutils.rotate(np.array(image), angle)

        landmarks = landmarks - 0.5
        new_landmarks = np.matmul(landmarks, transformation_matrix)
        new_landmarks = new_landmarks + 0.5
        return Image.fromarray(image), new_landmarks

    def resize(self, image, landmarks, img_size):
        image = TF.resize(image, img_size)
        return image, landmarks

    def color_jitter(self, image, landmarks):
        color_jitter = transforms.ColorJitter(
            brightness=0.5,
            contrast=0.5,
            saturation=0.5,
            hue=0.2
        )
        image = color_jitter(image)
        return image, landmarks

    def brightness_adjust(self, image, landmarks):
        brightness_factor = random.uniform(0.5, 1.5)
        image = TF.adjust_brightness(image, brightness_factor)
        return image, landmarks

    def add_gaussian_noise(self, image, landmarks):
        noise = np.random.normal(0, 0.1, np.array(image).shape)
        image = np.clip(np.array(image) + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(image), landmarks

    def crop_face(self, image, landmarks, crops):
        left = int(crops['left'])
        top = int(crops['top'])
        width = int(crops['width'])
        height = int(crops['height'])

        image = TF.crop(image, top, left, height, width)

        img_shape = np.array(image).shape
        landmarks = torch.tensor(landmarks) - torch.tensor([[left, top]])
        landmarks = landmarks / torch.tensor([img_shape[1], img_shape[0]])
        return image, landmarks

    def __call__(self, image, landmarks, crops):
        image = Image.fromarray(image)
        image, landmarks = self.crop_face(image, landmarks, crops)
        image, landmarks = self.resize(image, landmarks, (224, 224))
        image, landmarks = self.color_jitter(image, landmarks)
        image, landmarks = self.brightness_adjust(image, landmarks)
        image, landmarks = self.add_gaussian_noise(image, landmarks)
        image, landmarks = self.rotate(image, landmarks, angle=50)

        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.5], [0.5])
        return image, landmarks


class FaceLandmarksDataset(Dataset):

    def __init__(self, transform=None):

        tree = ET.parse('../../../../ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml')
        root = tree.getroot()

        self.image_filenames = []
        self.landmarks = []
        self.crops = []
        self.transform = transform
        self.root_dir = '../../../../ibug_300W_large_face_landmark_dataset'

        for filename in root[2]:
            self.image_filenames.append(os.path.join(self.root_dir, filename.attrib['file']))

            self.crops.append(filename[0].attrib)

            landmark = []
            for num in range(68):
                x_coordinate = int(filename[0][num].attrib['x'])
                y_coordinate = int(filename[0][num].attrib['y'])
                landmark.append([x_coordinate, y_coordinate])
            self.landmarks.append(landmark)

        self.landmarks = np.array(self.landmarks).astype('float32')

        assert len(self.image_filenames) == len(self.landmarks)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image = cv2.imread(self.image_filenames[index], 0)
        landmarks = self.landmarks[index]

        if self.transform:
            image, landmarks = self.transform(image, landmarks, self.crops[index])

        landmarks = landmarks - 0.5

        return image, landmarks


dataset = FaceLandmarksDataset(Transforms())


def main():
    # split the dataset into testing and test sets
    len_valid_set = int(0.1 * len(dataset))
    len_train_set = len(dataset) - len_valid_set

    print("The length of Train set is {}".format(len_train_set))
    print("The length of Valid set is {}".format(len_valid_set))

    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [len_train_set, len_valid_set])

    # shuffle and batch the datasets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=8, shuffle=True, num_workers=4)

    images, landmarks = next(iter(train_loader))

    print(images.shape)
    print(landmarks.shape)

    class Network(nn.Module):
        def __init__(self, num_classes=136):
            super().__init__()
            self.model_name = 'resnet18'
            self.model = models.resnet18()
            self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        def forward(self, x):
            x = self.model(x)
            return x

    import sys

    def print_overwrite(step, total_step, loss, operation):
        sys.stdout.write('\r')
        if operation == 'train':
            sys.stdout.write("Train Steps: %d/%d  Loss: %.6f " % (step, total_step, loss))
        else:
            sys.stdout.write("Valid Steps: %d/%d  Loss: %.6f " % (step, total_step, loss))

        sys.stdout.flush()

    torch.autograd.set_detect_anomaly(True)
    network = Network()
    network.cuda()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(network.parameters(), lr=0.0001)

    loss_min = np.inf
    num_epochs = 2000

    start_time = time.time()
    for epoch in range(1, num_epochs + 1):

        loss_train = 0
        loss_valid = 0
        running_loss = 0

        network.train()
        for step, (images, landmarks) in enumerate(train_loader, 1):
            images = images.cuda()
            landmarks = landmarks.view(landmarks.size(0), -1).cuda()

            predictions = network(images)

            # clear all the gradients before calculating them
            optimizer.zero_grad()

            # find the loss for the current step
            loss_train_step = criterion(predictions, landmarks)

            # calculate the gradients
            loss_train_step.backward()

            # update the parameters
            optimizer.step()

            loss_train += loss_train_step.item()
            running_loss = loss_train / step

            print_overwrite(step, len(train_loader), running_loss, 'train')

        network.eval()
        with torch.no_grad():
            for step, (images, landmarks) in enumerate(valid_loader, 1):
                images = images.cuda()
                landmarks = landmarks.view(landmarks.size(0), -1).cuda()

                predictions = network(images)

                # find the loss for the current step
                loss_valid_step = criterion(predictions, landmarks)

                loss_valid += loss_valid_step.item()
                running_loss = loss_valid / step

                print_overwrite(step, len(valid_loader), running_loss, 'valid')

        loss_train /= len(train_loader)
        loss_valid /= len(valid_loader)

        print('\n--------------------------------------------------')
        print('Epoch: {}  Train Loss: {:.6f}  Valid Loss: {:.6f}'.format(epoch, loss_train, loss_valid))
        print('--------------------------------------------------')

        if loss_valid < loss_min:
            loss_min = loss_valid
            save_path_cuda = f'weights_3000_epochs_cuda/face_landmarks_epoch_{epoch}.pth'
            torch.save(network.state_dict(), save_path_cuda)
            print(f"\nMinimum Validation Loss of {loss_min:.6f} at epoch {epoch}/{num_epochs}")
            print('Model Saved\n')

    print('Training Complete')
    print(f"Total Elapsed Time : {time.time() - start_time} s")


if __name__ == "__main__":
    main()
