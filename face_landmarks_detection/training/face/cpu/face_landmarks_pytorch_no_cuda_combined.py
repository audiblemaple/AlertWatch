import os
import time
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from face_landmarks_detection.utils.transforms import Transforms
from face_landmarks_detection.utils.network import Network
from face_landmarks_detection.utils.utils import *


def main():
    """
    Main function for training and validating the facial landmarks detection model.

    This function handles the following tasks:
    - Parsing command-line arguments
    - Loading and splitting the dataset
    - Initializing the neural network model, loss function, and optimizer
    - Training the model for a specified number of epochs
    - Validating the model and saving checkpoints
    - Visualizing the results

    Command-line arguments:
    -e, --epoch: Number of epochs to train the model (default: 100)
    -m, --mode: Mode of detection, either 'face' or 'eyes' (default: face)
    -d, --dev: Device to run the model on 'cpu', 'cuda' or 'rocm' (default: cpu)
    -o, --out: Output directory to save the model checkpoints (default: output)
    """
    parser = argparse.ArgumentParser(description="Facial Landmarks Detection Training")
    parser.add_argument("-e", "--epoch",
                        default=100,
                        type=int,
                        help="Number of epochs to train the model.",
                        required=False
                        )
    parser.add_argument("-m", "--mode",
                        default="face",
                        type=str,
                        help="face / eyes",
                        required=False
                        )
    parser.add_argument("-d", "--dev",
                        default="cpu",
                        type=str,
                        help="cpu, cuda:0 or rocm-(not implemented).",
                        required=False
                        )
    parser.add_argument("-o", "--out",
                        default="output",
                        type=str,
                        help="Output directory to save the model checkpoints to.",
                        required=False
                        )
    args = parser.parse_args()

    # Load the dataset with the specified mode and transformations
    dataset = LandmarksDataset(mode=args.mode, transform=Transforms())

    # Split the dataset into training and validation sets
    len_valid_set = int(0.1 * len(dataset))
    len_train_set = len(dataset) - len_valid_set

    print(f"The length of Train set is {len_train_set}")
    print(f"The length of Valid set is {len_valid_set}")

    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [len_train_set, len_valid_set])
    num_cores = os.cpu_count()

    # Create data loaders for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=num_cores)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=True, num_workers=num_cores)

    # Determine the number of output classes based on the mode
    class_num = 136 if args.mode == "face" else 24

    # Enable anomaly detection for autograd
    torch.autograd.set_detect_anomaly(True)

    # Initialize the network, loss function, and optimizer
    network = Network(num_classes=class_num)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(network.parameters(), lr=0.0001)

    # Track the minimum validation loss for saving the best model
    loss_min = np.inf
    num_epochs = args.epoch

    # Create output directory if it doesn't exist
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    # Training loop
    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        loss_train = 0
        loss_valid = 0
        running_loss = 0

        network.train()
        for step, (images, landmarks) in enumerate(train_loader, 1):
            images = images.to(args.dev)
            landmarks = landmarks.view(landmarks.size(0), -1).to(args.dev)

            predictions = network(images)

            # Clear all the gradients before calculating them
            optimizer.zero_grad()

            # Find the loss for the current step
            loss_train_step = criterion(predictions, landmarks)

            # Calculate the gradients
            loss_train_step.backward()

            # Update the parameters
            optimizer.step()

            loss_train += loss_train_step.item()
            running_loss = loss_train / step

            # Print training progress
            print_overwrite(step, len(train_loader), running_loss, 'train')

        network.eval()
        with torch.no_grad():
            for step, (images, landmarks) in enumerate(valid_loader, 1):
                images = images.to(args.dev)
                landmarks = landmarks.view(landmarks.size(0), -1).to(args.dev)

                predictions = network(images)

                # Find the loss for the current step
                loss_valid_step = criterion(predictions, landmarks)

                loss_valid += loss_valid_step.item()
                running_loss = loss_valid / step

                # Print validation progress
                print_overwrite(step, len(valid_loader), running_loss, 'valid')

        loss_train /= len(train_loader)
        loss_valid /= len(valid_loader)

        # Print epoch results
        print_epoch_result(epoch, loss_train, loss_valid)

        # Save the model if it has the minimum validation loss
        model_save_path = f'{args.out}/{args.mode}_landmarks_epoch_{epoch}.pth'
        if loss_valid < loss_min:
            loss_min = loss_valid
            torch.save(network.state_dict(), model_save_path)
            print(f"\nMinimum Validation Loss of {loss_min:.4f} at epoch {epoch}/{num_epochs}")
            print(f'Model Saved at {model_save_path}\n')

    print('Training Complete')
    print(f"Total Elapsed Time : {time.time() - start_time} s")

    # Validation with visualization
    start_time = time.time()

    with torch.no_grad():
        best_network = Network(num_classes=class_num)
        best_network.load_state_dict(torch.load(model_save_path))
        best_network.eval()

        images, landmarks = next(iter(valid_loader))

        landmarks = (landmarks + 0.5) * 224

        predictions = (best_network(images) + 0.5) * 224
        predictions = predictions.view(-1, 68, 2)

        plt.figure(figsize=(10, 40))

        for img_num in range(8):
            plt.subplot(8, 1, img_num + 1)
            plt.imshow(images[img_num].cpu().numpy().transpose(1, 2, 0).squeeze(), cmap='gray')
            plt.scatter(predictions[img_num, :, 0].cpu(), predictions[img_num, :, 1].cpu(), c='r', s=5)
            plt.scatter(landmarks[img_num, :, 0].cpu(), landmarks[img_num, :, 1].cpu(), c='g', s=5)

    print(f'Total number of test images: {len(valid_dataset)}')

    end_time = time.time()
    print(f"Elapsed Time : {end_time - start_time} s")
    print(f"Model checkpoints will be in the ./{args.out} directory")


if __name__ == "__main__":
    main()









# import time
# import cv2
# import os
# import random
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import imutils
# import matplotlib.image as mpimg
# from collections import OrderedDict
# from skimage import io, transform
# from math import radians, cos, sin
# import xml.etree.ElementTree as ET
#
# import torch
# import torchvision
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import torchvision.transforms.functional as TF
# from torchvision import datasets, models, transforms
# from torch.utils.data import Dataset, DataLoader
#
#
# class Transforms():
#     def __init__(self):
#         pass
#
#     def rotate(self, image, landmarks, angle):
#         angle = random.uniform(-angle, +angle)
#
#         transformation_matrix = torch.tensor([
#             [+cos(radians(angle)), -sin(radians(angle))],
#             [+sin(radians(angle)), +cos(radians(angle))]
#         ])
#
#         image = imutils.rotate(np.array(image), angle)
#
#         landmarks = landmarks - 0.5
#         new_landmarks = np.matmul(landmarks, transformation_matrix)
#         new_landmarks = new_landmarks + 0.5
#         return Image.fromarray(image), new_landmarks
#
#     def resize(self, image, landmarks, img_size):
#         image = TF.resize(image, img_size)
#         return image, landmarks
#
#     def color_jitter(self, image, landmarks):
#         color_jitter = transforms.ColorJitter(brightness=0.5,
#                                               contrast=0.5,
#                                               saturation=0.5,
#                                               hue=0.2)
#         image = color_jitter(image)
#         return image, landmarks
#
#     def brightness_adjust(self, image, landmarks):
#         brightness_factor = random.uniform(0.5, 1.5)
#         image = TF.adjust_brightness(image, brightness_factor)
#         return image, landmarks
#
#     def add_gaussian_noise(self, image, landmarks):
#         noise = np.random.normal(0, 0.1, np.array(image).shape)
#         image = np.clip(np.array(image) + noise, 0, 255).astype(np.uint8)
#         return Image.fromarray(image), landmarks
#
#     def crop_face(self, image, landmarks, crops):
#         left = int(crops['left'])
#         top = int(crops['top'])
#         width = int(crops['width'])
#         height = int(crops['height'])
#
#         image = TF.crop(image, top, left, height, width)
#
#         img_shape = np.array(image).shape
#         landmarks = torch.tensor(landmarks) - torch.tensor([[left, top]])
#         landmarks = landmarks / torch.tensor([img_shape[1], img_shape[0]])
#         return image, landmarks
#
#     def random_eye_crop(self, image, landmarks):
#         eye_left = landmarks[36:42]
#         eye_right = landmarks[42:48]
#         eye_box = np.concatenate((eye_left, eye_right), axis=0)
#
#         if eye_box.size == 0:
#             return image, landmarks
#
#         min_x, min_y = np.min(eye_box, axis=0).astype(int)
#         max_x, max_y = np.max(eye_box, axis=0).astype(int)
#
#         padding = 10
#         min_x = max(0, min_x - padding)
#         min_y = max(0, min_y - padding)
#         max_x = min(image.size[0], max_x + padding)
#         max_y = min(image.size[1], max_y + padding)
#
#         image = TF.crop(image, min_y, min_x, max_y - min_y, max_x - min_x)
#
#         img_shape = np.array(image).shape
#         landmarks = torch.tensor(landmarks) - torch.tensor([[min_x, min_y]])
#         landmarks = landmarks / torch.tensor([img_shape[1], img_shape[0]])
#         return image, landmarks
#
#     def __call__(self, image, landmarks, crops):
#         image = Image.fromarray(image)
#         image, landmarks = self.crop_face(image, landmarks, crops)
#         image, landmarks = self.resize(image, landmarks, (224, 224))
#         image, landmarks = self.random_eye_crop(image, landmarks)
#         image, landmarks = self.color_jitter(image, landmarks)
#         image, landmarks = self.brightness_adjust(image, landmarks)
#         image, landmarks = self.add_gaussian_noise(image, landmarks)
#         image, landmarks = self.rotate(image, landmarks, angle=50)
#
#         image = TF.to_tensor(image)
#         image = TF.normalize(image, [0.5], [0.5])
#         return image, landmarks
#
#
#
# class FaceLandmarksDataset(Dataset):
#
#     def __init__(self, transform=None):
#         tree = ET.parse('ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml')
#         root = tree.getroot()
#
#         self.image_filenames = []
#         self.landmarks = []
#         self.crops = []
#         self.transform = transform
#         self.root_dir = 'ibug_300W_large_face_landmark_dataset'
#
#         for filename in root[2]:
#             self.image_filenames.append(os.path.join(self.root_dir, filename.attrib['file']))
#
#             self.crops.append(filename[0].attrib)
#
#             landmark = []
#             for num in range(68):
#                 x_coordinate = int(filename[0][num].attrib['x'])
#                 y_coordinate = int(filename[0][num].attrib['y'])
#                 landmark.append([x_coordinate, y_coordinate])
#             self.landmarks.append(landmark)
#
#         self.landmarks = np.array(self.landmarks).astype('float32')
#
#         assert len(self.image_filenames) == len(self.landmarks)
#
#     def __len__(self):
#         return len(self.image_filenames)
#
#     def __getitem__(self, index):
#         image = cv2.imread(self.image_filenames[index], 0)
#         landmarks = self.landmarks[index]
#
#         if self.transform:
#             image, landmarks = self.transform(image, landmarks, self.crops[index])
#
#         landmarks = landmarks - 0.5
#
#         return image, landmarks
#
# if __name__ == "__main__":
#     dataset = FaceLandmarksDataset(Transforms())
#
#     # Split the dataset into testing and test sets
#     len_valid_set = int(0.1 * len(dataset))
#     len_train_set = len(dataset) - len_valid_set
#
#     print("The length of Train set is {}".format(len_train_set))
#     print("The length of Valid set is {}".format(len_valid_set))
#
#     train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [len_train_set, len_valid_set])
#
#     # Shuffle and batch the datasets
#     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
#     valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=True, num_workers=4)
#
#     class Network(nn.Module):
#         def __init__(self, num_classes=136):
#             super().__init__()
#             self.model_name = 'resnet18'
#             self.model = models.resnet18()
#             self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#             self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
#
#         def forward(self, x):
#             x = self.model(x)
#             return x
#
#     import sys
#
#     def print_overwrite(step, total_step, loss, operation):
#         sys.stdout.write('\r')
#         if operation == 'train':
#             sys.stdout.write("Train Steps: %d/%d  Loss: %.4f " % (step, total_step, loss))
#         else:
#             sys.stdout.write("Valid Steps: %d/%d  Loss: %.4f " % (step, total_step, loss))
#
#         sys.stdout.flush()
#
#     torch.autograd.set_detect_anomaly(True)
#     network = Network()
#
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(network.parameters(), lr=0.0001)
#
#     loss_min = np.inf
#     num_epochs = 200
#
#     start_time = time.time()
#     for epoch in range(1, num_epochs + 1):
#
#         loss_train = 0
#         loss_valid = 0
#         running_loss = 0
#
#         network.train()
#         for step, (images, landmarks) in enumerate(train_loader, 1):
#             images = images
#             landmarks = landmarks.view(landmarks.size(0), -1)
#
#             predictions = network(images)
#
#             # Clear all the gradients before calculating them
#             optimizer.zero_grad()
#
#             # Find the loss for the current step
#             loss_train_step = criterion(predictions, landmarks)
#
#             # Calculate the gradients
#             loss_train_step.backward()
#
#             # Update the parameters
#             optimizer.step()
#
#             loss_train += loss_train_step.item()
#             running_loss = loss_train / step
#
#             print_overwrite(step, len(train_loader), running_loss, 'train')
#
#         network.eval()
#         with torch.no_grad():
#             for step, (images, landmarks) in enumerate(valid_loader, 1):
#                 images = images
#                 landmarks = landmarks.view(landmarks.size(0), -1)
#
#                 predictions = network(images)
#
#                 # Find the loss for the current step
#                 loss_valid_step = criterion(predictions, landmarks)
#
#                 loss_valid += loss_valid_step.item()
#                 running_loss = loss_valid / step
#
#                 print_overwrite(step, len(valid_loader), running_loss, 'valid')
#
#         loss_train /= len(train_loader)
#         loss_valid /= len(valid_loader)
#
#         print('\n--------------------------------------------------')
#         print(f'Epoch: {epoch}  Train Loss: {loss_train:.4f}  Valid Loss: {loss_valid:.4f}')
#         print('--------------------------------------------------')
#
#         if loss_valid < loss_min:
#             loss_min = loss_valid
#             torch.save(network.state_dict(), 'face_landmarks.pth')
#             print(f"\nMinimum Validation Loss of {loss_min:.4f} at epoch {epoch}/{num_epochs}")
#             print('Model Saved\n')
#
#     print('Training Complete')
#     print(f"Total Elapsed Time : {time.time() - start_time} s")
#
#     # Validation with visualization
#     start_time = time.time()
#
#     with torch.no_grad():
#         best_network = Network()
#         best_network.load_state_dict(torch.load('face_landmarks.pth'))
#         best_network.eval()
#
#         images, landmarks = next(iter(valid_loader))
#
#         landmarks = (landmarks + 0.5) * 224
#
#         predictions = (best_network(images) + 0.5) * 224
#         predictions = predictions.view(-1, 68, 2)
#
#         plt.figure(figsize=(10, 40))
#
#         for img_num in range(8):
#             plt.subplot(8, 1, img_num + 1)
#             plt.imshow(images[img_num].numpy().transpose(1, 2, 0).squeeze(), cmap='gray')
#             plt.scatter(predictions[img_num, :, 0], predictions[img_num, :, 1], c='r', s=5)
#             plt.scatter(landmarks[img_num, :, 0], landmarks[img_num, :, 1], c='g', s=5)
#
#     print(f'Total number of test images: {len(valid_dataset)}')
#
#     end_time = time.time()
#     print(f"Elapsed Time : {end_time - start_time} s")
