import os
import time
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imutils
from math import radians, cos, sin
import xml.etree.ElementTree as ET

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader

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
        color_jitter = transforms.ColorJitter(brightness=0.5,
                                              contrast=0.5,
                                              saturation=0.5,
                                              hue=0.2)
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

    def random_eye_crop(self, image, landmarks):
        eye_left = landmarks[36:42]
        eye_right = landmarks[42:48]
        eye_box = np.concatenate((eye_left, eye_right), axis=0)

        if eye_box.size == 0:
            return image, landmarks

        min_x, min_y = np.min(eye_box, axis=0).astype(int)
        max_x, max_y = np.max(eye_box, axis=0).astype(int)

        padding = 10
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(image.size[0], max_x + padding)
        max_y = min(image.size[1], max_y + padding)

        image = TF.crop(image, min_y, min_x, max_y - min_y, max_x - min_x)

        img_shape = np.array(image).shape
        landmarks = torch.tensor(landmarks) - torch.tensor([[min_x, min_y]])
        landmarks = landmarks / torch.tensor([img_shape[1], img_shape[0]])
        return image, landmarks

    def __call__(self, image, landmarks, crops):
        image = Image.fromarray(image)
        image, landmarks = self.crop_face(image, landmarks, crops)
        image, landmarks = self.resize(image, landmarks, (224, 224))
        image, landmarks = self.random_eye_crop(image, landmarks)
        image, landmarks = self.color_jitter(image, landmarks)
        image, landmarks = self.brightness_adjust(image, landmarks)
        image, landmarks = self.add_gaussian_noise(image, landmarks)
        image, landmarks = self.rotate(image, landmarks, angle=30)

        image = TF.to_tensor(image)
        image = TF.normalize(image, [0.5], [0.5])
        return image, landmarks

class EyeLandmarksDataset(Dataset):

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
        eye_landmarks = landmarks[36:48]  # Only use eye landmarks (index 36-47)

        if self.transform:
            image, eye_landmarks = self.transform(image, eye_landmarks, self.crops[index])

        eye_landmarks = eye_landmarks - 0.5

        return image, eye_landmarks

if __name__ == "__main__":
    dataset = EyeLandmarksDataset(Transforms())

    len_valid_set = int(0.1 * len(dataset))
    len_train_set = len(dataset) - len_valid_set

    print("The length of Train set is {}".format(len_train_set))
    print("The length of Valid set is {}".format(len_valid_set))

    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [len_train_set, len_valid_set])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=True, num_workers=4)

    class Network(nn.Module):
        def __init__(self, num_classes=24):  # 12 points for 2 eyes
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
            sys.stdout.write("Train Steps: %d/%d  Loss: %.4f " % (step, total_step, loss))
        else:
            sys.stdout.write("Valid Steps: %d/%d  Loss: %.4f " % (step, total_step, loss))

        sys.stdout.flush()

    torch.autograd.set_detect_anomaly(True)
    network = Network()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(network.parameters(), lr=0.0001)

    loss_min = np.inf
    num_epochs = 5000

    # Create directory if it doesn't exist
    if not os.path.exists('weights_2000_epochs'):
        os.makedirs('weights_2000_epochs')

    start_time = time.time()
    for epoch in range(1, num_epochs + 1):

        loss_train = 0
        loss_valid = 0
        running_loss = 0

        network.train()
        for step, (images, eye_landmarks) in enumerate(train_loader, 1):
            images = images
            eye_landmarks = eye_landmarks.view(eye_landmarks.size(0), -1)

            predictions = network(images)

            optimizer.zero_grad()

            loss_train_step = criterion(predictions, eye_landmarks)

            loss_train_step.backward()

            optimizer.step()

            loss_train += loss_train_step.item()
            running_loss = loss_train / step

            print_overwrite(step, len(train_loader), running_loss, 'train')

        network.eval()
        with torch.no_grad():
            for step, (images, eye_landmarks) in enumerate(valid_loader, 1):
                images = images
                eye_landmarks = eye_landmarks.view(eye_landmarks.size(0), -1)

                predictions = network(images)

                loss_valid_step = criterion(predictions, eye_landmarks)

                loss_valid += loss_valid_step.item()
                running_loss = loss_valid / step

                print_overwrite(step, len(valid_loader), running_loss, 'valid')

        loss_train /= len(train_loader)
        loss_valid /= len(valid_loader)

        print('\n--------------------------------------------------')
        print(f'Epoch: {epoch}  Train Loss: {loss_train:.4f}  Valid Loss: {loss_valid:.4f}')
        print('--------------------------------------------------')

        # Save model with the specified format
        model_save_path = f'weights_2000_epochs/eyes_landmarks_epoch_{epoch}.pth'
        if loss_valid < loss_min:
            loss_min = loss_valid
            torch.save(network.state_dict(), model_save_path)
            print(f"\nMinimum Validation Loss of {loss_min:.4f} at epoch {epoch}/{num_epochs}")
            print(f'Model Saved at {model_save_path}\n')

    print('Training Complete')
    print(f"Total Elapsed Time : {time.time() - start_time} s")

    with torch.no_grad():
        best_network = Network()
        best_network.load_state_dict(torch.load(model_save_path))
        best_network.eval()

        images, eye_landmarks = next(iter(valid_loader))

        eye_landmarks = (eye_landmarks + 0.5) * 224

        predictions = (best_network(images) + 0.5) * 224
        predictions = predictions.view(-1, 12, 2)

        plt.figure(figsize=(10, 40))

        for img_num in range(8):
            plt.subplot(8, 1, img_num + 1)
            plt.imshow(images[img_num].numpy().transpose(1, 2, 0).squeeze(), cmap='gray')
            plt.scatter(predictions[img_num, :, 0], predictions[img_num, :, 1], c='r', s=5)
            plt.scatter(eye_landmarks[img_num, :, 0], eye_landmarks[img_num, :, 1], c='g', s=5)

    print(f'Total number of test images: {len(valid_dataset)}')

    end_time = time.time()
    print(f"Elapsed Time : {end_time - start_time} s")



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
# class EyeLandmarksDataset(Dataset):
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
#         eye_landmarks = landmarks[36:48]  # Only use eye landmarks (index 36-47)
#
#         if self.transform:
#             image, eye_landmarks = self.transform(image, eye_landmarks, self.crops[index])
#
#         eye_landmarks = eye_landmarks - 0.5
#
#         return image, eye_landmarks
#
# if __name__ == "__main__":
#     dataset = EyeLandmarksDataset(Transforms())
#
#     len_valid_set = int(0.1 * len(dataset))
#     len_train_set = len(dataset) - len_valid_set
#
#     print("The length of Train set is {}".format(len_train_set))
#     print("The length of Valid set is {}".format(len_valid_set))
#
#     train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [len_train_set, len_valid_set])
#
#     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
#     valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=True, num_workers=4)
#
#     class Network(nn.Module):
#         def __init__(self, num_classes=24):  # 12 points for 2 eyes
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
#         for step, (images, eye_landmarks) in enumerate(train_loader, 1):
#             images = images
#             eye_landmarks = eye_landmarks.view(eye_landmarks.size(0), -1)
#
#             predictions = network(images)
#
#             optimizer.zero_grad()
#
#             loss_train_step = criterion(predictions, eye_landmarks)
#
#             loss_train_step.backward()
#
#             optimizer.step()
#
#             loss_train += loss_train_step.item()
#             running_loss = loss_train / step
#
#             print_overwrite(step, len(train_loader), running_loss, 'train')
#
#         network.eval()
#         with torch.no_grad():
#             for step, (images, eye_landmarks) in enumerate(valid_loader, 1):
#                 images = images
#                 eye_landmarks = eye_landmarks.view(eye_landmarks.size(0), -1)
#
#                 predictions = network(images)
#
#                 loss_valid_step = criterion(predictions, eye_landmarks)
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
#             torch.save(network.state_dict(), 'eye_landmarks.pth')
#             print(f"\nMinimum Validation Loss of {loss_min:.4f} at epoch {epoch}/{num_epochs}")
#             print('Model Saved\n')
#
#     print('Training Complete')
#     print(f"Total Elapsed Time : {time.time() - start_time} s")
#
#     with torch.no_grad():
#         best_network = Network()
#         best_network.load_state_dict(torch.load('eye_landmarks.pth'))
#         best_network.eval()
#
#         images, eye_landmarks = next(iter(valid_loader))
#
#         eye_landmarks = (eye_landmarks + 0.5) * 224
#
#         predictions = (best_network(images) + 0.5) * 224
#         predictions = predictions.view(-1, 12, 2)
#
#         plt.figure(figsize=(10, 40))
#
#         for img_num in range(8):
#             plt.subplot(8, 1, img_num + 1)
#             plt.imshow(images[img_num].numpy().transpose(1, 2, 0).squeeze(), cmap='gray')
#             plt.scatter(predictions[img_num, :, 0], predictions[img_num, :, 1], c='r', s=5)
#             plt.scatter(eye_landmarks[img_num, :, 0], eye_landmarks[img_num, :, 1], c='g', s=5)
#
#     print(f'Total number of test images: {len(valid_dataset)}')
#
#     end_time = time.time()
#     print(f"Elapsed Time : {end_time - start_time} s")
