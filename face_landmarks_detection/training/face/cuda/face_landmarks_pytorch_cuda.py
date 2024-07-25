import time
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from face_landmarks_detection.training.utils.transforms import Transforms
from face_landmarks_detection.training.utils.network import Network

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

        image = torch.from_numpy(image).unsqueeze(0).float()  # Add channel dimension and convert to tensor
        landmarks = torch.from_numpy(landmarks).float()

        return image, landmarks

if __name__ == "__main__":
    dataset = FaceLandmarksDataset(Transforms())

    # Split the dataset into testing and test sets
    len_valid_set = int(0.1 * len(dataset))
    len_train_set = len(dataset) - len_valid_set

    print("The length of Train set is {}".format(len_train_set))
    print("The length of Valid set is {}".format(len_valid_set))

    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [len_train_set, len_valid_set])

    # Shuffle and batch the datasets
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=True, num_workers=4)

    import sys

    def print_overwrite(step, total_step, loss, operation):
        sys.stdout.write('\r')
        if operation == 'train':
            sys.stdout.write("Train Steps: %d/%d  Loss: %.4f " % (step, total_step, loss))
        else:
            sys.stdout.write("Valid Steps: %d/%d  Loss: %.4f " % (step, total_step, loss))

        sys.stdout.flush()

    torch.autograd.set_detect_anomaly(True)
    network = Network(num_classes=136).to(device)  # Move model to GPU if available

    criterion = nn.MSELoss()
    optimizer = optim.Adam(network.parameters(), lr=0.0001)

    loss_min = np.inf
    num_epochs = 5000

    # Create directory if it doesn't exist
    if not os.path.exists('output'):
        os.makedirs('output')

    start_time = time.time()
    for epoch in range(1, num_epochs + 1):

        loss_train = 0
        loss_valid = 0
        running_loss = 0

        network.train()
        for step, (images, landmarks) in enumerate(train_loader, 1):
            images, landmarks = images.to(device), landmarks.to(device)  # Move data to GPU if available

            landmarks = landmarks.view(landmarks.size(0), -1)

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

            print_overwrite(step, len(train_loader), running_loss, 'train')

        network.eval()
        with torch.no_grad():
            for step, (images, landmarks) in enumerate(valid_loader, 1):
                images, landmarks = images.to(device), landmarks.to(device)  # Move data to GPU if available

                landmarks = landmarks.view(landmarks.size(0), -1)

                predictions = network(images)

                # Find the loss for the current step
                loss_valid_step = criterion(predictions, landmarks)

                loss_valid += loss_valid_step.item()
                running_loss = loss_valid / step

                print_overwrite(step, len(valid_loader), running_loss, 'valid')

        loss_train /= len(train_loader)
        loss_valid /= len(valid_loader)

        print('\n--------------------------------------------------')
        print(f'Epoch: {epoch}  Train Loss: {loss_train:.4f}  Valid Loss: {loss_valid:.4f}')
        print('--------------------------------------------------')

        # Save model with the specified format
        model_save_path = f'output/face_landmarks_epoch_{epoch}.pth'
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
        best_network = Network(num_classes=136).to(device)
        best_network.load_state_dict(torch.load(model_save_path))
        best_network.eval()

        images, landmarks = next(iter(valid_loader))
        images, landmarks = images.to(device), landmarks.to(device)

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
    print("Models checkpoints will be in the ./output directory")
