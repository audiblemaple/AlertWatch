import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader

from face_landmarks_detection.training.utils.transforms import Transforms
from face_landmarks_detection.training.utils.network import Network
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

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)




    def print_overwrite(step, total_step, loss, operation):
        sys.stdout.write('\r')
        if operation == 'train':
            sys.stdout.write("Train Steps: %d/%d  Loss: %.6f " % (step, total_step, loss))
        else:
            sys.stdout.write("Valid Steps: %d/%d  Loss: %.6f " % (step, total_step, loss))

        sys.stdout.flush()


    # Set device to CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.autograd.set_detect_anomaly(True)
    network = Network(num_classes=24).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(network.parameters(), lr=0.0001)

    loss_min = np.inf
    num_epochs = 3000

    # Create directory if it doesn't exist
    if not os.path.exists('output'):
        os.makedirs('output')

    start_time = time.time()
    for epoch in range(1, num_epochs + 1):

        loss_train = 0
        loss_valid = 0
        running_loss = 0

        network.train()
        for step, (images, eye_landmarks) in enumerate(train_loader, 1):
            images = images.to(device, non_blocking=True)
            eye_landmarks = eye_landmarks.view(eye_landmarks.size(0), -1).to(device, non_blocking=True)

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
                images = images.to(device, non_blocking=True)
                eye_landmarks = eye_landmarks.view(eye_landmarks.size(0), -1).to(device, non_blocking=True)

                predictions = network(images)

                loss_valid_step = criterion(predictions, eye_landmarks)

                loss_valid += loss_valid_step.item()
                running_loss = loss_valid / step

                print_overwrite(step, len(valid_loader), running_loss, 'valid')

        loss_train /= len(train_loader)
        loss_valid /= len(valid_loader)

        print('\n--------------------------------------------------')
        print(f'Epoch: {epoch}  Train Loss: {loss_train:.6f}  Valid Loss: {loss_valid:.6f}')
        print('--------------------------------------------------')

        # Save model with the specified format
        model_save_path = f'output/eyes_landmarks_epoch_{epoch}.pth'
        if loss_valid < loss_min:
            loss_min = loss_valid
            torch.save(network.state_dict(), model_save_path)
            print(f"\nMinimum Validation Loss of {loss_min:.6f} at epoch {epoch}/{num_epochs}")
            print(f'Model Saved at {model_save_path}\n')

    print('Training Complete')
    print(f"Total Elapsed Time : {time.time() - start_time} s")

    with torch.no_grad():
        best_network = Network(num_classes=24).to(device)
        best_network.load_state_dict(torch.load(model_save_path))
        best_network.eval()

        images, eye_landmarks = next(iter(valid_loader))

        images = images.to(device)
        eye_landmarks = (eye_landmarks + 0.5) * 224

        predictions = (best_network(images) + 0.5) * 224
        predictions = predictions.view(-1, 12, 2).cpu()

        plt.figure(figsize=(10, 40))

        for img_num in range(8):
            plt.subplot(8, 1, img_num + 1)
            plt.imshow(images[img_num].cpu().numpy().transpose(1, 2, 0).squeeze(), cmap='gray')
            plt.scatter(predictions[img_num, :, 0], predictions[img_num, :, 1], c='r', s=5)
            plt.scatter(eye_landmarks[img_num, :, 0], eye_landmarks[img_num, :, 1], c='g', s=5)

    print(f'Total number of test images: {len(valid_dataset)}')

    end_time = time.time()
    print(f"Elapsed Time : {end_time - start_time} s")
    print("Models checkpoints will be in the ./output directory")
