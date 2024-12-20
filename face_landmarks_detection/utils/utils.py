import sys
import torch
from torch.utils.data import Dataset, DataLoader
from face_landmarks_detection.utils.landmarksDataSet import LandmarksDataset
import cv2

def print_overwrite(step, total_step, loss, operation):
    sys.stdout.write('\r')
    if operation == 'train':
        sys.stdout.write("Train Steps: %d/%d  Loss: %.6f " % (step, total_step, loss))
    else:
        sys.stdout.write("Valid Steps: %d/%d  Loss: %.6f " % (step, total_step, loss))

    sys.stdout.flush()


def print_epoch_result(epoch, loss_train, loss_valid):
    print('\n--------------------------------------------------')
    print(f'Epoch: {epoch}  Train Loss: {loss_train:.6f}  Valid Loss: {loss_valid:.6f}')
    print('--------------------------------------------------')


def create_datasets(mode, transform, split_ratio=0.1):
    dataset = LandmarksDataset(mode=mode, transform=transform)
    len_valid_set = int(split_ratio * len(dataset))
    len_train_set = len(dataset) - len_valid_set
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [len_train_set, len_valid_set])
    return train_dataset, valid_dataset, len_train_set, len_valid_set


def create_dataloaders(train_dataset, valid_dataset, batch_size_train=64, batch_size_valid=8, num_workers=4):
    train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size_valid, shuffle=True, num_workers=num_workers)
    return train_loader, valid_loader

def init_cv_cap(width=640, height=640, fps=30):
    if 'win' in sys.platform:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def init_cascade():
    haar_cascade_path = '../../haarcascades/haarcascade_frontalface_alt.xml'
    face_cascade = cv2.CascadeClassifier(haar_cascade_path)
    if face_cascade.empty():
        raise IOError(f"Could not load Haar cascade file from {haar_cascade_path}")

    return face_cascade