import sys
import torch
from torch.utils.data import Dataset, DataLoader
from face_landmarks_detection.training.utils.landmarksDataSet import LandmarksDataset


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
