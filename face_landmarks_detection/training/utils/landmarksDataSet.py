"""
Module: LandmarksDataset

This module provides a class `LandmarksDataset` that defines a dataset for either eye or face landmarks.

Classes:
    LandmarksDataset

Methods:
    __init__(self, mode='face', transform=None)
        Initializes the LandmarksDataset class with the specified mode ('eyes' or 'face') and optional transform.

    __len__(self)
        Returns the length of the dataset.

    __getitem__(self, index)
        Retrieves the image and landmarks at the specified index.
"""

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset


class LandmarksDataset(Dataset):
    """
    A dataset class for either eye or face landmarks.

    Args:
        mode (str): Specifies whether the dataset is for 'eyes' or 'face'.
        transform (callable, optional): A function/transform to apply to the images and landmarks.
    """

    def __init__(self, mode='face', transform=None):
        """
        Initializes the LandmarksDataset class with the specified mode ('eyes' or 'face') and optional transform.

        Args:
            mode (str): Specifies whether the dataset is for 'eyes' or 'face'.
            transform (callable, optional): A function/transform to apply to the images and landmarks.
        """
        assert mode in ['eyes', 'face'], "mode should be 'eyes' or 'face'"
        self.mode = mode
        self.transform = transform
        self.image_filenames = []
        self.landmarks = []
        self.crops = []
        self.root_dir = '../../../../ibug_300W_large_face_landmark_dataset'

        tree = ET.parse(os.path.join(self.root_dir, 'labels_ibug_300W_train.xml'))
        root = tree.getroot()

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
        """
        Returns the length of the dataset.

        Returns:
            int: The number of images in the dataset.
        """
        return len(self.image_filenames)

    def __getitem__(self, index):
        """
        Retrieves the image and landmarks at the specified index.

        Args:
            index (int): The index of the image and landmarks to retrieve.

        Returns:
            tuple: A tuple containing the image and landmarks.
        """
        image = cv2.imread(self.image_filenames[index], 0)
        landmarks = self.landmarks[index]

        if self.mode == 'eyes':
            landmarks = landmarks[36:48]  # Only use eye landmarks (index 36-47)

        if self.transform:
            image, landmarks = self.transform(image, landmarks, self.crops[index])

        landmarks = landmarks - 0.5

        return image, landmarks
