"""
Module: Transforms

This module provides a class `Transforms` that applies various transformations to images and their corresponding landmarks. The transformations include rotation, resizing, color jitter, brightness adjustment, Gaussian noise addition, face cropping, and eye cropping.

Classes:
    Transforms

Methods:
    rotate(image, landmarks, angle)
        Rotates the image and landmarks by a random angle within the specified range.

    resize(image, landmarks, img_size)
        Resizes the image to the specified size.

    color_jitter(image, landmarks)
        Applies random color jitter to the image.

    brightness_adjust(image, landmarks)
        Adjusts the brightness of the image by a random factor.

    add_gaussian_noise(image, landmarks)
        Adds Gaussian noise to the image.

    crop_face(image, landmarks, crops)
        Crops the face region from the image based on the specified crop parameters.

    random_eye_crop(image, landmarks)
        Crops the region around the eyes from the image.

    __call__(image, landmarks, crops)
        Applies a series of transformations to the image and landmarks.

"""

import random
import numpy as np
from PIL import Image
import imutils
from math import radians, cos, sin
import torch
import torchvision.transforms.functional as transforms_functional
from torchvision import transforms


class Transforms:
    """
    A class to apply various transformations to images and their corresponding landmarks.
    """
    def __init__(self):
        """
        Initializes the Transforms class.
        """
        pass

    @staticmethod
    def rotate(image, landmarks, angle):
        """
        Rotates the image and landmarks by a random angle within the specified range.

        Args:
            image (PIL.Image or numpy.ndarray): The input image.
            landmarks (numpy.ndarray): The landmarks corresponding to the image.
            angle (float): The maximum absolute value of the rotation angle.

        Returns:
            PIL.Image: The rotated image.
            numpy.ndarray: The rotated landmarks.
        """
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

    @staticmethod
    def resize(image, landmarks, img_size):
        """
        Resizes the image to the specified size.

        Args:
            image (PIL.Image or numpy.ndarray): The input image.
            landmarks (numpy.ndarray): The landmarks corresponding to the image.
            img_size (tuple): The desired output size.

        Returns:
            PIL.Image: The resized image.
            numpy.ndarray: The landmarks (unchanged).
        """
        image = transforms_functional.resize(image, img_size)
        return image, landmarks

    @staticmethod
    def color_jitter(image, landmarks):
        """
        Applies random color jitter to the image.

        Args:
            image (PIL.Image or numpy.ndarray): The input image.
            landmarks (numpy.ndarray): The landmarks corresponding to the image.

        Returns:
            PIL.Image: The color jittered image.
            numpy.ndarray: The landmarks (unchanged).
        """
        color_jitter = transforms.ColorJitter(brightness=0.5,
                                              contrast=0.5,
                                              saturation=0.5,
                                              hue=0.2)
        image = color_jitter(image)
        return image, landmarks

    @staticmethod
    def brightness_adjust(image, landmarks):
        """
        Adjusts the brightness of the image by a random factor.

        Args:
            image (PIL.Image or numpy.ndarray): The input image.
            landmarks (numpy.ndarray): The landmarks corresponding to the image.

        Returns:
            PIL.Image: The brightness adjusted image.
            numpy.ndarray: The landmarks (unchanged).
        """
        brightness_factor = random.uniform(0.5, 1.5)
        image = transforms_functional.adjust_brightness(image, brightness_factor)
        return image, landmarks

    @staticmethod
    def add_gaussian_noise(image, landmarks):
        """
        Adds Gaussian noise to the image.

        Args:
            image (PIL.Image or numpy.ndarray): The input image.
            landmarks (numpy.ndarray): The landmarks corresponding to the image.

        Returns:
            PIL.Image: The image with added Gaussian noise.
            numpy.ndarray: The landmarks (unchanged).
        """
        noise = np.random.normal(0, 0.1, np.array(image).shape)
        image = np.clip(np.array(image) + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(image), landmarks

    @staticmethod
    def crop_face(image, landmarks, crops):
        """
        Crops the face region from the image based on the specified crop parameters.

        Args:
            image (PIL.Image or numpy.ndarray): The input image.
            landmarks (numpy.ndarray): The landmarks corresponding to the image.
            crops (dict): A dictionary containing crop parameters ('left', 'top', 'width', 'height').

        Returns:
            PIL.Image: The cropped face image.
            numpy.ndarray: The cropped landmarks.
        """
        left = int(crops['left'])
        top = int(crops['top'])
        width = int(crops['width'])
        height = int(crops['height'])

        image = transforms_functional.crop(image, top, left, height, width)

        img_shape = np.array(image).shape
        landmarks = torch.tensor(landmarks) - torch.tensor([[left, top]])
        landmarks = landmarks / torch.tensor([img_shape[1], img_shape[0]])
        return image, landmarks

    @staticmethod
    def random_eye_crop(image, landmarks):
        """
        Crops the region around the eyes from the image.

        Args:
            image (PIL.Image or numpy.ndarray): The input image.
            landmarks (numpy.ndarray): The landmarks corresponding to the image.

        Returns:
            PIL.Image: The cropped eye region image.
            numpy.ndarray: The cropped eye landmarks.
        """
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

        image = transforms_functional.crop(image, min_y, min_x, max_y - min_y, max_x - min_x)

        img_shape = np.array(image).shape
        landmarks = torch.tensor(landmarks) - torch.tensor([[min_x, min_y]])
        landmarks = landmarks / torch.tensor([img_shape[1], img_shape[0]])
        return image, landmarks

    def __call__(self, image, landmarks, crops):
        """
        Applies a series of transformations to the image and landmarks.

        Args:
            image (PIL.Image or numpy.ndarray): The input image.
            landmarks (numpy.ndarray): The landmarks corresponding to the image.
            crops (dict): A dictionary containing crop parameters ('left', 'top', 'width', 'height').

        Returns:
            torch.Tensor: The transformed image.
            numpy.ndarray: The transformed landmarks.
        """
        image = Image.fromarray(image)
        image, landmarks = self.crop_face(image, landmarks, crops)
        image, landmarks = self.resize(image, landmarks, (224, 224))

        # TODO: Decide if I will run it only if its the eyes training.
        # image, landmarks = self.random_eye_crop(image, landmarks)

        image, landmarks = self.color_jitter(image, landmarks)
        image, landmarks = self.brightness_adjust(image, landmarks)
        image, landmarks = self.add_gaussian_noise(image, landmarks)
        image, landmarks = self.rotate(image, landmarks, angle=30)

        image = transforms_functional.to_tensor(image)
        image = transforms_functional.normalize(image, [0.5], [0.5])
        return image, landmarks
