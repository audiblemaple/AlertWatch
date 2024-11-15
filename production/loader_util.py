import torch
from face_landmarks_detection.utils.network import Network
import cv2

def load_model(weights_path, class_num, device):
    """
    Loads the neural network model with the specified weights and configuration.
    """
    network = Network(num_classes=class_num)
    network.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
    network.eval()
    return network

def load_face_cascade(cascade_path):
    """
    Loads the Haar Cascade for face detection.
    """
    return cv2.CascadeClassifier(cascade_path)