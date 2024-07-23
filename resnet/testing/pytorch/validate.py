import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms.functional as TF

#######################################################################
image_path = '../../pics/pic.jpg'

# face = False
face = True

if face:
    weights_path = '../../models/face_landmarks_epoch_1412.pth'
    num_classes = 136

else:
    weights_path = '../../models/face_landmarks_epoch_1412.pth'
    num_classes = 24

frontal_face_cascade_path = '../../../haarcascades/haarcascade_frontalface_default.xml'


#######################################################################
class Network(nn.Module):
    def __init__(self, num_classes=136):
        super().__init__()
        self.model_name = 'resnet18'
        self.model = models.resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x


#######################################################################
face_cascade = cv2.CascadeClassifier(frontal_face_cascade_path)

best_network = Network(num_classes)
best_network.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
best_network.eval()

image = cv2.imread(image_path)
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width, _ = image.shape

faces = face_cascade.detectMultiScale(grayscale_image, 1.1, 4)

all_landmarks = []
for (x, y, w, h) in faces:
    image = grayscale_image[y:y + h, x:x + w]
    image = TF.resize(Image.fromarray(image), size=(224, 224))
    image = TF.to_tensor(image)
    image = TF.normalize(image, [0.5], [0.5])

    with torch.no_grad():
        landmarks = best_network(image.unsqueeze(0))

    if num_classes == 24:
        landmarks = (landmarks.view(12, 2).detach().numpy() + 0.5) * np.array([[w, h]]) + np.array([[x, y]])
    else:
        landmarks = (landmarks.view(68, 2).detach().numpy() + 0.5) * np.array([[w, h]]) + np.array([[x, y]])
    all_landmarks.append(landmarks)

plt.figure()
plt.imshow(display_image)
for landmarks in all_landmarks:
    plt.scatter(landmarks[:, 0], landmarks[:, 1], c='c', s=5)

plt.show()
