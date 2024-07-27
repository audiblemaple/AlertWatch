import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision.transforms.functional as transforms_functional
from face_landmarks_detection.utils.network import Network


image_path = '../../pics/pic.jpg'

# face = False
face = True

if face:
    weights_path = '../../models/face/face_landmarks_epoch_3559.pth'
    num_classes = 136

else:
    weights_path = '../../models/face/face_landmarks_epoch_3559.pth'
    num_classes = 24

frontal_face_cascade_path = '../../../haarcascades/haarcascade_frontalface_alt.xml'

face_cascade = cv2.CascadeClassifier(frontal_face_cascade_path)

best_network = Network(num_classes=num_classes)
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
    image = transforms_functional.resize(Image.fromarray(image), size=(224, 224))
    image = transforms_functional.to_tensor(image)
    image = transforms_functional.normalize(image, [0.5], [0.5])

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
