import cv2
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import time
from face_landmarks_detection.utils.network import Network


# Paths to the model weights and Haar cascade file
weights_path = '../../weights_2000_epochs/eyes_landmarks_epoch_216.pth'
frontal_face_cascade_path = '../../../haarcascades/haarcascade_frontalface_alt.xml'

# Load the model
best_network = Network(num_classes=24)
best_network.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
best_network.eval()

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(frontal_face_cascade_path)

# Start video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)

prev_time = time.time()
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = grayscale_image.shape

    faces = face_cascade.detectMultiScale(grayscale_image, 1.1, 4)

    face_detected = len(faces) > 0

    all_landmarks = []
    for (x, y, w, h) in faces:
        face = grayscale_image[y:y + h, x:x + w]
        face = TF.resize(Image.fromarray(face), size=(224, 224))
        face = TF.to_tensor(face)
        face = TF.normalize(face, [0.5], [0.5])

        with torch.no_grad():
            landmarks = best_network(face.unsqueeze(0))

        landmarks = (landmarks.view(12, 2).detach().numpy() + 0.5) * np.array([[w, h]]) + np.array([[x, y]])
        all_landmarks.append(landmarks)

    for landmarks in all_landmarks:
        for (x, y) in landmarks:
            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)

    # Display FPS on the frame
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display face detection status
    if face_detected:
        cv2.putText(frame, 'Face Detected', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'No Face Detected', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Webcam Eye Landmarks', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



# import cv2
# import torch
# import torch.nn as nn
# from torchvision import models
# import torchvision.transforms.functional as TF
# from PIL import Image
# import numpy as np
#
# class Network(nn.Module):
#     def __init__(self, num_classes=24):  # Adjusted for eye-only model
#         super(Network, self).__init__()
#         self.model = models.resnet18(pretrained=False)
#         self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
#
#     def forward(self, x):
#         x = self.model(x)
#         return x
#
# # Paths to the model weights and Haar cascade file
# # weights_path = 'weights_2000_epochs/eyes_landmarks_epoch_76.pth'
# weights_path = 'weights_2000_epochs/eyes_landmarks_epoch_149.pth'
# frontal_face_cascade_path = 'haarcascades/haarcascade_frontalface_alt.xml'
#
# # Load the model
# best_network = Network()
# best_network.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
# best_network.eval()
#
# # Load Haar Cascade for face detection
# face_cascade = cv2.CascadeClassifier(frontal_face_cascade_path)
#
# # Start video capture
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
# # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# cap.set(cv2.CAP_PROP_FPS, 90)
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     height, width = grayscale_image.shape
#
#     faces = face_cascade.detectMultiScale(grayscale_image, 1.1, 4)
#
#     all_landmarks = []
#     for (x, y, w, h) in faces:
#         face = grayscale_image[y:y + h, x:x + w]
#         face = TF.resize(Image.fromarray(face), size=(224, 224))
#         face = TF.to_tensor(face)
#         face = TF.normalize(face, [0.5], [0.5])
#
#         with torch.no_grad():
#             landmarks = best_network(face.unsqueeze(0))
#
#         landmarks = (landmarks.view(12, 2).detach().numpy() + 0.5) * np.array([[w, h]]) + np.array([[x, y]])
#         all_landmarks.append(landmarks)
#
#     for landmarks in all_landmarks:
#         for (x, y) in landmarks:
#             cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
#
#     cv2.imshow('Webcam Eye Landmarks', frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
