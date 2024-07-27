import cv2
import torch
import torchvision.transforms.functional as transforms_functional
from PIL import Image
import numpy as np

from face_landmarks_detection.utils.network import Network

# Paths to the model weights and Haar cascade file
weights_path = '../../models/face/face_landmarks_epoch_4396.pth'
frontal_face_cascade_path = '../../../haarcascades/haarcascade_frontalface_alt.xml'

# Load the model
network = Network(num_classes=136)
network.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
network.eval()

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(frontal_face_cascade_path)

# Start video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = grayscale_image.shape

    faces = face_cascade.detectMultiScale(grayscale_image, 1.1, 4)

    all_landmarks = []
    for (x, y, w, h) in faces:
        face = grayscale_image[y:y + h, x:x + w]
        face = transforms_functional.resize(Image.fromarray(face), size=(224, 224))
        face = transforms_functional.to_tensor(face)
        face = transforms_functional.normalize(face, [0.5], [0.5])

        with torch.no_grad():
            landmarks = network(face.unsqueeze(0))

        landmarks = (landmarks.view(68, 2).detach().numpy() + 0.5) * np.array([[w, h]]) + np.array([[x, y]])
        all_landmarks.append(landmarks)

    for landmarks in all_landmarks:
        for (x, y) in landmarks:
            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)

    cv2.imshow('Webcam Face Landmarks', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
