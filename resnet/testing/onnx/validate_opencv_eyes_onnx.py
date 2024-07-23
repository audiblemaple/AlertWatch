import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import torchvision.transforms.functional as TF

# Load the ONNX model
onnx_path = 'eye_landmarks.onnx'
ort_session = ort.InferenceSession(onnx_path)

# Load Haar Cascade for face detection
frontal_face_cascade_path = '../../../haarcascades/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(frontal_face_cascade_path)

# Start video capture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

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
        face = TF.resize(Image.fromarray(face), size=(224, 224))
        face = TF.to_tensor(face)
        face = TF.normalize(face, [0.5], [0.5])
        face = face.unsqueeze(0).numpy()

        # Run inference
        ort_inputs = {'input': face}
        ort_outs = ort_session.run(None, ort_inputs)
        landmarks = ort_outs[0]

        landmarks = (landmarks.reshape(12, 2) + 0.5) * np.array([[w, h]]) + np.array([[x, y]])
        all_landmarks.append(landmarks)

    for landmarks in all_landmarks:
        for (lx, ly) in landmarks:
            cv2.circle(frame, (int(lx), int(ly)), 2, (0, 255, 0), -1)

    cv2.imshow('Webcam Eye Landmarks', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
