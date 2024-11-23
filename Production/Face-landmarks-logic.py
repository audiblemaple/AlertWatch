import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import argparse
from face_landmarks_detection.utils.network import Network
from face_landmarks_detection.utils.utils import init_cv_cap

def parse_arguments():
    """
    Parses command-line arguments for the script.
    """
    parser = argparse.ArgumentParser(description="Facial Landmarks Detection Testing")
    parser.add_argument("-n", "--net",
                        default='../../models/face/10K_epochs/face_landmarks_epoch_9319.pth',
                        type=str,
                        help="Path for the .pth model.",
                        required=False)
    parser.add_argument("-m", "--mode",
                        default="face",
                        type=str,
                        help="face / eyes",
                        required=False)
    parser.add_argument("-d", "--dev",
                        default="cpu",
                        type=str,
                        help="cpu, cuda:0 or rocm-(not implemented yet)",
                        required=False)
    return parser.parse_args()

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

def get_preprocessing_pipeline():
    """
    Defines the preprocessing pipeline for face images.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

def process_frame(frame, face_cascade, preprocess_pipeline, network, class_num):
    """
    Processes a single video frame: detects faces, extracts landmarks, and returns landmarks and the processed frame.
    """
    grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayscale_image, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
    all_landmarks = []
    face_tensors = []

    for (x, y, w, h) in faces:
        face = grayscale_image[y:y + h, x:x + w]
        face = preprocess_pipeline(Image.fromarray(face))
        face_tensors.append(face.unsqueeze(0))

    if face_tensors:
        face_batch = torch.cat(face_tensors)
        with torch.no_grad():
            landmarks_batch = network(face_batch)
        for i, (x, y, w, h) in enumerate(faces):
            landmarks = (landmarks_batch[i].view(int(class_num / 2), 2).detach().numpy() + 0.5) * np.array(
                [[w, h]]) + np.array([[x, y]])
            all_landmarks.append(landmarks)

    return all_landmarks, frame

def draw_landmarks(frame, all_landmarks):
    """
    Draws detected landmarks on the frame.
    """
    for landmarks in all_landmarks:
        for (x, y) in landmarks:
            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)

def display_fps(frame, fps):
    """
    Displays the frames per second (FPS) on the frame.
    """
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def main():
    args = parse_arguments()

    weights_path = args.net
    frontal_face_cascade_path = '../../../haarcascades/haarcascade_frontalface_alt.xml'
    class_num = 136 if args.mode == "face" else 24

    network = load_model(weights_path, class_num, args.dev)
    face_cascade = load_face_cascade(frontal_face_cascade_path)
    preprocess_pipeline = get_preprocessing_pipeline()
    cap = init_cv_cap(640, 480, 30)

    prev_time = cv2.getTickCount()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = cv2.getTickCount()
        time_diff = (current_time - prev_time) / cv2.getTickFrequency()
        prev_time = current_time
        fps = 1 / time_diff

        all_landmarks, frame = process_frame(frame, face_cascade, preprocess_pipeline, network, class_num)
        draw_landmarks(frame, all_landmarks)
        display_fps(frame, fps)

        cv2.imshow('Webcam Face Landmarks', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
