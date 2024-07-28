import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import argparse
from face_landmarks_detection.utils.network import Network
from face_landmarks_detection.utils.utils import init_cv_cap

def main():
    """
    Main function for facial landmarks detection testing using a webcam feed.

    The function captures video from the webcam, detects faces using a Haar cascade classifier,
    processes each detected face through a neural network to predict facial landmarks, and displays
    the results in real-time with FPS information.
    """
    parser = argparse.ArgumentParser(description="Facial Landmarks Detection Testing")
    parser.add_argument("-n", "--net",
                        default='../../models/face/face_landmarks_epoch_4396.pth',
                        type=str,
                        help="Path for the .pth model.",
                        required=False
                        )
    parser.add_argument("-m", "--mode",
                        default="face",
                        type=str,
                        help="face / eyes",
                        required=False
                        )
    parser.add_argument("-d", "--dev",
                        default="cpu",
                        type=str,
                        help="cpu, cuda:0 or rocm-(not implemented yet)",
                        required=False
                        )
    args = parser.parse_args()

    # Paths to the model weights and Haar cascade file
    weights_path = args.net
    frontal_face_cascade_path = '../../../haarcascades/haarcascade_frontalface_alt.xml'

    # Determine the number of classes based on the mode (face or eyes)
    class_num = 136 if args.mode == "face" else 24

    # Load the neural network model with the specified number of classes
    network = Network(num_classes=class_num)
    network.load_state_dict(torch.load(weights_path, map_location=torch.device(args.dev)))
    network.eval()

    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(frontal_face_cascade_path)

    # Start video capture
    cap = init_cv_cap(640, 480, 30)

    # Define preprocessing steps for the face images
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Initialize time for FPS calculation
    prev_time = cv2.getTickCount()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate FPS
        current_time = cv2.getTickCount()
        time_diff = (current_time - prev_time) / cv2.getTickFrequency()
        prev_time = current_time
        fps = 1 / time_diff

        # Convert frame to grayscale
        grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        faces = face_cascade.detectMultiScale(grayscale_image, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30),
                                              flags=cv2.CASCADE_SCALE_IMAGE)

        # Initialize lists to store landmarks and face tensors
        all_landmarks = []
        face_tensors = []

        # Process each detected face
        for (x, y, w, h) in faces:
            face = grayscale_image[y:y + h, x:x + w]
            face = preprocess(Image.fromarray(face))
            face_tensors.append(face.unsqueeze(0))

        # If faces are detected, process them in a batch
        if face_tensors:
            face_batch = torch.cat(face_tensors)
            with torch.no_grad():
                landmarks_batch = network(face_batch)
            for i, (x, y, w, h) in enumerate(faces):
                landmarks = (landmarks_batch[i].view(int(class_num / 2), 2).detach().numpy() + 0.5) * np.array(
                    [[w, h]]) + np.array([[x, y]])
                all_landmarks.append(landmarks)

        # Draw landmarks on the frame
        for landmarks in all_landmarks:
            for (x, y) in landmarks:
                cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)

        # Display FPS on the frame
        cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame with landmarks
        cv2.imshow('Webcam Face Landmarks', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
