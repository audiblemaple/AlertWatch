import cv2
import argparse
import time

import numpy as np

from face_landmarks_detection.utils.utils import init_cv_cap
from inference import HailoInference
from processingUtil import preprocess_face_landmarks


def init_cascade():
    haar_cascade_path = '../../haarcascades/haarcascade_frontalface_alt.xml'
    face_cascade = cv2.CascadeClassifier(haar_cascade_path)
    if face_cascade.empty():
        raise IOError(f"Could not load Haar cascade file from {haar_cascade_path}")

    return face_cascade


def main():
    parser = argparse.ArgumentParser(description="Facial Landmarks Detection Example")
    # parser.add_argument("-n", "--net", help="Path for the HEF model.", default="model/face-landmarks-detection.hef", required=False)
    parser.add_argument("-n", "--net", help="Path for the HEF model.", default="model/face-landmarks-detection_float.hef", required=False)
    parser.add_argument("-a", "--arch", help="Hailo architecture, h8, h15h", required=False)  # For future
    args = parser.parse_args()

    # Load the HEF model and Get input shape
    hailo_inference = HailoInference(args.net, input_type="UINT8", output_type='FLOAT32')
    input_height, input_width, _ = hailo_inference.get_input_shape()

    cap = init_cv_cap()
    face_cascade = init_cascade()



    fps_start_time = 0
    fps = 0
    class_num = 136
    padding = 10

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        fps_end_time = time.time()
        time_diff = fps_end_time - fps_start_time
        fps = 1 / time_diff
        fps_start_time = fps_end_time

        # Run face detection without scaling
        faces = face_cascade.detectMultiScale(frame, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Add padding to the face region
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(frame.shape[1], x + w + padding)
            y_end = min(frame.shape[0], y + h + padding)

            face_roi = frame[y_start:y_end, x_start:x_end]

            # Preprocess the face region
            preprocessed_face = preprocess_face_landmarks(face_roi, target_size=(input_height, input_width))

            all_landmarks = []

            # Run inference
            results = hailo_inference.run(preprocessed_face)
            landmarks_batch = results['face-landmarks-detection/fc1']


            # Process each face's landmarks
            for i, (x, y, w, h) in enumerate(faces):
                # Reshape the landmarks for the current face
                try:
                    # Reshape to (num_landmarks, 2)
                    landmarks = landmarks_batch[0].reshape(int(class_num / 2), 2)
                except ValueError as e:
                    raise ValueError(f"Error reshaping landmarks for face {i}: {e}")

                # Adjust landmarks based on face position and size
                # Adding 0.5 as in the example for potential alignment purposes
                landmarks = (landmarks + 0.5) * np.array([[w, h]]) + np.array([[x, y]])

                # Append the processed landmarks to the all_landmarks list
                all_landmarks.append(landmarks)


            # Draw landmarks on the frame
            for landmarks in all_landmarks:
                for (x, y) in landmarks:
                    cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)



        # Display FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 150), 2, cv2.LINE_AA)

        # Display face detection status
        face_detected_text = "Face Detected" if len(faces) > 0 else "No Face Detected"
        color = (0, 255, 0) if len(faces) > 0 else (0, 0, 255)
        cv2.putText(frame, face_detected_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        # Show the frame with landmarks
        cv2.imshow('Webcam Face Landmarks', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    hailo_inference.release_device()

if __name__ == "__main__":
    main()
