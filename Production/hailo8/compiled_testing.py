import cv2
import time

import numpy as np

from util import init_cv_cap, init_cascade
from inference import HailoInference
from processingUtil import preprocess_face_landmarks


def main():
    network="model/face-landmarks-detection.hef"

    # Load the HEF model and Get input shape
    hailo_inference = HailoInference(network, input_type="UINT8", output_type='FLOAT32')
    input_height, input_width, _ = hailo_inference.get_input_shape()

    cap = init_cv_cap()
    face_cascade = init_cascade()


    fps_start_time = 0
    fps = 0
    class_num = 136

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        fps_end_time = time.time()
        time_diff = fps_end_time - fps_start_time
        fps = 1 / time_diff
        fps_start_time = fps_end_time

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Run face detection without scaling
        faces = face_cascade.detectMultiScale(gray_frame, minNeighbors=4, minSize=(30, 30),
                                              flags=cv2.CASCADE_SCALE_IMAGE)

        all_landmarks = []

        for (x, y, w, h) in faces:
            # Add padding to the face region
            x_start = max(0, x)
            y_start = max(0, y)
            x_end = min(frame.shape[1], x + w)
            y_end = min(frame.shape[0], y + h)

            # Crop region of interest of the face itself
            face_roi = frame[y_start:y_end, x_start:x_end]

            # Preprocess the face region
            preprocessed_face = preprocess_face_landmarks(face_roi, target_size=(input_height, input_width))

            # Run inference
            results = hailo_inference.run(preprocessed_face)
            landmarks_batch = results['face-landmarks-detection/fc1']

            # Reshape the landmarks for the current face
            try:
                # Reshape to (num_landmarks, 2)
                landmarks = landmarks_batch[0].reshape(int(class_num / 2), 2)
            except ValueError as e:
                raise ValueError(f"Error reshaping landmarks for face: {e}")

            # Adjust landmarks based on face position and size
            landmarks = (landmarks + 0.5) * np.array([[w, h]]) + np.array([[x, y]])  # Adjust based on the face ROI

            # Append the processed landmarks to the all_landmarks list
            all_landmarks.append(landmarks)

        # Draw landmarks on the frame
        for landmarks in all_landmarks:
            for (x, y) in landmarks:
                cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)


        # Display FPS
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 150), 2, cv2.LINE_AA)

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
