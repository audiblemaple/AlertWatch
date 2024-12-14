"""
Face Landmarks Detection with Hailo Inference and OpenCV.

This script captures video from a webcam, detects faces using a Haar cascade,
and predicts facial landmarks using a pretrained model on the Hailo hardware.

Modules:
- OpenCV: For video capturing and face detection.
- NumPy: For numerical operations on the landmarks.
- util: Custom utility functions to initialize webcam and cascade.
- inference: Wrapper for Hailo inference API.
- prePostProcessing: For preprocessing input frames.
"""

import cv2
import time
import numpy as np
from util import init_cv_cap, init_cascade
from inference import HailoInference
from processingUtil import preprocess_face_landmarks
from headPoseUtil import estimate_head_pose_with_direction
from eyeUtil import handle_eye_aspect_ratio, calculate_blink_rate, calibrate_ear, calculate_ear
from drawUtil import draw_text, draw_landmarks, draw_head_pose, display_fps, draw_bounding_box

def initialize_hailo_inference(model_path, input_type="UINT8", output_type="FLOAT32"):
    """
    Initializes the Hailo inference engine.

    Args:
        model_path (str): Path to the Hailo HEF model.
        input_type (str): Data type for model input (default: "UINT8").
        output_type (str): Data type for model output (default: "FLOAT32").

    Returns:
        HailoInference: Initialized inference object.
        tuple: Input shape (height, width, channels).
    """
    hailo_inference = HailoInference(model_path, input_type=input_type, output_type=output_type)
    input_shape = hailo_inference.get_input_shape()
    return hailo_inference, input_shape


def detect_faces(face_cascade, frame):
    """
    Detects faces in a frame using the Haar cascade.

    Args:
        face_cascade (cv2.CascadeClassifier): Initialized Haar cascade.
        frame (np.ndarray): Grayscale video frame.

    Returns:
        list: List of bounding boxes for detected faces.
    """
    return face_cascade.detectMultiScale(
        frame, minNeighbors=4, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
    )


def process_face(frame, bbox, input_shape):
    """
    Prepares the face region for inference.

    Args:
        frame (np.ndarray): Video frame.
        bbox (tuple): Bounding box of the face (x, y, w, h).
        input_shape (tuple): Shape of the model input (height, width, channels).

    Returns:
        np.ndarray: Preprocessed face region.
        tuple: Adjusted bounding box for cropping.
    """
    x, y, w, h = bbox
    x_start = max(0, x)
    y_start = max(0, y)
    x_end = min(frame.shape[1], x + w)
    y_end = min(frame.shape[0], y + h)
    face_roi = frame[y_start:y_end, x_start:x_end]

    preprocessed_face = preprocess_face_landmarks(face_roi, target_size=input_shape[:2])
    return preprocessed_face, (x, y, w, h)


def adjust_landmarks(landmarks, bbox):
    """
    Adjusts landmark positions based on the face bounding box.

    Args:
        landmarks (np.ndarray): Predicted landmarks.
        bbox (tuple): Bounding box of the face (x, y, w, h).

    Returns:
        np.ndarray: Adjusted landmarks.
    """
    x, y, w, h = bbox
    return (landmarks + 0.5) * np.array([[w, h]]) + np.array([[x, y]])


def draw_results(frame, all_landmarks, fps, face_detected):
    """
    Draws landmarks, FPS, and detection status on the frame.

    Args:
        frame (np.ndarray): Video frame.
        all_landmarks (list): List of landmarks for all detected faces.
        fps (float): Current frames per second.
        face_detected (bool): Whether a face was detected.
    """
    # Draw landmarks
    for landmarks in all_landmarks:
        for (x, y) in landmarks:
            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)

    # Display FPS
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 150), 2, cv2.LINE_AA)

    # Display face detection status
    text = "Face Detected" if face_detected else "No Face Detected"
    color = (0, 255, 0) if face_detected else (0, 0, 255)
    cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)


def estimate_head_pose_for_all_faces(all_landmarks, frame_shape):
    """
    Estimate head pose for all detected faces.
    """
    head_poses = []
    directions = []
    for landmarks in all_landmarks:
        rotation_vector, translation_vector, direction = estimate_head_pose_with_direction(landmarks, frame_shape)
        head_poses.append((rotation_vector, translation_vector))
        directions.append(direction)
    return head_poses, directions

def process_ear_for_all_faces(all_landmarks, frame, frame_count, fps, blink_counter, current_blink_start, blink_durations):
    """
    Process EAR and blink detection for all detected faces.
    """
    for landmarks in all_landmarks:
        frame, blink_counter, current_blink_start, blink_durations = handle_eye_aspect_ratio(
            frame, landmarks, frame_count, fps, blink_counter, current_blink_start, blink_durations
        )
    return frame, blink_counter, current_blink_start, blink_durations


def main():
    """
    Main function for real-time face landmark detection.
    """
    # Initialize Hailo inference
    model_path = "model/face-landmarks-detection.hef"
    hailo_inference, input_shape = initialize_hailo_inference(model_path)

    # Initialize webcam and face cascade
    cap = init_cv_cap()
    face_cascade = init_cascade()

    fps_start_time = time.time()
    class_num = 136

    frame_count = 0
    # Variables for EAR calibration and blink tracking
    blink_counter = 0
    blink_durations = []
    current_blink_start = None
    ear_values = []
    calibrated_ear_threshold = 0.25
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        frame_count += 1

        # Calculate FPS
        fps_end_time = time.time()
        fps = 1 / (fps_end_time - fps_start_time)
        fps_start_time = fps_end_time

        # Convert to grayscale and detect faces
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(face_cascade, gray_frame)

        all_landmarks = []
        for bbox in faces:
            # Process face and run inference
            preprocessed_face, adjusted_bbox = process_face(frame, bbox, input_shape)


            # Display the preprocessed face
            if preprocessed_face is not None:
                # Reshape preprocessed face for display if needed
                preprocessed_face_display = preprocessed_face.squeeze()  # Remove batch and channel dimensions if present
                if len(preprocessed_face_display.shape) == 2:
                    # Convert grayscale to BGR for consistent display with OpenCV
                    preprocessed_face_display = cv2.cvtColor(preprocessed_face_display, cv2.COLOR_GRAY2BGR)
                cv2.imshow('Preprocessed Face', preprocessed_face_display)
            else:
                print("preprocessed_face is None")
                continue

            results = hailo_inference.run(preprocessed_face)
            landmarks_batch = results['face-landmarks-detection/fc1']

            try:
                # Reshape and adjust landmarks
                landmarks = landmarks_batch[0].reshape(class_num // 2, 2)
                # adjusted_landmarks = adjust_landmarks(landmarks, adjusted_bbox)
                adjusted_landmarks = adjust_landmarks(landmarks, bbox)
                all_landmarks.append(adjusted_landmarks)
            except ValueError as e:
                print(f"Error processing landmarks: {e}")
                continue

        # Draw results on the frame
        draw_results(frame, all_landmarks, fps, len(faces) > 0)

        head_poses, directions = estimate_head_pose_for_all_faces(all_landmarks, frame.shape)

        # EAR and blink detection
        frame, blink_counter, current_blink_start, blink_durations = process_ear_for_all_faces(
            all_landmarks, frame, frame_count, fps, blink_counter, current_blink_start, blink_durations
        )

        # Draw landmarks and head pose on the frame
        for landmarks, (rotation_vector, translation_vector), direction in zip(all_landmarks, head_poses, directions):
            # draw_landmarks(frame, [landmarks])
            # draw_head_pose(frame, rotation_vector, translation_vector, get_camera_intrinsics(frame.shape))
            cv2.putText(frame, f'Direction: {direction}', (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


        # Display the frame
        cv2.imshow('Webcam Face Landmarks', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hailo_inference.release_device()


if __name__ == "__main__":
    main()

# --------------------------------------------- working -------------------------------------------------------------
# """
# Face Landmarks Detection with Hailo Inference and OpenCV.
#
# This script captures video from a webcam, detects faces using a Haar cascade,
# and predicts facial landmarks using a pretrained model on the Hailo hardware.
#
# Modules:
# - OpenCV: For video capturing and face detection.
# - NumPy: For numerical operations on the landmarks.
# - util: Custom utility functions to initialize webcam and cascade.
# - inference: Wrapper for Hailo inference API.
# - prePostProcessing: For preprocessing input frames.
# """
#
# import cv2
# import time
# import numpy as np
# from util import init_cv_cap, init_cascade
# from inference import HailoInference
# from prePostProcessing import preprocess_face_landmarks
#
#
# def initialize_hailo_inference(model_path, input_type="UINT8", output_type="FLOAT32"):
#     """
#     Initializes the Hailo inference engine.
#
#     Args:
#         model_path (str): Path to the Hailo HEF model.
#         input_type (str): Data type for model input (default: "UINT8").
#         output_type (str): Data type for model output (default: "FLOAT32").
#
#     Returns:
#         HailoInference: Initialized inference object.
#         tuple: Input shape (height, width, channels).
#     """
#     hailo_inference = HailoInference(model_path, input_type=input_type, output_type=output_type)
#     input_shape = hailo_inference.get_input_shape()
#     return hailo_inference, input_shape
#
#
# def detect_faces(face_cascade, frame):
#     """
#     Detects faces in a frame using the Haar cascade.
#
#     Args:
#         face_cascade (cv2.CascadeClassifier): Initialized Haar cascade.
#         frame (np.ndarray): Grayscale video frame.
#
#     Returns:
#         list: List of bounding boxes for detected faces.
#     """
#     return face_cascade.detectMultiScale(
#         frame, minNeighbors=4, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
#     )
#
#
# def process_face(frame, bbox, input_shape):
#     """
#     Prepares the face region for inference.
#
#     Args:
#         frame (np.ndarray): Video frame.
#         bbox (tuple): Bounding box of the face (x, y, w, h).
#         input_shape (tuple): Shape of the model input (height, width, channels).
#
#     Returns:
#         np.ndarray: Preprocessed face region.
#         tuple: Adjusted bounding box for cropping.
#     """
#     x, y, w, h = bbox
#     x_start = max(0, x)
#     y_start = max(0, y)
#     x_end = min(frame.shape[1], x + w)
#     y_end = min(frame.shape[0], y + h)
#     face_roi = frame[y_start:y_end, x_start:x_end]
#
#     preprocessed_face = preprocess_face_landmarks(face_roi, target_size=input_shape[:2])
#     return preprocessed_face, (x, y, w, h)
#
#
# def adjust_landmarks(landmarks, bbox):
#     """
#     Adjusts landmark positions based on the face bounding box.
#
#     Args:
#         landmarks (np.ndarray): Predicted landmarks.
#         bbox (tuple): Bounding box of the face (x, y, w, h).
#
#     Returns:
#         np.ndarray: Adjusted landmarks.
#     """
#     x, y, w, h = bbox
#     return (landmarks + 0.5) * np.array([[w, h]]) + np.array([[x, y]])
#
#
# def draw_results(frame, all_landmarks, fps, face_detected):
#     """
#     Draws landmarks, FPS, and detection status on the frame.
#
#     Args:
#         frame (np.ndarray): Video frame.
#         all_landmarks (list): List of landmarks for all detected faces.
#         fps (float): Current frames per second.
#         face_detected (bool): Whether a face was detected.
#     """
#     # Draw landmarks
#     for landmarks in all_landmarks:
#         for (x, y) in landmarks:
#             cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
#
#     # Display FPS
#     cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 150), 2, cv2.LINE_AA)
#
#     # Display face detection status
#     text = "Face Detected" if face_detected else "No Face Detected"
#     color = (0, 255, 0) if face_detected else (0, 0, 255)
#     cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
#
#
# def main():
#     """
#     Main function for real-time face landmark detection.
#     """
#     # Initialize Hailo inference
#     model_path = "model/face-landmarks-detection.hef"
#     hailo_inference, input_shape = initialize_hailo_inference(model_path)
#
#     # Initialize webcam and face cascade
#     cap = init_cv_cap()
#     face_cascade = init_cascade()
#
#     fps_start_time = time.time()
#     class_num = 136
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Failed to capture image.")
#             break
#
#         # Calculate FPS
#         fps_end_time = time.time()
#         fps = 1 / (fps_end_time - fps_start_time)
#         fps_start_time = fps_end_time
#
#         # Convert to grayscale and detect faces
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = detect_faces(face_cascade, gray_frame)
#
#         all_landmarks = []
#         for bbox in faces:
#             # Process face and run inference
#             preprocessed_face, adjusted_bbox = process_face(frame, bbox, input_shape)
#             results = hailo_inference.run(preprocessed_face)
#             landmarks_batch = results['face-landmarks-detection/fc1']
#
#             try:
#                 # Reshape and adjust landmarks
#                 landmarks = landmarks_batch[0].reshape(class_num // 2, 2)
#                 adjusted_landmarks = adjust_landmarks(landmarks, adjusted_bbox)
#                 all_landmarks.append(adjusted_landmarks)
#             except ValueError as e:
#                 print(f"Error processing landmarks: {e}")
#                 continue
#
#         # Draw results on the frame
#         draw_results(frame, all_landmarks, fps, len(faces) > 0)
#
#         # Display the frame
#         cv2.imshow('Webcam Face Landmarks', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     # Cleanup
#     cap.release()
#     cv2.destroyAllWindows()
#     hailo_inference.release_device()
#
#
# if __name__ == "__main__":
#     main()



## -------------------------------------------- LEGACY -------------------------------------------------------
# """
# Face Landmarks Detection with Hailo Inference and OpenCV.
#
# This script captures video from a webcam, detects faces using a Haar cascade,
# and predicts facial landmarks using a pretrained model on the Hailo hardware.
#
# Modules:
# - OpenCV: For video capturing and face detection.
# - NumPy: For numerical operations.
# - util: Custom utility functions to initialize webcam and cascade.
# - inference: Wrapper for Hailo inference API.
# - prePostProcessing: For preprocessing input frames.
# """
# import cv2
# import time
#
# import numpy as np
#
# from util import init_cv_cap, init_cascade
# from inference import HailoInference
# from prePostProcessing import preprocess_face_landmarks
#
#
# def main():
#     """
#     Main function for detecting faces and predicting facial landmarks.
#
#     This function initializes the webcam, loads a pre-trained HEF model,
#     performs face detection using OpenCV's Haar cascade, and predicts
#     facial landmarks using the Hailo hardware. The results are displayed
#     in real-time with FPS and detection status on the video feed.
#
#     Steps:
#     1. Initialize webcam and Haar cascade for face detection.
#     2. Load the Hailo HEF model for inference.
#     3. Perform real-time video capture and face detection.
#     4. Preprocess detected face regions for landmark prediction.
#     5. Run the landmarks model and overlay results on the video frame.
#     6. Display the video feed with FPS and face detection status.
#
#     Press 'q' to quit the application.
#
#     Raises:
#         ValueError: If there is an issue reshaping the landmarks array.
#     """
#     network="model/face-landmarks-detection.hef"
#
#     # Load the HEF model and Get input shape
#     hailo_inference = HailoInference(network, input_type="UINT8", output_type='FLOAT32')
#     input_height, input_width, _ = hailo_inference.get_input_shape()
#
#     cap = init_cv_cap()
#     face_cascade = init_cascade()
#
#
#     fps_start_time = 0
#     fps = 0
#     class_num = 136
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Failed to capture image.")
#             break
#
#         fps_end_time = time.time()
#         time_diff = fps_end_time - fps_start_time
#         fps = 1 / time_diff
#         fps_start_time = fps_end_time
#
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#         # Run face detection without scaling
#         faces = face_cascade.detectMultiScale(gray_frame, minNeighbors=4, minSize=(30, 30),
#                                               flags=cv2.CASCADE_SCALE_IMAGE)
#
#         all_landmarks = []
#
#         for (x, y, w, h) in faces:
#             # Add padding to the face region
#             x_start = max(0, x)
#             y_start = max(0, y)
#             x_end = min(frame.shape[1], x + w)
#             y_end = min(frame.shape[0], y + h)
#
#             # Crop region of interest of the face itself
#             face_roi = frame[y_start:y_end, x_start:x_end]
#
#             # Preprocess the face region
#             preprocessed_face = preprocess_face_landmarks(face_roi, target_size=(input_height, input_width))
#
#             # Run inference
#             results = hailo_inference.run(preprocessed_face)
#             landmarks_batch = results['face-landmarks-detection/fc1']
#
#             # Reshape the landmarks for the current face
#             try:
#                 # Reshape to (num_landmarks, 2)
#                 landmarks = landmarks_batch[0].reshape(int(class_num / 2), 2)
#             except ValueError as e:
#                 raise ValueError(f"Error reshaping landmarks for face: {e}")
#
#             # Adjust landmarks based on face position and size
#             landmarks = (landmarks + 0.5) * np.array([[w, h]]) + np.array([[x, y]])  # Adjust based on the face ROI
#
#             # Append the processed landmarks to the all_landmarks list
#             all_landmarks.append(landmarks)
#
#         # Draw landmarks on the frame
#         for landmarks in all_landmarks:
#             for (x, y) in landmarks:
#                 cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
#
#
#         # Display FPS
#         cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 150), 2, cv2.LINE_AA)
#
#         # Display face detection status
#         face_detected_text = "Face Detected" if len(faces) > 0 else "No Face Detected"
#         color = (0, 255, 0) if len(faces) > 0 else (0, 0, 255)
#         cv2.putText(frame, face_detected_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
#
#         # Show the frame with landmarks
#         cv2.imshow('Webcam Face Landmarks', frame)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     # Release resources
#     cap.release()
#     cv2.destroyAllWindows()
#     hailo_inference.release_device()
#
# if __name__ == "__main__":
#     main()
