#!/usr/bin/env python3
import time

import cv2
import asyncio
import numpy as np
# from socketUtil import WebSocketClient
from drawUtil import draw_bounding_box
from inference import HailoInferenceAsyncMultiModel
from hailo_platform import VDevice, HailoSchedulingAlgorithm, FormatType
from processingUtil import preprocess_faces, postprocess_faces, preprocess_face_landmarks

from util import init_cv_cap


# WS_URL = "ws://192.168.0.183:5000"
WS_URL = "ws://192.168.0.233:5000"
RECONNECT_INTERVAL = 3

def preprocess_face(frame, bbox, input_shape, gray=True):
    x1, y1, x2, y2 = bbox

    # Ensure the coordinates are integers and within image bounds
    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(frame.shape[1], int(round(x2)))
    y2 = min(frame.shape[0], int(round(y2)))

    # Validate bounding box dimensions
    if x2 <= x1 or y2 <= y1:
        print(f"Invalid bounding box with non-positive width or height: {bbox}")
        return None, bbox, None

    face_roi = frame[y1:y2, x1:x2]

    if face_roi.size == 0:
        print(f"Invalid face ROI: {bbox}")
        return None, bbox, None

    if gray:
        # Convert to grayscale
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

    # Original face ROI dimensions
    face_h, face_w = face_roi.shape[:2]
    target_h, target_w = input_shape[:2]

    # Validate face ROI dimensions
    if face_w <= 0 or face_h <= 0:
        print(f"Invalid face ROI dimensions: width={face_w}, height={face_h}")
        return None, bbox, None

    # Calculate scaling factor to ensure the resized image is at least as big as the target size
    scale_w = target_w / face_w
    scale_h = target_h / face_h
    scale = max(scale_w, scale_h)

    # New dimensions after scaling
    new_w = int(face_w * scale)
    new_h = int(face_h * scale)

    # Validate new dimensions
    if new_w <= 0 or new_h <= 0:
        print(f"Invalid new dimensions after scaling: new_w={new_w}, new_h={new_h}")
        return None, bbox, None

    # Resize the face ROI
    resized_face = cv2.resize(face_roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Calculate offsets to crop the center part
    x_offset = (new_w - target_w) // 2
    y_offset = (new_h - target_h) // 2

    # Handle cases where the resized image is smaller than the target size
    preprocessed_face = np.zeros((target_h, target_w), dtype=resized_face.dtype)

    x_start = max(0, -x_offset)
    y_start = max(0, -y_offset)
    x_end = x_start + min(new_w, target_w)
    y_end = y_start + min(new_h, target_h)

    resized_x_start = max(0, x_offset)
    resized_y_start = max(0, y_offset)
    resized_x_end = resized_x_start + (x_end - x_start)
    resized_y_end = resized_y_start + (y_end - y_start)

    preprocessed_face[y_start:y_end, x_start:x_end] = resized_face[resized_y_start:resized_y_end, resized_x_start:resized_x_end]

    # Expand dimensions to fit model input
    if gray:
        preprocessed_face = np.expand_dims(preprocessed_face, axis=[0, -1])
    else:
        preprocessed_face = np.expand_dims(preprocessed_face, axis=0)


    # Prepare transformation parameters for mapping landmarks back
    transformation_params = {
        'scale': scale,
        'face_offset_x': x1,   # Offset from top-left corner of frame
        'face_offset_y': y1,
        'crop_offset_x': x_offset,  # Offset due to cropping after scaling
        'crop_offset_y': y_offset,
        'original_face_size': (face_w, face_h),
        'resized_face_size': (new_w, new_h),
        'input_size': (target_w, target_h)
    }

    return preprocessed_face, bbox

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

    # Display face detection status
    text = "Face Detected" if face_detected else "No Face Detected"
    color = (0, 255, 0) if face_detected else (0, 0, 255)
    cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

async def main():
    # model paths
    face_det = "../models/hailo8/scrfd_10g.hef"
    face_landmark = "../models/hailo8/face-landmarks-detection.hef"

    # Create WebSocket client
    # ws_client = WebSocketClient(WS_URL, RECONNECT_INTERVAL)

    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

    # Run bash commands and capture their outputs (for initial connection only)
    # command_outputs = await ws_client.run_bash_commands()

    # Try to connect to the WebSocket server initially
    # await ws_client.connect()

    # Capture video from webcam
    cap = init_cv_cap(640, 360 )
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    class_num = 136
    fps_start_time = 0
    fps = 0
    timeout_ms = 10000

    # Initialize the object with two models
    model_paths = (face_det, face_landmark)
    hailo_inference = HailoInferenceAsyncMultiModel(
        model_paths, input_types=('FLOAT32', 'UINT8'), output_types=('FLOAT32', 'FLOAT32')
    )

    # Get input shapes for the models
    face_detection_input_shape = hailo_inference.get_input_shape(model_id=1)
    face_landmarks_input_shape = hailo_inference.get_input_shape(model_id=2)
    print(f"face_detection_input_shape: {face_detection_input_shape}")
    print(f"face_landmarks_input_shape: {face_landmarks_input_shape}")

    # TODO: Find the out layer using the shape of the layers
    face_det_input_name = hailo_inference.models['model_1']['input_vstream_info']
    face_det_output_name = hailo_inference.models['model_1']['output_vstream_info']
    # TODO

    # TODO: Nothing, this is good
    face_land_output_name = hailo_inference.models['model_2']['output_vstream_info'][0].name

    face_size = 150

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        fps_end_time = time.time()
        time_diff = fps_end_time - fps_start_time
        fps = 1 / time_diff
        fps_start_time = fps_end_time

        original_frame = frame.copy()

        preprocessed_frame, scale, pad_w, pad_h, img_w, img_h = preprocess_faces(
            frame, input_size=(face_detection_input_shape[0], face_detection_input_shape[1])
        )
        preprocessed_frame = preprocessed_frame.astype(np.float32)
        preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)

        ''' Running inference for face detection '''
        raw_faces = hailo_inference.run(model_id=1, input_data=preprocessed_frame)

        faces = postprocess_faces(raw_faces, img_w, img_h, scale, pad_w, pad_h)

        all_landmarks = []

        if len(faces) == 0:
            print("No faces detected.")
            continue

        # Draw bounding boxes
        for (x1, y1, x2, y2, score) in faces:
            # Clip bounding box coordinates
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)

            adjusted_x2 = x2 - face_size
            adjusted_y2 = y2 - face_size

            draw_bounding_box(original_frame, score, (adjusted_x2, adjusted_y2), (x2, y2))

            preprocessed_face, adjusted_bbox  = preprocess_face(frame, (adjusted_x2, adjusted_y2, x2, y2), face_landmarks_input_shape)

            if preprocessed_face is None:
                print("preprocessed_face is None")
                continue

            # Display the preprocessed face
            # Reshape preprocessed face for display if needed
            preprocessed_face_display = preprocessed_face.squeeze()
            if len(preprocessed_face_display.shape) == 2:
                # Convert grayscale to BGR for consistent display with OpenCV
                preprocessed_face_display = cv2.cvtColor(preprocessed_face_display, cv2.COLOR_GRAY2BGR)
            cv2.imshow('Preprocessed Face', preprocessed_face_display)

            ''' Running inference for Face Landmarks Detection '''
            landmarks = hailo_inference.run(model_id=2, input_data=preprocessed_face)
            landmarks_batch = landmarks.get(face_land_output_name, None)
            if landmarks_batch is None:
                print("Error: Landmarks output missing!")
                continue
            # print("Model 2 Output:", landmarks_batch)

            try:
                # Reshape and adjust landmarks
                landmarks = landmarks_batch[0].reshape(class_num // 2, 2)
                adjusted_landmarks = adjust_landmarks(landmarks, (adjusted_x2, adjusted_y2, face_size, face_size))
                all_landmarks.append(adjusted_landmarks)
            except ValueError as e:
                print(f"Error processing landmarks: {e}")
                continue


            # Ensure all data types are compatible with JSON serialization
            tensors = [{'x1': int(adjusted_x2), 'y1': int(adjusted_y2), 'x2': int(x2 + face_size), 'y2': int(y2 + face_size), 'score': float(score)} for
                       (x1, y1, x2, y2, score) in faces]


        # Draw results on the frame
        draw_results(original_frame, all_landmarks, fps, len(faces) > 0)

        # Display FPS
        cv2.putText(original_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 250, 100), 2, cv2.LINE_AA)

        # Attempt reconnection if WebSocket is not connected
        # if ws_client.websocket is None:
        #     ws_client.first_connection = True
        #     print("Attempting to reconnect to WebSocket...")
        #     await ws_client.connect()
        #     if ws_client.websocket is None:
        #         print("Reconnection failed. Retrying in 5 seconds.")
        #         await asyncio.sleep(RECONNECT_INTERVAL)
        #         continue

        # Send frame, tensors, and command outputs (only on first connect)
        # if ws_client.first_connection:
        #     await ws_client.send_data(original_frame, tensors, command_outputs)
        #     ws_client.first_connection = False  # Reset flag after first send
        # else:
        #     await ws_client.send_data(original_frame, tensors)

        # Show the frame with landmarks
        cv2.imshow('Webcam Face Landmarks', original_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    asyncio.run(main())
