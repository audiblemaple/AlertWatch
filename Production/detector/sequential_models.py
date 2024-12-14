#!/usr/bin/env python3

import time
import datetime
import cv2
import asyncio
import numpy as np
from collections        import deque
from appState           import AppState
from socketUtil         import initialize_websocket
from inference          import initialize_inference_models
from util               import init_cv_cap, calculate_EAR, estimate_fps
from drawUtil           import draw_bounding_box, display_fps, draw_landmarks, display_blink_info
from prePostProcessing  import preprocess_face_detection, postprocess_faces, preprocess_face_landmarks, adjust_landmarks

# Constants
WS_URL = "ws://192.168.0.239:5000"
RECONNECT_INTERVAL = 2
FACE_SIZE = 140
CLASS_NUM = 136

# Blink Detection Constants
EAR_THRESHOLD = 0.21  # Threshold to detect blink
CONSEC_FRAMES = 2  # Number of consecutive frames the EAR should be below threshold
BLINK_COUNT_THRESHOLD = 10  # Number of blinks to trigger video saving

# Buffer Configuration
BUFFER_DURATION = 10  # seconds

async def handle_faces(faces, frame, hailo_inference, face_land_output_name, face_landmarks_input_shape, all_landmarks,
                       state: AppState, tensors):
    """
    Processes detected faces: draws bounding boxes, preprocesses faces, runs landmark detection, and handles blink counting.

    Args:
        faces (list): List of detected faces.
        frame (np.ndarray): The original video frame.
        hailo_inference (HailoInferenceAsyncMultiModel): The inference model.
        face_land_output_name (str): The output name for landmarks.
        face_landmarks_input_shape (tuple): Input shape for landmarks model.
        state (AppState): The application state.
        tensors (list): List to store tensor data for WebSocket.
    """
    for (x1, y1, x2, y2, score) in faces:
        # Clip bounding box coordinates and convert to int
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(frame.shape[1], int(x2))
        y2 = min(frame.shape[0], int(y2) - 10)

        adjusted_x2 = x2 - FACE_SIZE
        adjusted_y2 = y2 - FACE_SIZE

        draw_bounding_box(frame, score, (adjusted_x2, adjusted_y2), (x2, y2))

        preprocessed_face, adjusted_bbox = preprocess_face_landmarks(
            frame, (adjusted_x2, adjusted_y2, x2, y2), face_landmarks_input_shape
        )

        if preprocessed_face is None:
            continue

        ''' Running inference for Face Landmarks Detection '''
        landmarks = hailo_inference.run(model_id=2, input_data=preprocessed_face)

        landmarks_batch = landmarks.get(face_land_output_name, None)
        if landmarks_batch is None:
            continue

        try:
            # Reshape and adjust landmarks
            landmarks = landmarks_batch[0].reshape(CLASS_NUM // 2, 2)
            adjusted_landmarks = adjust_landmarks(landmarks, (adjusted_x2, adjusted_y2, FACE_SIZE, FACE_SIZE))
            all_landmarks.append(adjusted_landmarks)

            # Blink Detection
            # Assuming landmarks are ordered and follow the 68-point model
            # Left eye indices: 42-47, Right eye indices: 36-41 (0-based)
            # Adjust indices based on your landmark model if different
            left_eye = adjusted_landmarks[42:48]
            right_eye = adjusted_landmarks[36:42]

            left_EAR = calculate_EAR(left_eye)
            right_EAR = calculate_EAR(right_eye)
            avg_EAR = (left_EAR + right_EAR) / 2.0

            if avg_EAR < EAR_THRESHOLD:
                state.EAR_consec_frames += 1
                if not state.is_blinking:
                    state.is_blinking = True
                    state.current_blink_start = time.time()
            else:
                if state.EAR_consec_frames >= CONSEC_FRAMES:
                    state.blink_counter += 1
                    state.total_blinks += 1
                    if state.is_blinking:
                        blink_duration = time.time() - state.current_blink_start
                        state.blink_durations.append(blink_duration)
                        print(f"Blink {state.total_blinks}: Duration = {blink_duration:.3f} seconds")
                state.EAR_consec_frames = 0
                state.is_blinking = False

            # Optionally, display EAR on frame for debugging
            cv2.putText(frame, f"EAR: {avg_EAR:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 250, 100), 2, cv2.LINE_AA)

        except ValueError:
            continue

        # Prepare tensor data for WebSocket with native types
        tensors.append({
            'x1': adjusted_x2,
            'y1': adjusted_y2,
            'x2': x2 + FACE_SIZE,
            'y2': y2 + FACE_SIZE,
            'score': float(score)
        })

async def process_frame(frame, hailo_inference, face_detection_input_shape, face_landmarks_input_shape,
                        face_land_output_name, state: AppState):
    """
    Processes a single video frame: detects faces, detects landmarks, and prepares data for WebSocket.

    Args:
        frame (np.ndarray): The original video frame.
        hailo_inference (HailoInferenceAsyncMultiModel): The inference model.
        face_detection_input_shape (tuple): Input shape for face detection model.
        face_landmarks_input_shape (tuple): Input shape for landmarks model.
        face_land_output_name (str): The output name for landmarks.

    Returns:
        tuple: (processed_frame, tensors, all_landmarks)
    """
    preprocessed_frame, scale, pad_w, pad_h, img_w, img_h = preprocess_face_detection(
        frame, input_size=(face_detection_input_shape[0], face_detection_input_shape[1])
    )
    preprocessed_frame = preprocessed_frame.astype(np.float32)
    preprocessed_frame = preprocessed_frame[np.newaxis, :, :, :]

    ''' Running inference for face detection '''
    raw_faces = hailo_inference.run(model_id=1, input_data=preprocessed_frame)
    faces = postprocess_faces(raw_faces, img_w, img_h, scale, pad_w, pad_h)

    all_landmarks = []
    tensors = []

    if not faces:
        return frame, tensors, all_landmarks

    await handle_faces(faces, frame, hailo_inference, face_land_output_name, face_landmarks_input_shape, all_landmarks,
                      state, tensors)

    return frame, tensors, all_landmarks


async def save_video(buffer, fps, output_path='output_blink_detected.avi'):
    """
    Saves the buffered frames to a video file.

    Args:
        buffer (deque): The frame buffer containing frames to save.
        fps (float): Frames per second of the video.
        output_path (str): Path to save the output video.
    """
    if not buffer:
        print("No frames to save.")
        return

    # Get frame dimensions from the first frame
    height, width, channels = buffer[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in buffer:
        out.write(frame)

    out.release()
    print(f"Saved video to {output_path}")

async def main():
    # HEF paths for the models
    face_det = "../models/hailo8/scrfd_10g.hef"
    face_landmarks = "../models/hailo8/face-landmarks-detection.hef"

    # Initialize application state
    state = AppState()

    # Initialize components
    ws_client = await initialize_websocket(WS_URL, RECONNECT_INTERVAL)
    hailo_inference, face_detection_input_shape, face_landmarks_input_shape, face_land_output_name = initialize_inference_models(face_det, face_landmarks)
    cap = init_cv_cap(640, 360)
    if cap is None:
        return

    # Estimate FPS and set buffer size accordingly
    estimated_fps = estimate_fps(cap, warmup_frames=120)
    state.fps = estimated_fps
    state.buffer_size = int(estimated_fps * BUFFER_DURATION)
    state.frame_buffer = deque(maxlen=state.buffer_size)
    print(f"Estimated FPS: {state.fps:.2f}. Buffer size set to {state.buffer_size} frames.")

    # Reset the video capture to start
    cap.release()
    # cap = init_cv_cap(640, 360)
    cap = init_cv_cap(640, 480)
    if cap is None:
        print("Error: Could not reopen webcam.")
        return

    fps_start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Add frame to buffer
        state.frame_buffer.append(frame.copy())

        fps_end_time = time.time()
        time_diff = fps_end_time - fps_start_time
        state.fps = 1 / time_diff if time_diff > 0 else state.fps
        fps_start_time = fps_end_time

        # Process the current frame
        processed_frame, tensors, all_landmarks = await process_frame(
            frame, hailo_inference, face_detection_input_shape, face_landmarks_input_shape, face_land_output_name, state
        )

        # Draw landmarks on the frame
        draw_landmarks(processed_frame, all_landmarks)

        # Display FPS
        display_fps(processed_frame, state.fps)

        # Display Blink Info
        display_blink_info(processed_frame, state.blink_counter, state.total_blinks, state.blink_durations)

        # Attempt reconnection if WebSocket is not connected
        if ws_client.websocket is None:
            ws_client.first_connection = True
            await ws_client.connect()
            if ws_client.websocket is None:
                await asyncio.sleep(RECONNECT_INTERVAL)
                continue

        # Check if blink count exceeds threshold
        if state.blink_counter >= BLINK_COUNT_THRESHOLD:
            # Save the buffered frames to a video file
            timestamp = datetime.datetime.now()
            output_filename = f"blink_detected_{timestamp.isoformat()}"
            await save_video(state.frame_buffer, state.fps, output_path=output_filename)
            # Clear the buffer and reset blink counter
            state.frame_buffer.clear()
            state.blink_counter = 0
            print(f"Blinks detected: {state.total_blinks}. Buffer cleared after saving video.")

        # Send data via WebSocket
        if ws_client.first_connection:
            # Prepare command outputs if needed
            command_outputs = await ws_client.run_bash_commands()

            # Send the data
            await ws_client.send_data(processed_frame, tensors, command_outputs)
            ws_client.first_connection = False
        else:
            # Send the data without command outputs
            await ws_client.send_data(processed_frame, tensors, command_outputs=None)

        # Show the frame with landmarks
        cv2.imshow('Webcam Face Landmarks', processed_frame)

        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    asyncio.run(main())
