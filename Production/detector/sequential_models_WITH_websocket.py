#!/usr/bin/env python3

import time
import cv2
import numpy as np
import os
import threading
from collections import deque
import base64
import asyncio
from websockets.server import serve

from inference  import initialize_inference_models
# from logger     import initialize_logging
from util       import init_cv_cap, handle_blink_detection, handle_drowsiness_detection, process_bounding_box, run_landmark_inference, AppState
from drawUtil   import draw_bounding_box, display_fps, draw_landmarks, display_blink_info
from prePostProcessing import preprocess_face_detection, postprocess_faces, preprocess_face_landmarks, adjust_landmarks

''' Constants '''
CLASS_NUM: int = 136 >> 1

''' Blink Detection Constants '''
EAR_THRESHOLD: float = 0.25
CONSEC_FRAMES: int = 2 # Frames below threshold for a blink

''' Buffer Configuration '''
BUFFER_DURATION: int = 30  # seconds

''' Frames to skip when detecting faces '''
FRAMES_TO_SKIP: int = 2

''' The face to storewhen skipping frames '''
FACES: int | None = None

latest_frame = None  # We'll store the latest frame here so the WS server can access it
lock = threading.Lock()

def ensure_directory_exists(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def handle_faces(faces, frame, hailo_inference, face_land_output_name, face_landmarks_input_shape, all_landmarks,
                 state: AppState, tensors) -> None:
    x1, y1, x2, y2, score = faces[0]
    draw_bounding_box(frame, score, (x1, y1), (x2, y2))

    preprocessed_face, adjusted_bbox = preprocess_face_landmarks(
        frame, (x1, y1, x2, y2), face_landmarks_input_shape
    )
    if preprocessed_face is None:
        print("Preprocessed face is None. Skipping...")
        return None

    if preprocessed_face.size != 50176:
        print("Preprocessed face size is invalid. Skipping...")
        return None

    # Run Landmark Inference
    landmarks = run_landmark_inference(hailo_inference, preprocessed_face, face_land_output_name, CLASS_NUM)
    if landmarks is None:
        print("Landmarks batch is None. Skipping...")
        return None

    try:
        adjusted_landmarks = adjust_landmarks(landmarks, (x1, y1, x2 - x1, y2 - y1))
        all_landmarks.append(adjusted_landmarks)

        # Blink Detection
        left_eye = adjusted_landmarks[42:48]
        right_eye = adjusted_landmarks[36:42]
        avg_EAR = handle_blink_detection(left_eye, right_eye, state, EAR_THRESHOLD, CONSEC_FRAMES)

        state.add_ear_measurement(avg_EAR)

        # Display EAR
        cv2.putText(frame, f"EAR: {avg_EAR:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 250, 100), 2, cv2.LINE_AA)

        # Drowsiness Detection
        handle_drowsiness_detection(avg_EAR, state, frame)

    except ValueError as ve:
        print(f"ValueError during landmarks processing: {ve}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

    # Keep any tensors or data you want to store locally, if needed
    tensors.append({
        'x1': x1,
        'y1': y1,
        'x2': x2,
        'y2': y2,
        'score': float(score)
    })

def get_faces(frame, hailo_inference, face_detection_input_shape) -> list[(int, int, int, int, float)] | None:
    preprocessed_frame, scale, pad_w, pad_h, img_w, img_h = preprocess_face_detection(
        frame, input_size=(face_detection_input_shape[0], face_detection_input_shape[1])
    )

    # Check size validity
    if preprocessed_frame.size != 1228800:
        return None

    preprocessed_frame = preprocessed_frame.astype(np.float32)
    preprocessed_frame = preprocessed_frame[np.newaxis, :, :, :]

    # Running inference for face detection
    raw_faces = hailo_inference.run(model_id=1, input_data=preprocessed_frame)
    return postprocess_faces(raw_faces, pad_w, pad_h)

def video_processing_loop(hailo_inference, face_detection_input_shape, face_landmarks_input_shape, face_land_output_name, state: AppState):
    global latest_frame
    cap = init_cv_cap(640, 480, 70)
    if cap is None:
        print("Error: Could not open camera.")
        return

    fps_start_time = time.time()
    total_start_time = time.time()
    total_frames: int = 0
    face_buff = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        '''
        - This code is used of the camera does not support the needed resolution, so we resize the full frame, its not good 
        - but its something if we have the arducam which supports only 1080P
            # Get the dimensions of the frame
            height, width, _ = frame.shape
            
            # Desired crop size
            resize_width = 680
            resize_height = 360
            
            # Calculate start indices to take a center crop
            frame = cv2.resize(frame, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
        '''

        total_frames += 1
        state.frame_buffer.append(frame.copy())

        # Calculate FPS
        fps_end_time = time.time()
        time_diff = fps_end_time - fps_start_time
        total_time = fps_end_time - total_start_time
        state.fps = 1 / time_diff if time_diff > 0 else state.fps
        fps_start_time = fps_end_time

        # Calculate average FPS
        avg_fps = total_frames / total_time if total_time > 0 else 0

        # Face detection every FRAMES_TO_SKIP frames
        if total_frames % FRAMES_TO_SKIP == 0:
            face = get_faces(frame, hailo_inference, face_detection_input_shape)
            if face is None:
                face = face_buff
                face_buff = None
            else:
                face_buff = face
        else:
            face = face_buff

        if not face:
            with lock:
                latest_frame = frame.copy()
            cv2.imshow('Webcam Face Landmarks', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting...")
                break
            continue

        all_landmarks = []
        tensors = []

        # Analyze face
        handle_faces(face, frame, hailo_inference, face_land_output_name, face_landmarks_input_shape,
                     all_landmarks, state, tensors)

        # Draw landmarks
        draw_landmarks(frame, all_landmarks)

        # Display FPS, Blink info
        display_fps(frame, state.fps, avg_fps)
        display_blink_info(frame, state.blink_counter, state.total_blinks, state.blink_durations)

        cv2.imshow('Webcam Face Landmarks', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

        with lock:
            latest_frame = frame.copy()

    cap.release()
    cv2.destroyAllWindows()

async def send_frames(websocket):
    """Continuously send frames to a single connected client."""
    while True:
        # Grab the latest frame
        with lock:
            frame_copy = latest_frame.copy() if latest_frame is not None else None

        if frame_copy is not None:
            # Encode the frame as JPEG
            _, jpeg_data = cv2.imencode('.jpg', frame_copy)
            # Convert to base64
            b64_bytes = base64.b64encode(jpeg_data)
            b64_string = b64_bytes.decode('utf-8')

            # Send the base64 string over WebSocket
            await websocket.send(b64_string)

        # Throttle frame rate slightly
        await asyncio.sleep(0.03)  # ~30 FPS

async def websocket_handler(websocket, path):
    """Handle a new WebSocket connection."""
    print("New client connected")
    try:
        await send_frames(websocket)
    except Exception as e:
        print(f"Client disconnected: {e}")

async def start_websocket_server():
    """Launch the WebSocket server on port 8765."""
    async with serve(websocket_handler, "0.0.0.0", 8765):
        print("WebSocket server started on ws://0.0.0.0:8765")
        await asyncio.Future()  # run forever

def main():
    # initialize_logging()

    face_det = "../models/hailo8/scrfd_10g.hef"
    face_landmarks = "../models/hailo8/face-landmarks-detection.hef"

    state = AppState()
    ensure_directory_exists('videos')

    hailo_inference, face_detection_input_shape, face_landmarks_input_shape, face_land_output_name = initialize_inference_models(
        face_det, face_landmarks
    )

    estimated_fps = 60
    state.fps = estimated_fps
    state.buffer_size = int(estimated_fps * BUFFER_DURATION)
    state.frame_buffer = deque(maxlen=state.buffer_size)
    print(f"Estimated FPS: {state.fps:.2f}. Buffer size set to {state.buffer_size} frames.")

    # Run video processing in a separate thread
    video_thread = threading.Thread(
        target=video_processing_loop,
        args=(hailo_inference, face_detection_input_shape, face_landmarks_input_shape, face_land_output_name, state),
        daemon=True
    )
    video_thread.start()

    # Start the asyncio event loop for the WebSocket server
    asyncio.run(start_websocket_server())

    video_thread.join()

if __name__ == '__main__':
    main()
