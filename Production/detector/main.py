#!/usr/bin/env python3

"""
Real-time Face Detection and Landmark Inference with WebSocket Streaming

This script performs real-time face detection, landmark inference, and monitoring
using a camera feed. It also streams the processed video frames over a WebSocket server.

Features:
    - Real-time face detection and facial landmarks inference.
    - Blink detection and drowsiness monitoring.
    - WebSocket server for streaming video frames to clients.
    - Modular design with support for Hailo inference models.

Modules:
    - time, os, platform: System utilities for managing time, files, and platform specifics.
    - threading, asyncio: For managing multithreading and asynchronous tasks.
    - collections.deque: For maintaining a rolling buffer of video frames.
    - cv2, numpy: OpenCV and NumPy for image processing.
    - base64: For encoding video frames as base64 strings for WebSocket transmission.
    - inference: Initializes Hailo inference models.
    - util: Utility functions for preprocessing, blink/drowsiness detection, and state management.
    - drawUtil: Functions for drawing bounding boxes, landmarks, and other visual indicators.
    - prePostProcessing: Functions for preprocessing input and postprocessing model outputs.

Constants:
    - CLASS_NUM: Number of facial landmarks (136/2 for x, y coordinates).
    - EAR_THRESHOLD: Eye Aspect Ratio threshold for blink detection.
    - CONSEC_FRAMES: Consecutive frames below EAR threshold to count a blink.
    - BUFFER_DURATION: Duration (in seconds) of the frame buffer.
    - FRAMES_TO_SKIP: Number of frames to skip for face detection.
    - FACES: Store detected face bounding boxes when skipping frames.

Functions:
    - handle_faces(faces, frame, hailo_inference, face_land_output_name, ...): Handles face processing, including landmark inference, blink, and drowsiness detection.
    - get_faces(frame, hailo_inference, face_detection_input_shape): Runs face detection on a video frame and returns bounding boxes.
    - video_processing_loop(hailo_inference, face_detection_input_shape, ...): Processes video frames, detects faces/landmarks, and manages display.
    - send_frames(websocket): Continuously sends frames to a WebSocket client.
    - websocket_handler(websocket, path): Handles new WebSocket connections and sends video frames.
    - start_websocket_server(): Starts the WebSocket server on port 8765.
    - main(): Initializes the system, starts video processing, and launches the WebSocket server.

Usage:
    Run the script to start real-time face monitoring and WebSocket streaming.
    Clients can connect to the WebSocket server to receive the video feed.

Requirements:
    - Python 3.8+
    - OpenCV
    - NumPy
    - Hailo inference libraries
    - asyncio, websockets

Entry Point:
    The `main()` function initializes the system and runs the video processing
    and WebSocket server in separate threads.

Author:
    Lior Jigalo

License:
    MIT
"""
import json
import time
import cv2
import numpy as np
import threading
from collections import deque
import base64
import asyncio
from websockets.server import serve
import platform

from inference import initialize_inference_models
from util     import (
    init_cv_cap, handle_blink_detection, handle_drowsiness_detection,
    run_landmark_inference, AppState, ensure_directory_exists
)
from drawUtil   import (
    draw_bounding_box, display_fps, draw_landmarks, display_blink_info
)
from prePostProcessing import (
    preprocess_face_detection, postprocess_faces,
    preprocess_face_landmarks, adjust_landmarks
)

''' Number of facial landmarks we have '''
CLASS_NUM: int = 136 >> 1

''' Blink Detection Constants '''
EAR_THRESHOLD: float = 0.20
CONSEC_FRAMES: int = 3
FRAMES_WITH_NO_FACE: int = 0

''' How much of the video time to save in the buffer '''
BUFFER_DURATION: int = 30  # seconds

''' Frames to skip for face detection '''
FRAMES_TO_SKIP: int = 2

''' The face to store when skipping frames '''
FACES: int | None = None

''' Latest frame is used to send it over websocket'''
latest_frame = None
''' Lock is used when we send the frames, there are frame drops when sending is locked but that is ok since its used for visualizing only '''
lock = threading.Lock()

"""
Handles face processing, including running landmark inference, blink detection,
and drowsiness monitoring.

Args:
    faces (list): List of detected face bounding boxes.
    frame (ndarray): The current video frame.
    hailo_inference: The inference engine for running models.
    face_land_output_name (str): Output name for face landmarks model.
    face_landmarks_input_shape (tuple): Shape of the face landmarks model input.
    all_landmarks (list): List to store all detected landmarks.
    state (AppState): Application state for tracking blink and drowsiness info.
    tensors (list): List to store face-related data for debugging or analysis.
"""
def handle_faces(
    faces, frame, hailo_inference, face_land_output_name,
    face_landmarks_input_shape, all_landmarks, state: AppState, tensors
) -> None:
    global EAR_THRESHOLD
    """Handle detected faces: run landmark inference, blink, drowsiness detection."""
    x1, y1, x2, y2, score = faces[0]
    draw_bounding_box(frame, score, (x1, y1), (x2, y2))

    preprocessed_face, adjusted_bbox = preprocess_face_landmarks(
        frame, (x1, y1, x2, y2), face_landmarks_input_shape
    )
    if preprocessed_face is None or preprocessed_face.size != 50176:
        # Avoid printing inside hot loop if possible
        return

    # Run Landmark Inference
    landmarks = run_landmark_inference(
        hailo_inference, preprocessed_face, face_land_output_name, CLASS_NUM
    )
    if landmarks is None:
        return

    try:
        adjusted_landmarks = adjust_landmarks(
            landmarks, (x1, y1, x2 - x1, y2 - y1)
        )
        all_landmarks.append(adjusted_landmarks)

        # Blink Detection
        left_eye = adjusted_landmarks[42:48]
        right_eye = adjusted_landmarks[36:42]
        avg_EAR = handle_blink_detection(
            left_eye, right_eye, state, EAR_THRESHOLD, CONSEC_FRAMES
        )
        state.add_ear_measurement(avg_EAR)

        # Get first minute measurement for the EAR values
        elapsed = time.time() - state.start_time
        if elapsed < 90:
            # Collect EAR data for baseline calculation
            state.ear_values_baseline.append(avg_EAR)
        else:
            # If we haven't computed the baseline yet, do it now
            if state.baseline_ear is None and len(state.ear_values_baseline) > 0:
                state.baseline_ear = np.mean(state.ear_values_baseline)
                state.EAR_THRESHOLD = state.baseline_ear * 0.52
                EAR_THRESHOLD = state.baseline_ear * 0.55
                print(f"Baseline EAR computed: {state.baseline_ear:.3f}")

        # If baseline is ready, compare and display difference
        if state.baseline_ear is not None:
            ear_diff = avg_EAR - state.baseline_ear
            cv2.putText(
                frame, f"Diff from baseline: {ear_diff:.2f}",
                (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA
            )

        # Display EAR
        baseline_display = f"{state.baseline_ear:.2f}" if state.baseline_ear is not None else "N/A"

        cv2.putText(
            frame,
            f"EAR / Baseline EAR: {avg_EAR:.2f} / {baseline_display}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,  # font scale
            (100, 250, 100),  # color
            1,  # thickness
            cv2.LINE_AA
        )

        # Drowsiness Detection
        handle_drowsiness_detection(avg_EAR, state, frame)

    except ValueError:
        return
    except Exception as e:
        # Handle unexpected exceptions
        print(f"Error in landmark processing: {e}")
        return

    # Store extra data if needed
    tensors.append({
        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
        'score': float(score)
    })

"""
Detects faces in a video frame.S

Args:
    frame (ndarray): The current video frame.
    hailo_inference: The inference engine for running models.
    face_detection_input_shape (tuple): Shape of the face detection model input.

Returns:
    list | None: List of face bounding boxes with scores, or None if no faces detected.
"""
def get_faces(frame, hailo_inference, face_detection_input_shape) -> list[(int, int, int, int, float)] | None:
    """Preprocess frame, run face detection, and return bounding boxes."""
    preprocessed_frame, scale, pad_w, pad_h, img_w, img_h = preprocess_face_detection(
        frame, input_size=(face_detection_input_shape[0], face_detection_input_shape[1])
    )

    # Check size validity
    if preprocessed_frame.size != 1228800:
        return None

    # In-place cast if possible
    preprocessed_frame = preprocessed_frame.astype(np.float32, copy=False)
    preprocessed_frame = preprocessed_frame[np.newaxis, :, :, :]

    # Running inference for face detection
    raw_faces = hailo_inference.run(model_id=1, input_data=preprocessed_frame)
    return postprocess_faces(raw_faces, pad_w, pad_h)

"""
Continuously processes video frames, runs face detection and landmark inference,
and manages the display and state updates.

Args:
    hailo_inference: The inference engine for running models.
    face_detection_input_shape (tuple): Shape of the face detection model input.
    face_landmarks_input_shape (tuple): Shape of the face landmarks model input.
    face_land_output_name (str): Output name for face landmarks model.
    state (AppState): Application state for tracking application metrics.
"""
def video_processing_loop(
    hailo_inference, face_detection_input_shape, face_landmarks_input_shape,
    face_land_output_name, state: AppState
):
    """Continually capture frames, run face detection/landmarks, and manage display."""
    global latest_frame
    global FRAMES_WITH_NO_FACE

    cap = init_cv_cap(640, 480, 60)
    if cap is None or not cap.isOpened():
        print("Error: Could not open camera.")
        return

    fps_start_time = time.perf_counter()
    total_start_time = fps_start_time
    total_frames = 0
    face_buff = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        total_frames += 1

        # Append to buffer for the video saving
        # state.frame_buffer.append(frame)

        # Calculate instantaneous FPS
        fps_end_time = time.perf_counter()
        time_diff = fps_end_time - fps_start_time
        total_time = fps_end_time - total_start_time
        # Update state.fps only if time_diff > 0
        if time_diff > 0:
            state.fps = 1.0 / time_diff
        fps_start_time = fps_end_time

        # Average FPS
        avg_fps = total_frames / total_time if total_time > 0 else 0.0

        # Run face detection every FRAMES_TO_SKIP frames
        if total_frames % FRAMES_TO_SKIP == 0:
            face = get_faces(frame, hailo_inference, face_detection_input_shape)
            if face is not None:
                face_buff = face
        else:
            face = face_buff

        # If no face detected
        if not face:
            with lock:
                latest_frame = frame
            if platform.node() != 'hailo15':
                cv2.imshow('Webcam Face Landmarks', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Quitting...")
                    break
            continue

        # If face is found, run landmark inference
        all_landmarks = []
        tensors = []
        handle_faces(
            face, frame, hailo_inference, face_land_output_name,
            face_landmarks_input_shape, all_landmarks, state, tensors
        )

        # Draw landmarks, display FPS & blink info
        if all_landmarks:
            draw_landmarks(frame, all_landmarks)

        display_fps(frame, state.fps, avg_fps)
        display_blink_info(
            frame, state.blink_counter, state.total_blinks, state.blink_durations
        )

        # Show the frame or do nothing
        if platform.node() != 'hailo15':
            cv2.imshow('Webcam Face Landmarks', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting...")
                break

        # Update latest_frame for the WS server
        with lock:
            latest_frame = frame

    cap.release()
    cv2.destroyAllWindows()


"""
Sends the latest processed video frames over a WebSocket connection.

Args:
    websocket (WebSocketServerProtocol): WebSocket connection object.
"""
async def send_frames(websocket):
    """Continuously send frames to connected client over WebSocket."""
    while True:
        with lock:
            frame_copy = latest_frame if latest_frame is not None else None

        if frame_copy is not None:
            # Encode the frame as JPEG (returns success flag and encoded img)
            _, jpeg_data = cv2.imencode('.jpg', frame_copy)
            # Convert to base64
            b64_bytes = base64.b64encode(jpeg_data)
            b64_string = b64_bytes.decode('utf-8')

            # Send the base64 string
            await websocket.send(b64_string)

        # Slight throttle to avoid saturating CPU
        await asyncio.sleep(0.016)  # ~ 60 FPS


def get_system_data():
    """
    Returns a dictionary containing platform, CPU, and memory data.
    """
    system_data = {
        'platform': platform.system(),
        'platform_release': platform.release(),
        'architecture': platform.machine(),
        'processor': platform.processor()
        }

    # Basic platform/OS info

    return system_data

"""
Handles incoming WebSocket connections and streams frames to the client.

Args:
    websocket (WebSocketServerProtocol): WebSocket connection object.
    path (str): The path of the WebSocket request.
"""
async def websocket_handler(websocket, path: str) -> None:
    """Handle new WebSocket connection."""
    print("New client connected")

    # Gather system data
    system_info = get_system_data()

    # Send system info as JSON
    await websocket.send(json.dumps({
        "type": "welcome",
        "systemData": system_info
    }))

    # Continuously send frames
    try:
        await send_frames(websocket)
    except Exception as e:
        print(f"Client disconnected: {e}")

"""
Starts the WebSocket server on port 8765 to stream video frames to clients.
"""
async def start_websocket_server() -> None:
    """Launch the WebSocket server on port 8765."""
    async with serve(websocket_handler, "0.0.0.0", 8765):
        print("WebSocket server started on ws://0.0.0.0:8765")
        await asyncio.Future()  # run forever


"""
Initializes the system, starts video processing in a separate thread,
and launches the WebSocket server.
"""
def main() -> None:
    face_det = f"../models/hailo{'15H' if platform.node() == 'hailo15' else '8'}/scrfd_10g.hef"
    face_landmarks = f"../models/hailo{'15H' if platform.node() == 'hailo15' else '8'}/face-landmarks-detection.hef"

    state = AppState()
    # ensure_directory_exists('videos')

    (hailo_inference,
     face_detection_input_shape,
     face_landmarks_input_shape,
     face_land_output_name) = initialize_inference_models(
         face_det, face_landmarks
    )

    # Estimate FPS if not known
    # estimated_fps = 60
    # state.fps = estimated_fps
    # state.buffer_size = int(estimated_fps * BUFFER_DURATION)
    # state.frame_buffer = deque(maxlen=state.buffer_size)
    # print(f"Estimated FPS: {state.fps:.2f}. Buffer size set to {state.buffer_size} frames.")

    # Run video processing in a separate thread
    video_thread = threading.Thread(
        target=video_processing_loop,
        args=(hailo_inference, face_detection_input_shape,
              face_landmarks_input_shape, face_land_output_name, state),
        daemon=True
    )
    video_thread.start()

    # Start the asyncio event loop for the WebSocket server
    asyncio.run(start_websocket_server())

    video_thread.join()

if __name__ == '__main__':
    main()
