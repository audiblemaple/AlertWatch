"""
Drowsiness Detection and Blink Monitoring Utilities

This module provides utilities for processing face bounding boxes, running landmark inference,
detecting blinks, and identifying signs of drowsiness. It also includes functionality for sending
alerts over WebSocket and saving video when drowsiness is detected.

Constants:
    - WS_URL (str): WebSocket URL for sending alerts.
    - RECONNECT_INTERVAL (int): Interval in seconds for reconnect attempts to the WebSocket server.

Functions:
    - process_bounding_box: Processes a bounding box to fit within frame boundaries.
    - run_landmark_inference: Runs inference for facial landmarks and reshapes the output.
    - handle_blink_detection: Detects blinks based on Eye Aspect Ratio (EAR) and updates state.
    - handle_drowsiness_detection: Detects drowsiness based on EAR and triggers alerts or video saving.
    - send_drowsiness_alert: Sends drowsiness alerts as JSON messages over a WebSocket.

Dependencies:
    - asyncio: For asynchronous WebSocket communication.
    - datetime: For timestamping video files and alerts.
    - json: For creating JSON messages.
    - threading: For running alert and video saving tasks in separate threads.
    - time: For measuring durations and handling debouncing.
    - cv2: For OpenCV operations like drawing text on frames.

Usage:
    Use the functions to process video frames, monitor blinks, and handle drowsiness alerts in
    real-time applications.

Author:
    Lior Jigalo

License:
    MIT
"""

import asyncio
import datetime
import json
import threading
import time
import cv2
import numpy as np

from .eyeController import  calculate_EAR
from .websocketController import initialize_websocket
from .videoController import save_video_sync

WS_URL: str = "ws://192.168.0.63:5000"
RECONNECT_INTERVAL: int = 2

def process_bounding_box(face, frame) -> tuple[int, int, int, int, float]:
    """
    Processes a bounding box to ensure it fits within frame boundaries.

    Args:
        face (tuple): Bounding box coordinates and confidence score (x1, y1, x2, y2, score).
        frame (np.ndarray): The video frame.

    Returns:
        tuple: Adjusted bounding box coordinates (x1, y1, x2, y2, score).
    """
    x1, y1, x2, y2, score = face
    # Access frame.shape once
    h, w = frame.shape[:2]

    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2) - 10)

    return x1, y1, x2, y2, score


def run_landmark_inference(hailo_inference, preprocessed_face, face_land_output_name, class_num) -> np.ndarray | None:
    """
    Runs inference for facial landmarks using a Hailo inference object.

    Args:
        hailo_inference: The inference object for running models.
        preprocessed_face (np.ndarray): Preprocessed face ROI.
        face_land_output_name (str): Name of the output tensor for landmarks.
        class_num (int): Number of landmark points.

    Returns:
        np.ndarray | None: Reshaped landmark coordinates (class_num x 2), or None if no output.
    """
    # Avoid repeated dictionary lookups
    landmarks = hailo_inference.run(model_id=2, input_data=preprocessed_face)
    landmarks_batch = landmarks.get(face_land_output_name, None)
    if landmarks_batch is not None:
        # Directly reshape and return
        return landmarks_batch[0].reshape(class_num, 2)
    return None


def handle_blink_detection(left_eye, right_eye, state, ear_threshold, consec_frames) -> float:
    """
    Detects blinks based on the Eye Aspect Ratio (EAR) and updates the application state.

    Args:
        left_eye (np.ndarray): Landmarks for the left eye.
        right_eye (np.ndarray): Landmarks for the right eye.
        state (AppState): The application state object.
        ear_threshold (float): Threshold for EAR to indicate eye closure.
        consec_frames (int): Minimum consecutive frames below threshold to count as a blink.

    Returns:
        float: The average EAR value for the current frame.
    """
    # Compute EAR once
    left_ear = calculate_EAR(left_eye)
    right_ear = calculate_EAR(right_eye)
    avg_ear = (left_ear + right_ear) / 2.0
    state.current_EAR = avg_ear

    if avg_ear < ear_threshold:
        state.EAR_consec_frames += 1
        if not state.is_blinking:
            state.is_blinking = True
            state.current_blink_start = time.time()
    else:
        if state.EAR_consec_frames >= consec_frames:
            state.blink_counter += 1
            state.total_blinks += 1
            now = time.time()
            state.blink_timestamps.append(now)
            if state.is_blinking:
                blink_duration = now - state.current_blink_start
                state.blink_durations.append(blink_duration)
                # Consider removing prints in production
                print(f"Blink {state.total_blinks}: Duration = {blink_duration:.3f} seconds")

        # Reset blinking state
        state.EAR_consec_frames = 0
        state.is_blinking = False

    return avg_ear


def handle_drowsiness_detection(avg_EAR, state, frame) -> None:
    """
    Detects drowsiness based on the EAR and application state, triggers alerts or video saving.

    Args:
        avg_EAR (float): The average EAR value for the current frame.
        state (AppState): The application state object.
        frame (np.ndarray): The video frame for annotation and display.
    """
    # Call time.time() once
    current_time = time.time()
    # Get drowsiness once
    drowsy, reason = state.is_drowsy(avg_EAR, state.current_blink_start if state.is_blinking else 0)

    if drowsy:
        ''' Handle Video Saving '''
        with state.video_lock:
            if (current_time - state.last_video_time) >= state.debounce_time_video:
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                output_filename = f"videos/blink_detected_{timestamp}.avi"
                video_thread = threading.Thread(
                    target=save_video_sync,
                    args=(state.frame_buffer, state.fps, output_filename),
                    daemon=True  # Use daemon=True so it won't block app exit
                )
                video_thread.start()
                state.last_video_time = current_time

        ''' Handle Alerts '''
        with state.alert_lock:
            if (current_time - state.last_alert_time) >= state.debounce_time_alert:
                cv2.putText(frame, "DROWSINESS DETECTED!", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                # Consider removing prints or using a faster logging method
                print(f"Drowsiness Alert: {reason}")

                state.last_alert_time = current_time

                threading.Thread(
                    target=send_drowsiness_alert,
                    args=(WS_URL, RECONNECT_INTERVAL, reason),
                    daemon=True
                ).start()

def send_drowsiness_alert(ws_url: str, reconnect_interval: int, reason: str) -> None:
    """
    Sends a drowsiness alert as a JSON message over a WebSocket connection.

    Args:
        ws_url (str): The WebSocket server URL.
        reconnect_interval (int): Interval in seconds for reconnect attempts.
        reason (str): Reason for the drowsiness alert.
    """
    async def async_send():
        ws_client = await initialize_websocket(ws_url, reconnect_interval)
        if ws_client.websocket is not None:
            try:
                message = {
                    "type": "alert",
                    "msgData": reason,
                    "event": "drowsiness"
                }
                # Convert Python dict to JSON string
                await ws_client.websocket.send(json.dumps(message))
            finally:
                # Close the connection after sending
                await ws_client.websocket.close()

    # Run the async portion in a blocking fashion within this thread
    asyncio.run(async_send())
