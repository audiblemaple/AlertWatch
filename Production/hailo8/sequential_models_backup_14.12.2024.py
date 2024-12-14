#!/usr/bin/env python3
import time
import datetime
import cv2
import asyncio
import numpy as np
from collections import deque
from socketUtil.websocketController import WebSocketClient  # Adjust import as needed
from drawUtil import draw_bounding_box
from inference import HailoInferenceAsyncMultiModel
from processingUtil import preprocess_faces, postprocess_faces
from util import init_cv_cap
from dataclasses import dataclass, field

# Constants
WS_URL = "ws://192.168.0.239:5000"
RECONNECT_INTERVAL = 2
FACE_SIZE = 140
CLASS_NUM = 136

# Blink Detection Constants
EAR_THRESHOLD = 0.2  # Threshold to detect blink
CONSEC_FRAMES = 2  # Number of consecutive frames the EAR should be below threshold
BLINK_COUNT_THRESHOLD = 10  # Number of blinks to trigger video saving

# Buffer Configuration
BUFFER_DURATION = 15  # seconds


@dataclass
class AppState:
    blink_counter: int = 0
    total_blinks: int = 0
    EAR_consec_frames: int = 0
    frame_buffer: deque = field(default_factory=lambda: deque())
    buffer_size: int = 300
    fps: float = 0.0
    blink_durations: list = field(default_factory=list)  # Stores duration of each blink
    is_blinking: bool = False  # Indicates if a blink is ongoing
    current_blink_start: float = 0.0  # Timestamp when the current blink started


def preprocess_face(frame, bbox, input_shape, gray=True):
    """
    Preprocesses the face ROI for landmark detection.

    Args:
        frame (np.ndarray): The original video frame.
        bbox (tuple): Bounding box of the face (x1, y1, x2, y2).
        input_shape (tuple): The target input shape for the model.
        gray (bool): Whether to convert the face ROI to grayscale.

    Returns:
        tuple: (preprocessed_face, adjusted_bbox) or (None, bbox) if invalid.
    """
    x1, y1, x2, y2 = bbox

    # Ensure the coordinates are integers and within image bounds
    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(frame.shape[1], int(round(x2)))
    y2 = min(frame.shape[0], int(round(y2)))

    # Validate bounding box dimensions
    if x2 <= x1 or y2 <= y1:
        return None, bbox

    face_roi = frame[y1:y2, x1:x2]

    if face_roi.size == 0:
        return None, bbox

    if gray:
        # Convert to grayscale
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

    # Original face ROI dimensions
    face_h, face_w = face_roi.shape[:2]
    target_h, target_w = input_shape[:2]

    # Validate face ROI dimensions
    if face_w <= 0 or face_h <= 0:
        return None, bbox

    # Calculate scaling factor to ensure the resized image is at least as big as the target size
    scale = max(target_w / face_w, target_h / face_h)

    # New dimensions after scaling
    new_w = int(face_w * scale)
    new_h = int(face_h * scale)

    # Validate new dimensions
    if new_w <= 0 or new_h <= 0:
        return None, bbox

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

    preprocessed_face[y_start:y_end, x_start:x_end] = resized_face[resized_y_start:resized_y_end,
                                                      resized_x_start:resized_x_end]

    # Expand dimensions to fit model input
    if gray:
        preprocessed_face = preprocessed_face[np.newaxis, :, :, np.newaxis]
    else:
        preprocessed_face = preprocessed_face[np.newaxis, :, :, :]

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


def calculate_EAR(eye):
    """
    Calculates the Eye Aspect Ratio (EAR) for blink detection.

    Args:
        eye (np.ndarray): Array of eye landmarks.

    Returns:
        float: The EAR value.
    """
    # Compute the euclidean distances between the vertical eye landmarks
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])

    # Compute the euclidean distance between the horizontal eye landmarks
    C = np.linalg.norm(eye[0] - eye[3])

    # Compute EAR
    ear = (A + B) / (2.0 * C)
    return ear


def draw_landmarks(frame, all_landmarks) -> None:
    """
    Draws detected landmarks on the frame.

    Args:
        frame (np.ndarray): The video frame.
        all_landmarks (list): List of arrays containing landmark coordinates.
    """
    for landmarks in all_landmarks:
        for (x, y) in landmarks:
            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)


def display_fps(frame, fps) -> None:
    """
    Draws FPS information on the frame.

    Args:
        frame (np.ndarray): The video frame.
        fps (float): Current Frames Per Second.
    """
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 250, 100), 2, cv2.LINE_AA)


def display_blink_info(frame, blink_count, total_blinks, blink_durations) -> None:
    """
    Draws blink count and average blink duration information on the frame.

    Args:
        frame (np.ndarray): The video frame.
        blink_count (int): Number of blinks since the last reset.
        total_blinks (int): Total number of blinks detected.
        blink_durations (list): List of blink durations.
    """
    cv2.putText(frame, f"Blinks: {blink_count}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Total Blinks: {total_blinks}", (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    if blink_durations:
        average_duration = sum(blink_durations) / len(blink_durations)
        cv2.putText(frame, f"Avg Blink Dur: {average_duration:.2f}s", (10, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)


async def initialize_websocket() -> WebSocketClient:
    """
    Initializes the WebSocket client.

    Returns:
        WebSocketClient: An instance of the WebSocketClient.
    """
    ws_client = WebSocketClient(WS_URL, RECONNECT_INTERVAL)
    await ws_client.run_bash_commands()  # Run initial bash commands
    await ws_client.connect()
    return ws_client


def initialize_inference_models():
    """
    Initializes the Hailo inference models.

    Returns:
        tuple: (hailo_inference, face_detection_input_shape, face_landmarks_input_shape, face_land_output_name)
    """
    face_det = "../models/hailo8/scrfd_10g.hef"
    face_landmark = "../models/hailo8/face-landmarks-detection.hef"

    model_paths = (face_det, face_landmark)
    hailo_inference = HailoInferenceAsyncMultiModel(
        model_paths, input_types=('FLOAT32', 'UINT8'), output_types=('FLOAT32', 'FLOAT32')
    )

    face_detection_input_shape = hailo_inference.get_input_shape(model_id=1)
    face_landmarks_input_shape = hailo_inference.get_input_shape(model_id=2)

    face_land_output_name = hailo_inference.models['model_2']['output_vstream_info'][0].name

    return hailo_inference, face_detection_input_shape, face_landmarks_input_shape, face_land_output_name


def initialize_video_capture(width=640, height=360):
    """
    Initializes video capture from the webcam.

    Args:
        width (int): Width of the video frame.
        height (int): Height of the video frame.

    Returns:
        cv2.VideoCapture: The video capture object or None if failed.
    """
    cap = init_cv_cap(width, height)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None
    return cap

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
        y2 = min(frame.shape[0], int(y2))

        adjusted_x2 = x2 - FACE_SIZE
        adjusted_y2 = y2 - FACE_SIZE

        draw_bounding_box(frame, score, (adjusted_x2, adjusted_y2), (x2, y2))

        preprocessed_face, adjusted_bbox = preprocess_face(
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
    preprocessed_frame, scale, pad_w, pad_h, img_w, img_h = preprocess_faces(
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
    # Initialize application state
    state = AppState()

    # Initialize components
    ws_client = await initialize_websocket()
    hailo_inference, face_detection_input_shape, face_landmarks_input_shape, face_land_output_name = initialize_inference_models()
    cap = initialize_video_capture(640, 360)
    if cap is None:
        return

    # Estimate FPS and set buffer size accordingly
    estimated_fps = estimate_fps(cap, warmup_frames=180)
    state.fps = estimated_fps
    state.buffer_size = int(estimated_fps * BUFFER_DURATION)
    state.frame_buffer = deque(maxlen=state.buffer_size)
    print(f"Estimated FPS: {state.fps:.2f}. Buffer size set to {state.buffer_size} frames.")

    # Reset the video capture to start
    cap.release()
    cap = initialize_video_capture(640, 360)
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
            # timestamp = int(time.time())
            timestamp = datetime.datetime.now()
            timestamp = timestamp.isoformat()
            output_filename = f"blink_detected_{timestamp}.avi"
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


def estimate_fps(cap, warmup_frames=30):
    """
    Estimates the FPS of the video capture device.

    Args:
        cap (cv2.VideoCapture): The video capture object.
        warmup_frames (int): Number of frames to capture for estimation.

    Returns:
        float: Estimated FPS.
    """
    start_time = time.time()
    for _ in range(warmup_frames):
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image during warm-up.")
            break
    end_time = time.time()
    elapsed = end_time - start_time
    estimated_fps = warmup_frames / elapsed if elapsed > 0 else 30
    return estimated_fps


if __name__ == '__main__':
    asyncio.run(main())
