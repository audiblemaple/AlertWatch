#!/usr/bin/env python3

import time
import cv2
import asyncio
import numpy as np
import os
import threading
from collections import deque

from appState   import AppState
from socketUtil import initialize_websocket
from inference  import initialize_inference_models
from logger     import initialize_logging
from util       import init_cv_cap, estimate_fps, handle_blink_detection, handle_drowsiness_detection, process_bounding_box, run_landmark_inference
from drawUtil   import draw_bounding_box, display_fps, draw_landmarks, display_blink_info
from prePostProcessing import preprocess_face_detection, postprocess_faces, preprocess_face_landmarks, adjust_landmarks

''' Constants '''
WS_URL: str = "ws://192.168.0.239:5000"
RECONNECT_INTERVAL: int = 2
FACE_SIZE: int = 165
CLASS_NUM: int = 136 >> 1

''' Blink Detection Constants '''
EAR_THRESHOLD: float = 0.21  # Threshold to detect blink
CONSEC_FRAMES: int = 2       # Number of consecutive frames the EAR should be below threshold

''' Buffer Configuration '''
BUFFER_DURATION: int = 30  # seconds
FRAMES_TO_SKIP = 2

FACES = None
FRAME = 0

def ensure_directory_exists(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def handle_faces_sync(faces, frame, hailo_inference, face_land_output_name, face_landmarks_input_shape, all_landmarks,
                      state: AppState, tensors):
    for face in faces:
        x1, y1, adjusted_x2, adjusted_y2, x2, y2, score = process_bounding_box(face, frame)
        draw_bounding_box(frame, score, (adjusted_x2, adjusted_y2), (x2, y2))

        preprocessed_face, adjusted_bbox = preprocess_face_landmarks(
            frame, (adjusted_x2, adjusted_y2, x2, y2), face_landmarks_input_shape
        )
        if preprocessed_face is None:
            print("Preprocessed face is None. Skipping...")
            continue

        # Run Landmark Inference
        landmarks = run_landmark_inference(hailo_inference, preprocessed_face, face_land_output_name, CLASS_NUM)
        if landmarks is None:
            print("Landmarks batch is None. Skipping...")
            continue

        try:
            adjusted_landmarks = adjust_landmarks(landmarks, (adjusted_x2, adjusted_y2, FACE_SIZE, FACE_SIZE))
            all_landmarks.append(adjusted_landmarks)

            # Blink Detection
            left_eye = adjusted_landmarks[42:48]
            right_eye = adjusted_landmarks[36:42]
            avg_EAR = handle_blink_detection(left_eye, right_eye, state, EAR_THRESHOLD, CONSEC_FRAMES)

            # Optionally, display EAR
            cv2.putText(frame, f"EAR: {avg_EAR:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 250, 100), 2, cv2.LINE_AA)

            # Drowsiness Detection
            handle_drowsiness_detection(avg_EAR, state, frame)

        except ValueError as ve:
            print(f"ValueError during landmarks processing: {ve}")
        except Exception as e:
            print(f"Unexpected error: {e}")
            continue

        # Prepare tensor data for WebSocket
        tensors.append({
            'x1': adjusted_x2,
            'y1': adjusted_y2,
            'x2': x2 + FACE_SIZE,
            'y2': y2 + FACE_SIZE,
            'score': float(score)
        })


def get_faces_sync(frame, hailo_inference, face_detection_input_shape):
    preprocessed_frame, scale, pad_w, pad_h, img_w, img_h = preprocess_face_detection(
        frame, input_size=(face_detection_input_shape[0], face_detection_input_shape[1])
    )
    preprocessed_frame = preprocessed_frame.astype(np.float32)
    preprocessed_frame = preprocessed_frame[np.newaxis, :, :, :]

    raw_faces = hailo_inference.run(model_id=1, input_data=preprocessed_frame)
    return postprocess_faces(raw_faces, img_w, img_h, scale, pad_w, pad_h)

def video_capture_thread(hailo_inference, face_detection_input_shape, face_landmarks_input_shape, face_land_output_name, state: AppState, frame_stack: deque):
    global FRAME

    cap = init_cv_cap(640, 360, 60, "videos/video.mp4")
    if cap is None:
        print("Error: Could not reopen video source.")
        return

    fps_start_time = time.time()
    total_start_time = time.time()
    total_frames = 0
    face_buff = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No more frames or error reading frame.")
            break

        FRAME += 1
        total_frames += 1

        state.frame_buffer.append(frame.copy())

        # Calculate FPS
        fps_end_time = time.time()
        time_diff = fps_end_time - fps_start_time
        total_time = fps_end_time - total_start_time
        state.fps = 1 / time_diff if time_diff > 0 else state.fps
        fps_start_time = fps_end_time
        avg_fps = total_frames / total_time if total_time > 0 else 0

        # Face detection every FRAMES_TO_SKIP frames
        if total_frames % FRAMES_TO_SKIP == 0:
            faces = get_faces_sync(frame, hailo_inference, face_detection_input_shape)
            face_buff = faces
        else:
            faces = face_buff

        if not faces:
            # Show the frame without annotations if no faces
            cv2.imshow('Webcam Face Landmarks', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        all_landmarks = []
        tensors = []

        # Analyze faces synchronously
        handle_faces_sync(faces, frame, hailo_inference, face_land_output_name, face_landmarks_input_shape,
                          all_landmarks, state, tensors)

        # Draw landmarks and other info
        draw_landmarks(frame, all_landmarks)
        display_fps(frame, state.fps, avg_fps)
        display_blink_info(frame, state.blink_counter, state.total_blinks, state.blink_durations)

        # Add frame and tensors to the stack (for the async WebSocket sender)
        # We store a copy to ensure we don't lose data
        frame_stack.append((frame.copy(), tensors))

        cv2.imshow('Webcam Face Landmarks', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting capture thread...")
            break

    cap.release()
    cv2.destroyAllWindows()


async def websocket_sending_loop(ws_url: str, reconnect_interval: int, frame_stack: deque):
    ws_client = await initialize_websocket(ws_url, reconnect_interval)
    first_connection = True

    while True:
        # Attempt reconnection if needed
        if ws_client.websocket is None:
            ws_client.first_connection = True
            await ws_client.connect()
            if ws_client.websocket is None:
                print("WebSocket not connected. Retrying...")
                await asyncio.sleep(reconnect_interval)
                continue
            else:
                first_connection = True

        # If no frames, just wait a bit
        if not frame_stack:
            await asyncio.sleep(0.01)
            continue

        # Pop from the stack (LIFO). If you prefer FIFO, use popleft()
        frame, tensors = frame_stack.pop()

        # Send data via WebSocket
        if first_connection:
            command_outputs = await ws_client.run_bash_commands()
            await ws_client.send_data(frame, tensors, command_outputs)
            first_connection = False
        else:
            await ws_client.send_data(frame, tensors, command_outputs=None)

        await asyncio.sleep(0.01)  # small sleep to allow event loop to cycle


async def main():
    initialize_logging()
    face_det = "../models/hailo8/scrfd_10g.hef"
    face_landmarks = "../models/hailo8/face-landmarks-detection.hef"

    state = AppState()
    ensure_directory_exists('videos')

    hailo_inference, face_detection_input_shape, face_landmarks_input_shape, face_land_output_name = initialize_inference_models(
        face_det, face_landmarks)

    # Temporary cap to estimate FPS
    cap = init_cv_cap(640, 360)
    if cap is None:
        print("Error: Could not initialize video capture.")
        return
    estimated_fps = estimate_fps(cap, warmup_frames=60)
    cap.release()

    state.fps = estimated_fps
    state.buffer_size = int(estimated_fps * BUFFER_DURATION)
    state.frame_buffer = deque(maxlen=state.buffer_size)
    print(f"Estimated FPS: {state.fps:.2f}. Buffer size set to {state.buffer_size} frames.")

    frame_stack = deque()

    # Start the video capture thread
    capture_thread = threading.Thread(
        target=video_capture_thread,
        args=(hailo_inference, face_detection_input_shape, face_landmarks_input_shape, face_land_output_name, state, frame_stack),
        daemon=True
    )
    capture_thread.start()

    # Run the websocket sending loop
    await websocket_sending_loop(WS_URL, RECONNECT_INTERVAL, frame_stack)

def analyze_logs(filename='drowsiness_log.csv'):
    try:
        import pandas as pd
        import matplotlib.pyplot as plt

        df = pd.read_csv(filename, parse_dates=['Timestamp'])
        df.set_index('Timestamp', inplace=True)

        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df['Blink Rate (blinks/min)'], label='Blink Rate')
        plt.axhline(y=15, color='r', linestyle='--', label='Blink Rate Threshold')
        plt.xlabel('Time')
        plt.ylabel('Blinks per Minute')
        plt.title('Blink Rate Over Time')
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df['Average Blink Duration'], label='Avg Blink Duration')
        plt.axhline(y=0.4, color='r', linestyle='--', label='Blink Duration Threshold')
        plt.xlabel('Time')
        plt.ylabel('Seconds')
        plt.title('Average Blink Duration Over Time')
        plt.legend()
        plt.show()

        drowsy_events = df[df['Drowsy'] == True]
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df['Blink Rate (blinks/min)'], label='Blink Rate')
        plt.scatter(drowsy_events.index, drowsy_events['Blink Rate (blinks/min)'], color='r',
                    label='Drowsiness Detected')
        plt.axhline(y=15, color='r', linestyle='--', label='Blink Rate Threshold')
        plt.xlabel('Time')
        plt.ylabel('Blinks per Minute')
        plt.title('Drowsiness Events Highlighted on Blink Rate')
        plt.legend()
        plt.show()
    except FileNotFoundError:
        print(f"Log file '{filename}' not found.")
    except Exception as e:
        print(f"Error analyzing logs: {e}")

if __name__ == '__main__':
    asyncio.run(main())
    # analyze_logs()
