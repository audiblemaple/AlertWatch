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
CLASS_NUM: int = 136 >> 1

''' Blink Detection Constants '''
EAR_THRESHOLD: float = 0.21  # Threshold for blink
CONSEC_FRAMES: int = 2       # Frames below threshold for a blink

''' Buffer Configuration '''
BUFFER_DURATION: int = 30  # seconds

''' Frames to skip when detecting faces '''
FRAMES_TO_SKIP = 4

FACES = None
FRAME = 0

def ensure_directory_exists(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def handle_faces(faces, frame, hailo_inference, face_land_output_name, face_landmarks_input_shape, all_landmarks,
                 state: AppState, tensors):
    for face in faces:
        x1, y1, x2, y2, score = process_bounding_box(face, frame)
        # draw_bounding_box(frame, score, (x1, y1), (x2, y2))
        # draw_bounding_box(frame, score, (adjusted_x2, adjusted_y2,), (x2, y2), (0, 0, 255))

        preprocessed_face, adjusted_bbox = preprocess_face_landmarks(
            frame, (x1, y1, x2, y2), face_landmarks_input_shape
            # frame, (adjusted_x2, adjusted_y2, x2, y2), face_landmarks_input_shape
        )
        if preprocessed_face is None:
            print("Preprocessed face is None. Skipping...")
            continue

        if preprocessed_face.size != 50176:
            print("Preprocessed face size is invalid. Skipping...")
            continue

        # Run Landmark Inference
        landmarks = run_landmark_inference(hailo_inference, preprocessed_face, face_land_output_name, CLASS_NUM)
        if landmarks is None:
            print("Landmarks batch is None. Skipping...")
            continue

        try:
            adjusted_landmarks = adjust_landmarks(landmarks, (x1, y1, x2 - x1, y2 - y1))
            # adjusted_landmarks = adjust_landmarks(landmarks, (adjusted_x2, adjusted_y2, FACE_SIZE, FACE_SIZE))
            all_landmarks.append(adjusted_landmarks)

            # Blink Detection
            left_eye = adjusted_landmarks[42:48]
            right_eye = adjusted_landmarks[36:42]
            avg_EAR = handle_blink_detection(left_eye, right_eye, state, EAR_THRESHOLD, CONSEC_FRAMES)

            # Display EAR
            cv2.putText(frame, f"EAR: {avg_EAR:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 250, 100), 2, cv2.LINE_AA)

            # Drowsiness Detection
            handle_drowsiness_detection(avg_EAR, state, frame)

        except ValueError as ve:
            print(f"ValueError during landmarks processing: {ve}")
        except Exception as e:
            print(f"Unexpected error: {e}")
            continue

        # Prepare tensor data for WebSocket with native types
        tensors.append({
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'score': float(score)
        })

def get_faces(frame, hailo_inference, face_detection_input_shape):
    preprocessed_frame, scale, pad_w, pad_h, img_w, img_h = preprocess_face_detection(
        frame, input_size=(face_detection_input_shape[0], face_detection_input_shape[1])
    )

    if preprocessed_frame.size != 1228800:
        return None

    preprocessed_frame = preprocessed_frame.astype(np.float32)
    preprocessed_frame = preprocessed_frame[np.newaxis, :, :, :]

    # Running inference for face detection
    raw_faces = hailo_inference.run(model_id=1, input_data=preprocessed_frame)
    return postprocess_faces(raw_faces, img_w, img_h, scale, pad_w, pad_h)

def video_processing_loop(hailo_inference, face_detection_input_shape, face_landmarks_input_shape, face_land_output_name, state: AppState, frame_stack: deque):
    global FRAME

    cap = init_cv_cap(640, 360)
    # cap = init_cv_cap(640, 360, 60, "videos/video.mp4")

    if cap is None:
        print("Error: Could not open camera.")
        return

    fps_start_time = time.time()
    total_start_time = time.time()
    total_frames = 0
    face_buff = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
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

        # Calculate average FPS
        avg_fps = total_frames / total_time if total_time > 0 else 0

        # Face detection every FRAMES_TO_SKIP frames
        if total_frames % FRAMES_TO_SKIP == 0:
            faces = get_faces(frame, hailo_inference, face_detection_input_shape)
            if faces is None:
                faces = face_buff
            else:
                face_buff = faces
        else:
            faces = face_buff

        if not faces:
            ''' No faces found, just show frame '''
            cv2.imshow('Webcam Face Landmarks', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting...")
                break
            continue

        all_landmarks = []
        tensors = []

        # Analyze faces and get tensors
        handle_faces(faces, frame, hailo_inference, face_land_output_name, face_landmarks_input_shape,
                     all_landmarks, state, tensors)

        # Draw landmarks
        draw_landmarks(frame, all_landmarks)

        # Display FPS, Blink info
        display_fps(frame, state.fps, avg_fps)
        display_blink_info(frame, state.blink_counter, state.total_blinks, state.blink_durations)

        # Push the processed frame and tensors onto the stack
        frame_stack.append((frame.copy(), tensors))

        cv2.imshow('Webcam Face Landmarks', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
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

        # If the stack is empty, just wait briefly
        if not frame_stack:
            await asyncio.sleep(0.01)
            continue

        # Get next frame/tensors
        frame, tensors = frame_stack.pop()  # LIFO

        # On first connection, run commands
        if first_connection:
            command_outputs = await ws_client.run_bash_commands()
            await ws_client.send_data(frame, tensors, command_outputs)
            first_connection = False
        else:
            await ws_client.send_data(frame, tensors, command_outputs=None)

        await asyncio.sleep(0.01)

async def main():
    initialize_logging()

    face_det = "../models/hailo8/scrfd_10g.hef"
    face_landmarks = "../models/hailo8/face-landmarks-detection.hef"

    state = AppState()
    ensure_directory_exists('videos')

    hailo_inference, face_detection_input_shape, face_landmarks_input_shape, face_land_output_name = initialize_inference_models(
        face_det, face_landmarks
    )

    # Open a temporary capture to estimate FPS
    cap = init_cv_cap(640, 360)
    if cap is None:
        print("Error: Could not initialize video capture for FPS estimation.")
        return
    estimated_fps = estimate_fps(cap, warmup_frames=120)
    state.fps = estimated_fps
    state.buffer_size = int(estimated_fps * BUFFER_DURATION)
    state.frame_buffer = deque(maxlen=state.buffer_size)
    print(f"Estimated FPS: {state.fps:.2f}. Buffer size set to {state.buffer_size} frames.")
    cap.release()

    frame_stack = deque()

    # Run video processing in a separate thread
    video_thread = threading.Thread(
        target=video_processing_loop,
        args=(hailo_inference, face_detection_input_shape, face_landmarks_input_shape, face_land_output_name, state, frame_stack),
        daemon=True
    )
    video_thread.start()

    # Run websocket sending loop in the main asyncio event loop
    await websocket_sending_loop(WS_URL, RECONNECT_INTERVAL, frame_stack)

def analyze_logs(filename='drowsiness_log.csv'):
    try:
        import pandas as pd
        import matplotlib.pyplot as plt

        df = pd.read_csv(filename, parse_dates=['Timestamp'])
        df.set_index('Timestamp', inplace=True)

        # Plot Blink Rate Over Time
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df['Blink Rate (blinks/min)'], label='Blink Rate')
        plt.axhline(y=15, color='r', linestyle='--', label='Blink Rate Threshold')
        plt.xlabel('Time')
        plt.ylabel('Blinks per Minute')
        plt.title('Blink Rate Over Time')
        plt.legend()
        plt.show()

        # Plot Average Blink Duration Over Time
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df['Average Blink Duration'], label='Avg Blink Duration')
        plt.axhline(y=0.4, color='r', linestyle='--', label='Blink Duration Threshold')
        plt.xlabel('Time')
        plt.ylabel('Seconds')
        plt.title('Average Blink Duration Over Time')
        plt.legend()
        plt.show()

        # Highlight Drowsiness Events
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
    # Optionally:
    # analyze_logs()
