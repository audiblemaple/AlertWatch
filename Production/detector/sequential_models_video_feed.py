# main.py
# !/usr/bin/env python3

import time
import datetime
import cv2
import asyncio
import numpy as np
import threading
import os

from collections import deque
from appState import AppState
from playsound import playsound

from socketUtil import initialize_websocket
from inference import initialize_inference_models
from logger import initialize_logging, log_data
from util import init_cv_cap, calculate_EAR, estimate_fps, save_video_sync, save_video
from drawUtil import draw_bounding_box, display_fps, draw_landmarks, display_blink_info
from prePostProcessing import preprocess_face_detection, postprocess_faces, preprocess_face_landmarks, adjust_landmarks

# Constants
WS_URL: str = "ws://192.168.0.239:5000"
RECONNECT_INTERVAL: int = 2
FACE_SIZE: int = 165
CLASS_NUM: int = 136

# Blink Detection Constants
EAR_THRESHOLD: float = 0.21  # Threshold to detect blink
CONSEC_FRAMES: int = 2  # Number of consecutive frames the EAR should be below threshold

# Buffer Configuration
BUFFER_DURATION: int = 30  # seconds


FACES = None
FRAME = 0
# Function to play alert sound
def play_alert_sound():
    try:
        playsound('alert.wav')  # Ensure you have an 'alert.wav' file in your project directory
    except Exception as e:
        print(f"Error playing sound: {e}")


# Utility function to ensure directory exists
def ensure_directory_exists(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


# Handle faces function with drowsiness detection and delayed video saving
async def handle_faces(faces, frame, hailo_inference, face_land_output_name, face_landmarks_input_shape, all_landmarks,
                       state: AppState, tensors):
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
            print("Preprocessed face is None. Skipping...")
            continue

        ''' Running inference for Face Landmarks Detection '''
        landmarks = hailo_inference.run(model_id=2, input_data=preprocessed_face)

        landmarks_batch = landmarks.get(face_land_output_name, None)
        if landmarks_batch is None:
            print("Landmarks batch is None. Skipping...")
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

            # Update current EAR in state for delayed save check
            state.current_EAR = avg_EAR

            if avg_EAR < EAR_THRESHOLD:
                state.EAR_consec_frames += 1
                if not state.is_blinking:
                    state.is_blinking = True
                    state.current_blink_start = time.time()
            else:
                if state.EAR_consec_frames >= CONSEC_FRAMES:
                    state.blink_counter += 1
                    state.total_blinks += 1
                    state.blink_timestamps.append(time.time())  # Record blink time for rate calculation
                    if state.is_blinking:
                        blink_duration = time.time() - state.current_blink_start
                        state.blink_durations.append(blink_duration)
                        print(f"Blink {state.total_blinks}: Duration = {blink_duration:.3f} seconds")
                state.EAR_consec_frames = 0
                state.is_blinking = False

            # Optionally, display EAR on frame for debugging
            cv2.putText(frame, f"EAR: {avg_EAR:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 250, 100), 2, cv2.LINE_AA)

            # Drowsiness Detection
            drowsy, reasons = state.is_drowsy(avg_EAR, state.current_blink_start if state.is_blinking else 0)
            if drowsy:
                current_time = time.time()

                # Handle Video Saving with Delay
                with state.video_lock:
                    if current_time - state.last_video_time >= state.debounce_time_video:
                        timestamp = datetime.datetime.now()
                        # Ensure the 'videos' directory exists
                        output_filename = f"videos/blink_detected_{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}.avi"
                        # Schedule the delayed_save_video coroutine
                        save_video_sync(state.frame_buffer, state.fps, output_filename)

                        # Update the last alert time
                        state.last_video_time = current_time


                # Handle Alerting
                with state.alert_lock:
                    if current_time - state.last_alert_time >= state.debounce_time_alert:
                        # Trigger visual alert
                        cv2.putText(frame, "DROWSINESS DETECTED!", (10, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                        print(f"Drowsiness Alert: {', '.join(reasons)}")

                        # Trigger auditory alert
                        threading.Thread(target=play_alert_sound, daemon=True).start()

                        # Update the last alert time
                        state.last_alert_time = current_time

                        # Log the drowsiness event
                        log_data(state, drowsy, reasons)

        except ValueError as ve:
            print(f"ValueError during landmarks processing: {ve}")
            continue
        except Exception as e:
            print(f"Unexpected error: {e}")
            continue

        # Prepare tensor data for WebSocket with native types
        tensors.append({
            'x1': adjusted_x2,
            'y1': adjusted_y2,
            'x2': x2 + FACE_SIZE,
            'y2': y2 + FACE_SIZE,
            'score': float(score)
        })


# Process a single frame
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
    global FACES, FRAME
    preprocessed_frame, scale, pad_w, pad_h, img_w, img_h = preprocess_face_detection(
        frame, input_size=(face_detection_input_shape[0], face_detection_input_shape[1])
    )
    preprocessed_frame = preprocessed_frame.astype(np.float32)
    preprocessed_frame = preprocessed_frame[np.newaxis, :, :, :]

    if FACES is None or FRAME % 3 == 0:
        ''' Running inference for face detection '''
        raw_faces = hailo_inference.run(model_id=1, input_data=preprocessed_frame)
        faces = postprocess_faces(raw_faces, img_w, img_h, scale, pad_w, pad_h)
        FACES = faces
    else:
        faces = FACES

    all_landmarks = []
    tensors = []

    if not faces:
        print("No faces detected in this frame.")
        return frame, tensors, all_landmarks

    await handle_faces(faces, frame, hailo_inference, face_land_output_name, face_landmarks_input_shape, all_landmarks,
                       state, tensors)

    return frame, tensors, all_landmarks

# Main function
async def main():
    global FRAME

    # Initialize analysis logging
    initialize_logging()

    # HEF paths for the models
    face_det = "../models/hailo8/scrfd_10g.hef"
    face_landmarks = "../models/hailo8/face-landmarks-detection.hef"

    # Initialize application state
    state = AppState()

    # Ensure the 'videos' directory exists
    ensure_directory_exists('videos')

    # Initialize components
    ws_client = await initialize_websocket(WS_URL, RECONNECT_INTERVAL)
    hailo_inference, face_detection_input_shape, face_landmarks_input_shape, face_land_output_name = initialize_inference_models(
        face_det, face_landmarks)
    cap = init_cv_cap(640, 360)
    if cap is None:
        print("Error: Could not initialize video capture.")
        return

    # Estimate FPS and set buffer size accordingly
    estimated_fps = estimate_fps(cap, warmup_frames=120)
    state.fps = estimated_fps
    state.buffer_size = int(estimated_fps * BUFFER_DURATION)
    state.frame_buffer = deque(maxlen=state.buffer_size)
    print(f"Estimated FPS: {state.fps:.2f}. Buffer size set to {state.buffer_size} frames.")

    # Reset the video capture to start
    cap.release()
    cap = init_cv_cap(640, 360)
    # cap = init_cv_cap(640, 360, 60, "videos/video.mp4")
    if cap is None:
        print("Error: Could not reopen webcam.")
        return


    fps_start_time = time.time()  # Start time for current FPS
    total_start_time = time.time()  # Start time for average FPS
    total_frames = 0  # Total frames processed
    total_time = 0  # Total elapsed time

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        FRAME += 1
        total_frames += 1  # Increment total frame count


        # Add frame to buffer
        state.frame_buffer.append(frame.copy())
        # print(f"Added frame to buffer. Current buffer size: {len(state.frame_buffer)}")

        # Calculate FPS
        fps_end_time = time.time()
        time_diff = fps_end_time - fps_start_time
        total_time = fps_end_time - total_start_time  # Total elapsed time
        state.fps = 1 / time_diff if time_diff > 0 else state.fps
        fps_start_time = fps_end_time

        # Calculate average FPS
        avg_fps = total_frames / total_time if total_time > 0 else 0

        # Process the current frame
        processed_frame, tensors, all_landmarks = await process_frame(
            frame, hailo_inference, face_detection_input_shape, face_landmarks_input_shape, face_land_output_name, state
        )

        # Draw landmarks on the frame
        draw_landmarks(processed_frame, all_landmarks)

        # Display FPS
        display_fps(processed_frame, state.fps, avg_fps)

        # Display Blink Info
        display_blink_info(processed_frame, state.blink_counter, state.total_blinks, state.blink_durations)

        # Attempt reconnection if WebSocket is not connected
        if ws_client.websocket is None:
            ws_client.first_connection = True
            await ws_client.connect()
            if ws_client.websocket is None:
                print("WebSocket not connected. Retrying...")
                await asyncio.sleep(RECONNECT_INTERVAL)
                continue

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


# Optional: Post-analysis function (if needed)
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
    # Uncomment the line below to perform post-analysis after the main loop ends
    analyze_logs()

# #!/usr/bin/env python3
#
# import time
# import datetime
# import cv2
# import asyncio
# import numpy as np
# import threading
# import aiohttp
#
# from collections        import deque
# from appState           import AppState
# from playsound          import playsound
# from socketUtil         import initialize_websocket
# from inference          import initialize_inference_models
# from logger             import initialize_logging, log_data
# from util               import init_cv_cap, calculate_EAR, estimate_fps, save_video, save_video_sync
# from drawUtil           import draw_bounding_box, display_fps, draw_landmarks, display_blink_info
# from prePostProcessing  import preprocess_face_detection, postprocess_faces, preprocess_face_landmarks, adjust_landmarks
#
# # Constants
# server_ip: str = "192.168.0.239:5000"
# WS_URL: str = f"ws://{server_ip}"
# RECONNECT_INTERVAL:int = 2
# FACE_SIZE:int = 130
# CLASS_NUM:int = 136
#
# # Blink Detection Constants
# EAR_THRESHOLD:float = 0.22  # Threshold to detect blink
# CONSEC_FRAMES:int   = 2  # Number of consecutive frames the EAR should be below threshold
#
# # Buffer Configuration
# BUFFER_DURATION:int = 30  # seconds
#
# # Function to play alert sound
# def play_alert_sound():
#     playsound('alert.wav')  # Ensure you have an 'alert.wav' file in your project directory
#
# async def send_drowsiness_alert(url: str, payload: dict, headers: dict = None, retries: int = 3, backoff_factor: float = 0.5):
#     """
#     Sends an HTTP POST request to the specified URL with the given payload and headers.
#     Retries the request on failure based on the specified retry count and backoff factor.
#
#     Args:
#         url (str): The endpoint URL to send the request to.
#         payload (dict): The JSON payload to include in the request body.
#         headers (dict, optional): Additional headers to include in the request.
#         retries (int): Number of retry attempts on failure.
#         backoff_factor (float): Factor by which the delay increases after each retry.
#
#     Returns:
#         bool: True if the request was successful (status code 200-299), False otherwise.
#     """
#     attempt = 0
#     while attempt < retries:
#         try:
#             async with aiohttp.ClientSession() as session:
#                 async with session.post(url, json=payload, headers=headers) as response:
#                     if 200 <= response.status < 300:
#                         print(f"Successfully sent drowsiness alert. Status Code: {response.status}")
#                         return True
#                     else:
#                         print(f"Failed to send drowsiness alert. Status Code: {response.status}")
#         except aiohttp.ClientError as e:
#             print(f"HTTP ClientError: {e}")
#         except asyncio.TimeoutError:
#             print("HTTP request timed out.")
#         except Exception as e:
#             print(f"Unexpected error during HTTP request: {e}")
#
#         attempt += 1
#         delay = backoff_factor * (2 ** (attempt - 1))
#         print(f"Retrying in {delay} seconds... (Attempt {attempt}/{retries})")
#         await asyncio.sleep(delay)
#
#     print("All retry attempts failed. Could not send drowsiness alert.")
#     return False
#
# async def handle_faces(faces, frame, hailo_inference, face_land_output_name, face_landmarks_input_shape, all_landmarks,
#                        state: AppState, tensors):
#     """
#     Processes detected faces: draws bounding boxes, preprocesses faces, runs landmark detection, and handles blink counting.
#
#     Args:
#         faces (list): List of detected faces.
#         frame (np.ndarray): The original video frame.
#         hailo_inference (HailoInferenceAsyncMultiModel): The inference model.
#         face_land_output_name (str): The output name for landmarks.
#         face_landmarks_input_shape (tuple): Input shape for landmarks model.
#         state (AppState): The application state.
#         tensors (list): List to store tensor data for WebSocket.
#     """
#     for (x1, y1, x2, y2, score) in faces:
#         # Clip bounding box coordinates and convert to int
#         x1 = max(0, int(x1))
#         y1 = max(0, int(y1))
#         x2 = min(frame.shape[1], int(x2))
#         y2 = min(frame.shape[0], int(y2) - 10)
#
#         adjusted_x2 = x2 - FACE_SIZE
#         adjusted_y2 = y2 - FACE_SIZE
#
#         draw_bounding_box(frame, score, (adjusted_x2, adjusted_y2), (x2, y2))
#
#         preprocessed_face, adjusted_bbox = preprocess_face_landmarks(
#             frame, (adjusted_x2, adjusted_y2, x2, y2), face_landmarks_input_shape
#         )
#
#         if preprocessed_face is None:
#             continue
#
#         ''' Running inference for Face Landmarks Detection '''
#         landmarks = hailo_inference.run(model_id=2, input_data=preprocessed_face)
#
#         landmarks_batch = landmarks.get(face_land_output_name, None)
#         if landmarks_batch is None:
#             continue
#
#         try:
#             # Reshape and adjust landmarks
#             landmarks = landmarks_batch[0].reshape(CLASS_NUM // 2, 2)
#             adjusted_landmarks = adjust_landmarks(landmarks, (adjusted_x2, adjusted_y2, FACE_SIZE, FACE_SIZE))
#             all_landmarks.append(adjusted_landmarks)
#
#             # Blink Detection
#             # Assuming landmarks are ordered and follow the 68-point model
#             # Left eye indices: 42-47, Right eye indices: 36-41 (0-based)
#             # Adjust indices based on your landmark model if different
#             left_eye = adjusted_landmarks[42:48]
#             right_eye = adjusted_landmarks[36:42]
#
#             left_EAR = calculate_EAR(left_eye)
#             right_EAR = calculate_EAR(right_eye)
#             avg_EAR = (left_EAR + right_EAR) / 2.0
#
#             if avg_EAR < EAR_THRESHOLD:
#                 state.EAR_consec_frames += 1
#                 if not state.is_blinking:
#                     state.is_blinking = True
#                     state.current_blink_start = time.time()
#             else:
#                 if state.EAR_consec_frames >= CONSEC_FRAMES:
#                     state.blink_counter += 1
#                     state.total_blinks += 1
#                     if state.is_blinking:
#                         blink_duration = time.time() - state.current_blink_start
#                         state.blink_durations.append(blink_duration)
#                         print(f"Blink {state.total_blinks}: Duration = {blink_duration:.3f} seconds")
#                 state.EAR_consec_frames = 0
#                 state.is_blinking = False
#
#             # Drowsiness Detection
#             drowsy, reasons = state.is_drowsy(avg_EAR, state.current_blink_start if state.is_blinking else 0)
#             if drowsy:
#                 current_time = time.time()
#
#                 with state.video_lock:
#                     if current_time - state.last_video_time >= state.debounce_time_video:
#                         state.video_save_pending = True
#                         # Save the buffered frames to a video file
#                         timestamp = datetime.datetime.now()
#                         # Replace ':' and other invalid characters
#                         output_filename = f"videos/blink_detected_{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}.avi"
#                         await save_video(state.frame_buffer, state.fps, output_path=output_filename)
#                         # Clear the buffer and reset blink counter
#                         state.frame_buffer.clear()
#                         state.blink_counter = 0
#                         print(f"Blinks detected: {state.total_blinks}. Buffer cleared after saving video.")
#
#                         # Update the last alert time
#                         state.last_video_time = current_time
#
#                 with state.alert_lock:
#                     # if current_time - state.last_alert_time >= state.debounce_time_alert:
#                     #     # Trigger visual alert
#                     #     cv2.putText(frame, "DROWSINESS DETECTED!", (10, 100),
#                     #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
#                     #     print(f"Drowsiness Alert: {', '.join(reasons)}")
#                     #
#                     #     # Trigger auditory alert
#                     #     threading.Thread(target=play_alert_sound, daemon=True).start()
#                     #
#                     #     # Update the last alert time
#                     #     state.last_alert_time = current_time
#                     #
#                     #     # Log the drowsiness event
#                     #     log_data(state, drowsy, reasons)
#                     if current_time - state.last_alert_time >= state.debounce_time_alert:
#                         # Trigger visual alert
#                         cv2.putText(frame, "DROWSINESS DETECTED!", (10, 100),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
#                         print(f"Drowsiness Alert: {', '.join(reasons)}")
#
#                         # Trigger auditory alert
#                         threading.Thread(target=play_alert_sound, daemon=True).start()
#
#                         # Update the last alert time
#                         state.last_alert_time = current_time
#
#                         # Log the drowsiness event
#                         log_data(state, drowsy, reasons)
#
#                         # Send HTTP Alert
#                         payload = {
#                             "timestamp": datetime.datetime.now().isoformat(),
#                             "event": "drowsiness_detected",
#                             "reasons": reasons
#                         }
#                         headers = {
#                             "Content-Type": "application/json",
#                             # "Authorization": "Bearer YOUR_API_TOKEN"  # Uncomment and set if needed
#                         }
#                         asyncio.create_task(send_drowsiness_alert(f"https://{server_ip}/api/v1/alert", payload, headers))
#
#         except ValueError:
#             continue
#
#         # Prepare tensor data for WebSocket with native types
#         tensors.append({
#             'x1': adjusted_x2,
#             'y1': adjusted_y2,
#             'x2': x2 + FACE_SIZE,
#             'y2': y2 + FACE_SIZE,
#             'score': float(score)
#         })
#
# async def process_frame(frame, hailo_inference, face_detection_input_shape, face_landmarks_input_shape,
#                         face_land_output_name, state: AppState):
#     """
#     Processes a single video frame: detects faces, detects landmarks, and prepares data for WebSocket.
#
#     Args:
#         frame (np.ndarray): The original video frame.
#         hailo_inference (HailoInferenceAsyncMultiModel): The inference model.
#         face_detection_input_shape (tuple): Input shape for face detection model.
#         face_landmarks_input_shape (tuple): Input shape for landmarks model.
#         face_land_output_name (str): The output name for landmarks.
#
#     Returns:
#         tuple: (processed_frame, tensors, all_landmarks)
#     """
#     preprocessed_frame, scale, pad_w, pad_h, img_w, img_h = preprocess_face_detection(
#         frame, input_size=(face_detection_input_shape[0], face_detection_input_shape[1])
#     )
#     preprocessed_frame = preprocessed_frame.astype(np.float32)
#     preprocessed_frame = preprocessed_frame[np.newaxis, :, :, :]
#
#     ''' Running inference for face detection '''
#     raw_faces = hailo_inference.run(model_id=1, input_data=preprocessed_frame)
#     faces = postprocess_faces(raw_faces, img_w, img_h, scale, pad_w, pad_h)
#
#     all_landmarks = []
#     tensors = []
#
#     if not faces:
#         return frame, tensors, all_landmarks
#
#     await handle_faces(faces, frame, hailo_inference, face_land_output_name, face_landmarks_input_shape, all_landmarks,
#                       state, tensors)
#
#     return frame, tensors, all_landmarks
#
# async def main():
#     # Initialize analysis logging
#     initialize_logging()
#
#     # HEF paths for the models
#     face_det = "../models/hailo8/scrfd_10g.hef"
#     face_landmarks = "../models/hailo8/face-landmarks-detection.hef"
#
#     # Initialize application state
#     state = AppState()
#
#     # Initialize components
#     ws_client = await initialize_websocket(WS_URL, RECONNECT_INTERVAL)
#     hailo_inference, face_detection_input_shape, face_landmarks_input_shape, face_land_output_name = initialize_inference_models(face_det, face_landmarks)
#     cap = init_cv_cap(640, 360)
#     if cap is None:
#         return
#
#     # Estimate FPS and set buffer size accordingly
#     estimated_fps = estimate_fps(cap, warmup_frames=120)
#     state.fps = estimated_fps
#     state.buffer_size = int(estimated_fps * BUFFER_DURATION)
#     state.frame_buffer = deque(maxlen=state.buffer_size)
#     print(f"Estimated FPS: {state.fps:.2f}. Buffer size set to {state.buffer_size} frames.")
#
#     # Reset the video capture to start
#     cap.release()
#     # cap = init_cv_cap(640, 360)
#     cap = init_cv_cap(640, 360)
#     if cap is None:
#         print("Error: Could not reopen webcam.")
#         return
#
#     fps_start_time = time.time()
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Failed to capture image.")
#             break
#
#         # Add frame to buffer
#         state.frame_buffer.append(frame.copy())
#
#         fps_end_time = time.time()
#         time_diff = fps_end_time - fps_start_time
#         state.fps = 1 / time_diff if time_diff > 0 else state.fps
#         fps_start_time = fps_end_time
#
#         # Process the current frame
#         processed_frame, tensors, all_landmarks = await process_frame(
#             frame, hailo_inference, face_detection_input_shape, face_landmarks_input_shape, face_land_output_name, state
#         )
#
#         # Draw landmarks on the frame
#         draw_landmarks(processed_frame, all_landmarks)
#
#         # Display FPS
#         display_fps(processed_frame, state.fps)
#
#         # Display Blink Info
#         display_blink_info(processed_frame, state.blink_counter, state.total_blinks, state.blink_durations)
#
#         # Attempt reconnection if WebSocket is not connected
#         if ws_client.websocket is None:
#             ws_client.first_connection = True
#             await ws_client.connect()
#             if ws_client.websocket is None:
#                 await asyncio.sleep(RECONNECT_INTERVAL)
#                 continue
#
#
#         # Send data via WebSocket
#         if ws_client.first_connection:
#             # Prepare command outputs if needed
#             command_outputs = await ws_client.run_bash_commands()
#
#             # Send the data
#             await ws_client.send_data(processed_frame, tensors, command_outputs)
#             ws_client.first_connection = False
#         else:
#             # Send the data without command outputs
#             await ws_client.send_data(processed_frame, tensors, command_outputs=None)
#
#         # Show the frame with landmarks
#         cv2.imshow('Webcam Face Landmarks', processed_frame)
#
#         # Check for 'q' key to quit
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             print("Quitting...")
#             break
#
#     # Release resources
#     cap.release()
#     cv2.destroyAllWindows()
#
# if __name__ == '__main__':
#     asyncio.run(main())
