import asyncio
import datetime
import json
import threading
import time

import cv2
from playsound import playsound

from Production.detector.logger import log_data
from Production.detector.socketUtil import initialize_websocket
from Production.detector.util import save_video_sync, calculate_EAR

# WS_URL: str = "ws://192.168.0.239:5000"
WS_URL: str = "ws://192.168.0.63:5000"
RECONNECT_INTERVAL: int = 2

def process_bounding_box(face, frame):
    x1, y1, x2, y2, score = face
    # Access frame.shape once
    h, w = frame.shape[:2]

    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2) - 10)


    return x1, y1, x2, y2, score


def run_landmark_inference(hailo_inference, preprocessed_face, face_land_output_name, class_num):
    # Avoid repeated dictionary lookups
    landmarks = hailo_inference.run(model_id=2, input_data=preprocessed_face)
    landmarks_batch = landmarks.get(face_land_output_name, None)
    if landmarks_batch is not None:
        # Directly reshape and return
        return landmarks_batch[0].reshape(class_num, 2)
    return None


def handle_blink_detection(left_eye, right_eye, state, ear_threshold, consec_frames):
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


def handle_drowsiness_detection(avg_EAR, state, frame):
    # Call time.time() once
    current_time = time.time()
    # Get drowsiness once
    drowsy, reasons = state.is_drowsy(avg_EAR, state.current_blink_start if state.is_blinking else 0)

    if drowsy:
        # Handle Video Saving
        with state.video_lock:
            if (current_time - state.last_video_time) >= state.debounce_time_video:
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                output_filename = f"videos/blink_detected_{timestamp}.avi"
                # Consider optimizing save_video_sync or using a faster codec
                save_video_sync(state.frame_buffer, state.fps, output_filename)
                state.last_video_time = current_time

        # Handle Alerts
        with state.alert_lock:
            if (current_time - state.last_alert_time) >= state.debounce_time_alert:
                cv2.putText(frame, "DROWSINESS DETECTED!", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                # Consider removing prints or using a faster logging method
                print(f"Drowsiness Alert: {', '.join(reasons)}")

                # Launch sound playback in a separate thread
                # threading.Thread(target=play_alert_sound, daemon=True).start()
                state.last_alert_time = current_time
                log_data(state, drowsy, reasons)

                threading.Thread(
                    target=send_drowsiness_alert,
                    args=(WS_URL, RECONNECT_INTERVAL),
                    daemon=True
                ).start()

# def play_alert_sound():
#     try:
#         playsound('alert.wav')
#     except Exception as e:
#         # Print errors conditionally or log them asynchronously to reduce blocking
#         print(f"Error playing sound: {e}")


def send_drowsiness_alert(ws_url: str, reconnect_interval: int) -> None:
    """
    Opens a WebSocket connection and sends a JSON message for drowsiness.
    This function is run in a separate thread whenever drowsiness is detected.
    """
    async def async_send():
        ws_client = await initialize_websocket(ws_url, reconnect_interval)
        if ws_client.websocket is not None:
            try:
                message = {
                    "type": "alert",
                    "data": "drowsiness detected!",
                    "event": "drowsiness"
                }
                # Convert Python dict to JSON string
                await ws_client.websocket.send(json.dumps(message))
            finally:
                # Close the connection after sending
                await ws_client.websocket.close()

    # Run the async portion in a blocking fashion within this thread
    asyncio.run(async_send())

# import datetime
# import threading
# import time
#
# import cv2
# from playsound import playsound
#
# from Production.detector.logger import log_data
# from Production.detector.util import save_video_sync, calculate_EAR
#
# def process_bounding_box(face, frame, face_size):
#     x1, y1, x2, y2, score = face
#     x1 = max(0, int(x1))
#     y1 = max(0, int(y1))
#     x2 = min(frame.shape[1], int(x2))
#     y2 = min(frame.shape[0], int(y2) - 10)
#     adjusted_x2 = x2 - face_size
#     adjusted_y2 = y2 - face_size
#     return x1, y1, adjusted_x2, adjusted_y2, x2, y2, score
#
#
#
# def run_landmark_inference(hailo_inference, preprocessed_face, face_land_output_name, class_num) -> list[int] | None :
#     landmarks = hailo_inference.run(model_id=2, input_data=preprocessed_face)
#     landmarks_batch = landmarks.get(face_land_output_name, None)
#     if landmarks_batch is not None:
#         return landmarks_batch[0].reshape(class_num, 2)
#     return None
#
#
# def handle_blink_detection(left_eye, right_eye, state, ear_threshold, consec_frames):
#     left_ear = calculate_EAR(left_eye)
#     right_ear = calculate_EAR(right_eye)
#     avg_ear = (left_ear + right_ear) / 2.0
#     state.current_EAR = avg_ear
#
#     if avg_ear < ear_threshold:
#         state.EAR_consec_frames += 1
#         if not state.is_blinking:
#             state.is_blinking = True
#             state.current_blink_start = time.time()
#     else:
#         if state.EAR_consec_frames >= consec_frames:
#             state.blink_counter += 1
#             state.total_blinks += 1
#             state.blink_timestamps.append(time.time())
#             if state.is_blinking:
#                 blink_duration = time.time() - state.current_blink_start
#                 state.blink_durations.append(blink_duration)
#                 print(f"Blink {state.total_blinks}: Duration = {blink_duration:.3f} seconds")
#         state.EAR_consec_frames = 0
#         state.is_blinking = False
#
#     return avg_ear
#
#
# def handle_drowsiness_detection(avg_EAR, state, frame):
#     drowsy, reasons = state.is_drowsy(avg_EAR, state.current_blink_start if state.is_blinking else 0)
#     if drowsy:
#         current_time = time.time()
#         # Handle Video Saving
#         with state.video_lock:
#             if current_time - state.last_video_time >= state.debounce_time_video:
#                 timestamp = datetime.datetime.now()
#                 output_filename = f"videos/blink_detected_{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}.avi"
#                 save_video_sync(state.frame_buffer, state.fps, output_filename)
#                 state.last_video_time = current_time
#
#         # Handle Alerts
#         with state.alert_lock:
#             if current_time - state.last_alert_time >= state.debounce_time_alert:
#                 cv2.putText(frame, "DROWSINESS DETECTED!", (10, 100),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
#                 print(f"Drowsiness Alert: {', '.join(reasons)}")
#                 threading.Thread(target=play_alert_sound, daemon=True).start()
#                 state.last_alert_time = current_time
#                 log_data(state, drowsy, reasons)
#
#
# # Function to play alert sound
# def play_alert_sound():
#     try:
#         playsound('alert.wav')  # Ensure you have an 'alert.wav' file in your project directory
#     except Exception as e:
#         print(f"Error playing sound: {e}")