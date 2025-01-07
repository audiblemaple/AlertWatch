import os
import cv2
import time
import asyncio
from .AppState import AppState

# Save video function (synchronous version)
def save_video_sync(buffer, fps, output_path='output_blink_detected.avi'):
    print(f"Attempting to save video to {output_path} with {len(buffer)} frames at {fps} FPS")
    if not buffer:
        print("No frames to save.")
        return

    try:
        # Get frame dimensions from the first frame
        height, width, channels = buffer[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Try 'MJPG' or 'mp4v' if 'XVID' doesn't work
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print("VideoWriter initialized.")

        for frame in buffer:
            out.write(frame)

        out.release()
        print(f"Saved video to {output_path}")
    except Exception as e:
        print(f"Error saving video: {e}")


# Asynchronous save_video using executor
async def save_video(buffer, fps, output_path='output_blink_detected.avi'):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, save_video_sync, buffer, fps, output_path)


# Utility function to ensure directory exists
def ensure_directory_exists(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


# Delayed video save coroutine
async def delayed_save_video(state: AppState, output_path: str, delay: float = 5.0):
    """
    Waits for a specified delay and saves the video if drowsiness is still detected.

    Args:
        state (AppState): The application state.
        output_path (str): The path where the video will be saved.
        delay (float): Delay in seconds before saving the video.
    """
    print(f"Scheduling video save in {delay} seconds...")
    await asyncio.sleep(delay)  # Wait for the specified delay

    # Acquire video lock to ensure thread safety
    with state.video_lock:
        # Check if drowsiness is still active
        drowsy, _ = state.is_drowsy(state.current_EAR, state.current_blink_start if state.is_blinking else 0)
        if drowsy:
            print("Drowsiness confirmed after delay. Saving video...")
            await save_video(state.frame_buffer, state.fps, output_path=output_path)
            # Clear the buffer and reset blink counter
            state.frame_buffer.clear()
            state.blink_counter = 0
            print(f"Blinks detected: {state.total_blinks}. Buffer cleared after saving video.")
            # Update the last_video_time
            state.last_video_time = time.time()
        else:
            print("Drowsiness no longer detected after delay. Video save aborted.")
        # Reset the pending flag
        state.video_save_pending = False
        state.video_save_task = None
