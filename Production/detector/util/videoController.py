"""
Video Saving Utilities for Blink Detection and Drowsiness Monitoring

This module provides functions for saving video frames to a file, ensuring directory existence,
and handling video saves.

Functions:
    - save_video_sync: Saves a video synchronously from a buffer of frames.
    - save_video: Asynchronously saves a video using an executor.

Dependencies:
    - cv2: For OpenCV video handling.
    - asyncio: For asynchronous tasks.

Usage:
    Use these functions to handle video saving during drowsiness detection in real-time applications.

Author:
    Lior Jigalo

License:
    MIT
"""

import cv2
import asyncio

# Save video function (synchronous version)
def save_video_sync(buffer, fps, output_path='output_blink_detected.avi') -> None:
    """
    Saves video frames from a buffer synchronously to a file.

    Args:
        buffer (deque): A buffer of video frames (each frame is a NumPy array).
        fps (float): Frames per second for the saved video.
        output_path (str): Path to save the video file (default: 'output_blink_detected.avi').

    Returns:
        None

    Side Effects:
        - Creates a video file at the specified output path.

    Example:
        >>> save_video_sync(buffer, 30, "output.avi")
    """
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
async def save_video(buffer, fps, output_path='output_blink_detected.avi') -> None:
    """
    Saves video frames from a buffer asynchronously to a file.

    Args:
        buffer (deque): A buffer of video frames (each frame is a NumPy array).
        fps (float): Frames per second for the saved video.
        output_path (str): Path to save the video file (default: 'output_blink_detected.avi').

    Returns:
        None

    Side Effects:
        - Creates a video file at the specified output path asynchronously.
    """
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, save_video_sync, buffer, fps, output_path)
