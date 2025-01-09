"""
Camera Initialization Utility

This module provides a utility function for initializing a video capture object using OpenCV.

Functions:
    - init_cv_cap: Initializes a video capture object for live video input or video file playback.

Dependencies:
    - sys: For platform-specific checks.
    - cv2: OpenCV library for video capture and manipulation.

Usage:
    Use the `init_cv_cap` function to create a video capture object for accessing live video from a camera
    or reading from a video file.

Author:
    Lior Jigalo

License:
    MIT
"""

import sys
import cv2

def init_cv_cap(width=640, height=640, fps=30, videopath=None):
    """
    Initializes an OpenCV video capture object with specified properties.

    Args:
        width (int): Desired width of the video frame (default: 640).
        height (int): Desired height of the video frame (default: 640).
        fps (int): Desired frames per second for the video capture (default: 30).
        videopath (str | None): Path to a video file for playback. If None, uses a live video source.

    Returns:
        cv2.VideoCapture: An OpenCV video capture object configured with the specified properties.

    Notes:
        - On Windows, the `cv2.CAP_DSHOW` backend is used for capturing live video.
        - On non-Windows platforms, `/dev/video0` with `cv2.CAP_V4L2` is used as the default video source.
        - The video capture is configured with the YUYV pixel format.

    Example:
        To initialize a camera capture:
        >>> cap = init_cv_cap(1280, 720, 60)

        To initialize playback from a video file:
        >>> cap = init_cv_cap(videopath="example.mp4")
    """
    if videopath is None:
        if 'win' in sys.platform:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(videopath)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    fourcc = cv2.VideoWriter_fourcc(*'YUYV')
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap

