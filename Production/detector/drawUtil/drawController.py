"""
Drawing Utilities for Video Frames

This module provides utility functions for annotating video frames with various information,
including bounding boxes, text, landmarks, head pose, and performance metrics like FPS.

Functions:
    - draw_bounding_box: Draws a bounding box with a detection score on the frame.
    - draw_text: Displays text at a specific location on the frame.
    - draw_landmarks: Draws detected landmarks on the frame.
    - draw_head_pose: Visualizes head pose using 3D axes on the frame. (unused)
    - display_fps: Displays current and average FPS on the frame.
    - display_blink_info: Displays blink-related information on the frame.

Dependencies:
    - OpenCV (cv2): For image processing and drawing operations.
    - NumPy: For numerical computations and matrix manipulations.
    - datetime: For timestamping frames.

Usage:
    Import the module and use the functions to annotate video frames during real-time video processing.

Author:
Lior Jigalo

License:
    MIT
"""

import cv2
import datetime
import numpy as np
from cv2 import Mat


def draw_bounding_box(frame, score, p1, p2, color=(0, 255, 0), size=2) -> None:
    """
    Draws a bounding box with the detection confidence score.

    Args:
        frame (np.ndarray): The video frame to annotate.
        score (float): Confidence score of the detection.
        p1 (tuple): Top-left corner of the bounding box (x, y).
        p2 (tuple): Bottom-right corner of the bounding box (x, y).
        color (tuple): Color of the bounding box in BGR format (default: green).
        size (int): Thickness of the bounding box (default: 2).
    """
    cv2.rectangle(frame, p1, p2, color, 2)
    cv2.putText(
        frame,
        f"Detection confidence: {score:.2f}%",
        (p1[0], p1[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1
    )

def draw_text(cv2, frame, offset, text, color, size=0.6):
    """
    Draws text on the frame at a specific vertical offset.

    Args:
        frame (np.ndarray): The video frame to annotate.
        offset (int): Vertical offset from the top of the frame.
        text (str): The text to display.
        color (tuple): Color of the text in BGR format.
        size (float): Font scale of the text (default: 0.6).
    """
    cv2.putText(frame, text, (10, offset), cv2.FONT_HERSHEY_SIMPLEX, size, color, 2)


def draw_landmarks(frame, all_landmarks) -> None:
    """
    Draws detected landmarks as points on the frame.

    Args:
        frame (np.ndarray): The video frame to annotate.
        all_landmarks (list of list of tuple): List of detected landmarks, where each landmark
            is represented as a list of (x, y) coordinates.
    """
    for landmarks in all_landmarks:
        for (x, y) in landmarks:
            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)

def draw_head_pose(frame, rotation_vector, translation_vector, camera_matrix) -> Mat | np.ndarray:
    """
    Visualizes head pose using 3D axes projected onto the frame.

    Args:
        frame (np.ndarray): The video frame to annotate.
        rotation_vector (np.ndarray): Rotation vector of the head pose.
        translation_vector (np.ndarray): Translation vector of the head pose.
        camera_matrix (np.ndarray): Camera matrix for the projection.

    Returns:
        np.ndarray: The annotated video frame.
    """
    axis = np.float32([[200, 0, 0], [0, 200, 0], [0, 0, 200]])

    imgpts, _ = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, np.zeros((4, 1)))

    nose_tip, _ = cv2.projectPoints(np.array([(0.0, 0.0, 0.0)]), rotation_vector, translation_vector, camera_matrix, np.zeros((4, 1)))

    nose_tip = tuple(nose_tip.ravel().astype(int))
    imgpts = imgpts.astype(int)

    frame = cv2.line(frame, nose_tip, tuple(imgpts[0].ravel()), (0, 0, 255), 2)  # Red axis (X)
    frame = cv2.line(frame, nose_tip, tuple(imgpts[1].ravel()), (0, 255, 0), 2)  # Green axis (Y)
    frame = cv2.line(frame, nose_tip, tuple(imgpts[2].ravel()), (255, 0, 0), 2)  # Blue axis (Z)
    return frame

def display_fps(frame, fps, avg_fps) -> None:
    """
    Displays current and average FPS on the frame.

    Args:
        frame (np.ndarray): The video frame to annotate.
        fps (float): Current frames per second.
        avg_fps (float): Average frames per second.
    """
    timestamp = datetime.datetime.now()
    # cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    # cv2.putText(frame, f"Avg FPS: {avg_fps:.2f}", (220, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    # cv2.putText(frame, f"timestamp: {timestamp.strftime('%Y-%m-%d_%H-%M-%S')}", (10, 470),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 250, 100), 2, cv2.LINE_AA)


    cv2.putText(frame, f"FPS / AVG FPS: {fps:.2f} / {avg_fps:.2f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Timestamp: {timestamp.strftime('%H:%M:%S-%d.%m.%Y')}", (10, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)


def display_blink_info(frame, blink_count, total_blinks, blink_durations) -> None:
    """
    Displays blink count and average blink duration on the frame.

    Args:
        frame (np.ndarray): The video frame to annotate.
        blink_count (int): Number of blinks detected in the current session.
        total_blinks (int): Total number of blinks detected.
        blink_durations (list of float): List of blink durations in seconds.
    """
    cv2.putText(frame, f"Blinks: {blink_count}", (10, 400),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Total Blinks: {total_blinks}", (10, 440),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    if blink_durations:
        average_duration = sum(blink_durations) / len(blink_durations)
        cv2.putText(frame, f"Avg Blink Dur: {average_duration:.2f}s", (280, 440),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)