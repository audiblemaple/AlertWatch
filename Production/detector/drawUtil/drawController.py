import cv2
import datetime
import numpy as np


def draw_bounding_box(frame, score, p1, p2, color=(0, 255, 0), size=2):
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
    cv2.putText(frame, text, (10, offset), cv2.FONT_HERSHEY_SIMPLEX, size, color, 2)

def draw_landmarks(frame, all_landmarks):
    """
    Draws detected landmarks on the frame.
    """
    for landmarks in all_landmarks:
        for (x, y) in landmarks:
            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)

def draw_head_pose(frame, rotation_vector, translation_vector, camera_matrix):
    """
    Visualizes head pose using axes drawn on the frame.
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

def display_fps(frame, fps, avg_fps):
    """
    Display current and average FPS on the frame.

    Args:
        frame (np.ndarray): The video frame to annotate.
        fps (float): Current FPS.
        avg_fps (float): Average FPS.
    """
    timestamp = datetime.datetime.now()
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Avg FPS: {avg_fps:.2f}", (200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"timestamp: {timestamp.strftime('%Y-%m-%d_%H-%M-%S')}", (10, 400),
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
    cv2.putText(frame, f"Blinks: {blink_count}", (10, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Total Blinks: {total_blinks}", (10, 330),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    if blink_durations:
        average_duration = sum(blink_durations) / len(blink_durations)
        cv2.putText(frame, f"Avg Blink Dur: {average_duration:.2f}s", (250, 330),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)