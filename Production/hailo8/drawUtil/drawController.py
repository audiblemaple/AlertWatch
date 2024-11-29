import cv2
import numpy as np
def draw_bounding_box(frame, score, p1, p2, color=(0, 255, 0), size=2):
    cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
    cv2.putText(
        frame,
        f"Detection confidence: {score:.2f}%",
        (p1[0], p1[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
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


def display_fps(frame, fps):
    """
    Displays the frames per second (FPS) on the frame.
    """
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
