import numpy as np

def calculate_EAR(eye):
    """
    Calculates the Eye Aspect Ratio (EAR) for blink detection.

    Args:
        eye (np.ndarray): Array of eye landmarks.

    Returns:
        float: The EAR value.
    """
    # Compute the euclidean distances between the vertical eye landmarks
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])

    # Compute the euclidean distance between the horizontal eye landmarks
    C = np.linalg.norm(eye[0] - eye[3])

    # Compute EAR
    ear = (A + B) / (2.0 * C)
    return ear