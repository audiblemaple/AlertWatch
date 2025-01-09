"""
Eye Calculations

This module provides a utility function to calculate the Eye Aspect Ratio (EAR),
commonly used in blink detection and drowsiness monitoring applications.

Functions:
    - calculate_EAR: Computes the Eye Aspect Ratio from eye landmarks.

Dependencies:
    - NumPy: For numerical computations and Euclidean distance calculations.

Usage:
    Use the `calculate_EAR` function to compute the EAR from an array of eye landmarks.

Author:
    Lior Jigalo

License:
    MIT
"""
import numpy as np

def calculate_EAR(eye) -> float:
    """
    Calculates the Eye Aspect Ratio (EAR) for blink detection.

    The EAR is a scalar value that indicates the openness of an eye, computed as the ratio of
    the sum of the distances between vertical eye landmarks to twice the horizontal distance.
    Lower EAR values typically correspond to closed eyes, while higher values indicate open eyes.

    Args:
        eye (np.ndarray): A 2D array of shape (6, 2) representing the (x, y) coordinates
                          of the six eye landmarks:
                          - eye[0]: Left corner
                          - eye[1]: Top-left
                          - eye[2]: Top-right
                          - eye[3]: Right corner
                          - eye[4]: Bottom-right
                          - eye[5]: Bottom-left

    Returns:
        float: The computed EAR value.

    Notes:
        - The EAR is dimensionless and invariant to the scale of the eye image.
        - It is a widely used metric for detecting blinks and monitoring eye openness.
    """
    # Compute the euclidean distances between the vertical eye landmarks
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])

    # Compute the euclidean distance between the horizontal eye landmarks
    C = np.linalg.norm(eye[0] - eye[3])

    # Compute EAR
    ear = (A + B) / (2.0 * C)
    return ear