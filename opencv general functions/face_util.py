import numpy as np
import cv2

def get_3d_model_points():
    """
    Returns the 3D model points for a standard face model.
    """
    return np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye corner
        (225.0, 170.0, -135.0),  # Right eye corner
        (-150.0, -150.0, -125.0),  # Left mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ], dtype=np.float32)


def get_camera_intrinsics(frame_shape):
    """
    Returns the camera intrinsic matrix based on the frame shape.
    """
    focal_length = frame_shape[1]
    center = (frame_shape[1] / 2, frame_shape[0] / 2)
    return np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)


def estimate_head_pose(landmarks, frame_shape):
    """
    Estimates the head pose (rotation and translation vectors) based on landmarks.
    """
    model_points = get_3d_model_points()
    image_points = np.array([
        landmarks[30],  # Nose tip
        landmarks[8],  # Chin
        landmarks[36],  # Left eye corner
        landmarks[45],  # Right eye corner
        landmarks[48],  # Left mouth corner
        landmarks[54]  # Right mouth corner
    ], dtype=np.float32)

    camera_matrix = get_camera_intrinsics(frame_shape)
    dist_coefficients = np.zeros((4, 1))  # Assuming no lens distortion

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coefficients
    )
    return rotation_vector, translation_vector


def estimate_head_pose_with_direction(landmarks, frame_shape):
    """
    Estimates the head pose (rotation and translation vectors) and infers the direction of looking.

    Args:
        landmarks: Facial landmarks.
        frame_shape: Shape of the video frame.

    Returns:
        rotation_vector: Rotation vector.
        translation_vector: Translation vector.
        direction: String representing the direction of looking.
    """
    model_points = get_3d_model_points()
    image_points = np.array([
        landmarks[30],  # Nose tip
        landmarks[8],  # Chin
        landmarks[36],  # Left eye corner
        landmarks[45],  # Right eye corner
        landmarks[48],  # Left mouth corner
        landmarks[54]  # Right mouth corner
    ], dtype=np.float32)

    camera_matrix = get_camera_intrinsics(frame_shape)
    dist_coefficients = np.zeros((4, 1))  # Assuming no lens distortion

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coefficients
    )

    # Convert rotation vector to Euler angles
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    euler_angles = cv2.decomposeProjectionMatrix(np.hstack((rotation_matrix, translation_vector)))[6]

    pitch, yaw, roll = euler_angles.flatten()

    # Determine direction
    direction = get_direction_from_angles(pitch, yaw)

    return rotation_vector, translation_vector, direction


def get_direction_from_angles(pitch, yaw):
    """
    Infers the direction of looking based on pitch and yaw angles.

    Args:
        pitch: Rotation around the X-axis (up-down).
        yaw: Rotation around the Y-axis (left-right).

    Returns:
        A string representing the direction.
    """

    # Reverse pitch and yaw if they are switched
    pitch = -pitch  # Negate to correct axis interpretation
    yaw = yaw * 0.95

    print(f"pitch: {pitch}, yaw: {yaw}")

    if pitch > 160:  # Looking down
        if yaw < -10:
            return "Down Left"
        elif yaw > 10:
            return "Down Right"
        return "Down"
    elif pitch < -160:  # Looking up
        if yaw < -10:
            return "Up Left"
        elif yaw > 10:
            return "Up Right"
        return "Up"
    else:  # Horizontal direction
        if yaw < -10:
            return "Left"
        elif yaw > 10:
            return "Right"
        return "Center"
