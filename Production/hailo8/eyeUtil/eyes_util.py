import numpy as np
import cv2




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














# Constants for blink detection
EAR_THRESHOLD = 0.25  # EAR below this value indicates a blink
CONSEC_FRAMES = 1     # Number of consecutive frames to count as a blink

# Variables for blink tracking
blink_counter = 0
blink_durations = []
current_blink_start = None


def calculate_ear(eye_landmarks):
    """
    Calculates the Eye Aspect Ratio (EAR) for a single eye.

    Args:
        eye_landmarks: A numpy array of shape (6, 2), representing the eye landmarks.
                       Points 1-6 correspond to the eye contour as shown below:
                       - [p1 (left corner), p2 (top left), p3 (top right),
                          p4 (right corner), p5 (bottom right), p6 (bottom left)]

    Returns:
        The Eye Aspect Ratio (EAR).
    """
    # Vertical distances
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])  # p2 - p6
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])  # p3 - p5

    # Horizontal distance
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])  # p1 - p4

    if C == 0:  # Prevent division by zero
        return 0.0

    # EAR calculation
    return (A + B) / (2.0 * C)

def calculate_blink_rate(blink_durations, frame_count, fps):
    """
    Calculates the blink rate in blinks per minute.

    Args:
        blink_durations: List of blink durations.
        frame_count: Current frame count.
        fps: Frames per second.

    Returns:
        Blink rate in blinks per minute.
    """
    if not blink_durations:
        return 0

    # Convert frame count to time in seconds
    elapsed_time = frame_count / fps
    return (len(blink_durations) / elapsed_time) * 60  # Blinks per minute


def calibrate_ear(ear_list):
    """
    Calibrates the EAR threshold dynamically based on open-eye EAR values.

    Args:
        ear_list: List of EAR values when the eyes are open.

    Returns:
        Dynamic EAR threshold.
    """
    if not ear_list:
        return EAR_THRESHOLD  # Fallback to static threshold

    # Set threshold as 80% of the average open-eye EAR
    return max(np.mean(ear_list), 0.20)  # Ensure a minimum threshold
    # return max(0.8 * np.mean(ear_list), 0.20)  # Ensure a minimum threshold


def handle_eye_aspect_ratio(frame, landmarks, frame_count, fps, blink_counter, current_blink_start, blink_durations):
    """
    Processes eye landmarks, calculates EAR, detects blinks, and provides feedback.

    Args:
        frame: The video frame.
        landmarks: Detected facial landmarks as a numpy array.
        frame_count: Current frame count.
        fps: Frames per second of the video feed.
        blink_counter: Counter for consecutive frames below the EAR threshold.
        current_blink_start: Frame count at the start of a blink.
        blink_durations: List of blink durations.

    Returns:
        Updated frame with EAR, blink feedback, and updated tracking variables.
    """
    # Left and right eye landmarks
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]

    # Calculate EAR for both eyes
    left_ear = calculate_ear(left_eye)
    right_ear = calculate_ear(right_eye)
    avg_ear = (left_ear + right_ear) / 2.0

    # Detect blinks
    blink_counter, current_blink_start, blink_durations, blink_detected = detect_blinks(
        avg_ear, frame_count, current_blink_start, blink_counter, blink_durations
    )

    # Display EAR on the frame
    cv2.putText(frame, f'EAR: {avg_ear:.2f}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if avg_ear >= EAR_THRESHOLD else (0, 0, 255), 2)

    # Display blink feedback
    # if blink_detected:
    #     cv2.putText(frame, "Blink Detected!", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.putText(frame, f"Blink count: {len(blink_durations)}", (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 230, 0), 2)

    # Display average blink duration
    avg_blink_duration = calculate_average_blink_duration(blink_durations, fps)
    cv2.putText(frame, f'Avg Blink Duration: {avg_blink_duration:.2f} sec', (10, 440),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)



    return frame, blink_counter, current_blink_start, blink_durations


def detect_blinks(ear, frame_count, current_blink_start, blink_counter, blink_durations):
    """
    Detects blinks based on EAR and calculates blink durations.

    Args:
        ear: Current Eye Aspect Ratio.
        frame_count: Current frame number.
        current_blink_start: Frame count at the start of a blink.
        blink_counter: Counter for consecutive frames below the EAR threshold.
        blink_durations: List to store durations of each blink.

    Returns:
        updated_blink_counter: Updated blink counter.
        updated_current_blink_start: Updated blink start frame.
        blink_durations: Updated list of blink durations.
        blink_detected: Whether a blink was detected in this frame.
    """
    blink_detected = False

    if ear < EAR_THRESHOLD:
        blink_counter += 1
        if blink_counter == CONSEC_FRAMES:
            # Blink detected
            current_blink_start = frame_count
    else:
        if blink_counter >= CONSEC_FRAMES and current_blink_start is not None:
            # Blink ended; calculate duration
            blink_duration = frame_count - current_blink_start
            blink_durations.append(blink_duration)
            blink_detected = True
        blink_counter = 0  # Reset counter

    return blink_counter, current_blink_start, blink_durations, blink_detected


def calculate_average_blink_duration(blink_durations, fps):
    """
    Calculates the average blink duration in seconds.

    Args:
        blink_durations: List of blink durations in frames.
        fps: Frames per second of the video feed.

    Returns:
        Average blink duration in seconds.
    """
    if not blink_durations:
        return 0.0
    return sum(blink_durations) / len(blink_durations) / fps



