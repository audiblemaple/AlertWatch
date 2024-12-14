import time

def estimate_fps(cap, warmup_frames=30):
    """
    Estimates the FPS of the video capture device.

    Args:
        cap (cv2.VideoCapture): The video capture object.
        warmup_frames (int): Number of frames to capture for estimation.

    Returns:
        float: Estimated FPS.
    """
    start_time = time.time()
    for _ in range(warmup_frames):
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image during warm-up.")
            break
    end_time = time.time()
    elapsed = end_time - start_time
    estimated_fps = warmup_frames / elapsed if elapsed > 0 else 30
    return estimated_fps