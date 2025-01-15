import pytest
import numpy as np
import time
import threading
from unittest.mock import patch, MagicMock

from .faceController import (
    process_bounding_box,
    run_landmark_inference,
    handle_blink_detection,
    handle_drowsiness_detection,
    send_drowsiness_alert,
    RECONNECT_INTERVAL,
)

WS_URL: str = "ws://192.168.0.63:5000"


class MockAppState:
    def __init__(self):
        self.current_EAR = 0.0
        self.EAR_consec_frames = 0
        self.is_blinking = False
        self.current_blink_start = 0.0
        self.blink_counter = 0
        self.total_blinks = 0
        self.blink_timestamps = []
        self.blink_durations = []
        self.frame_buffer = []
        self.fps = 30.0
        self.last_video_time = 0.0
        self.debounce_time_video = 30.0
        self.video_lock = threading.Lock()
        self.last_alert_time = 0.0
        self.debounce_time_alert = 4.0
        self.alert_lock = threading.Lock()
        # For testing handle_drowsiness_detection:
        self.is_drowsy = MagicMock(return_value=(False, ""))


@pytest.fixture
def dummy_frame():
    """Creates a simple dummy frame for bounding box tests."""
    # 480x640 with 3 channels
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_process_bounding_box(dummy_frame):
    """
    Tests that process_bounding_box clamps coordinates within frame boundaries.
    """
    face = (-10, -20, 700, 500, 0.9)  # Some out-of-bounds coordinates
    x1, y1, x2, y2, score = process_bounding_box(face, dummy_frame)
    # The final coords should be clamped to the frame size (width=640, height=480)
    assert x1 == 0
    assert y1 == 0
    assert x2 == 640
    #
    # The code in faceController does:
    #   y2 = min(h, int(y2) - 10)
    # For an original y2=500 -> 500 - 10 = 490, min(480, 490) = 480
    #
    assert y2 == 480
    assert score == 0.9


def test_run_landmark_inference_success():
    """
    Tests run_landmark_inference returning valid landmarks from mock hailo_inference.
    """
    hailo_inference = MagicMock()
    preprocessed_face = np.zeros((1, 3, 64, 64), dtype=np.float32)
    face_land_output_name = "landmarks"
    class_num = 68

    # Mock the inference to return a batch of shape [1, class_num*2]
    expected_landmarks = np.random.rand(1, class_num * 2).astype(np.float32)
    hailo_inference.run.return_value = {face_land_output_name: expected_landmarks}

    output = run_landmark_inference(hailo_inference, preprocessed_face, face_land_output_name, class_num)
    # Should reshape to (class_num, 2)
    assert output.shape == (class_num, 2)


def test_run_landmark_inference_none():
    """
    Tests run_landmark_inference returning None if output tensor is not found.
    """
    hailo_inference = MagicMock()
    preprocessed_face = np.zeros((1, 3, 64, 64), dtype=np.float32)
    face_land_output_name = "landmarks"
    class_num = 68

    # Mock the inference to return a dict without 'landmarks'
    hailo_inference.run.return_value = {}

    output = run_landmark_inference(hailo_inference, preprocessed_face, face_land_output_name, class_num)
    assert output is None


def test_handle_blink_detection_below_threshold():
    """
    Tests that EAR below threshold increments EAR_consec_frames
    and sets is_blinking to True on first detection.
    """
    from math import isclose
    state = MockAppState()
    left_eye = np.random.rand(6, 2)
    right_eye = np.random.rand(6, 2)

    # Patch calculate_EAR so we can force a specific EAR
    with patch("util.calculate_EAR", side_effect=[0.2, 0.2]):
        avg_ear = handle_blink_detection(left_eye, right_eye, state, ear_threshold=0.25, consec_frames=3)

    # avg_ear should be 0.2
    assert isclose(avg_ear, 0.2)
    assert state.EAR_consec_frames == 1
    assert state.is_blinking is True
    assert state.current_blink_start > 0.0  # Marked blink start time


def test_handle_blink_detection_above_threshold():
    """
    Tests that EAR above threshold resets EAR_consec_frames and
    registers a blink if it was below threshold for enough consecutive frames.
    """
    from math import isclose
    state = MockAppState()
    # Pretend we were blinking for 3 frames
    state.EAR_consec_frames = 3
    state.is_blinking = True
    state.current_blink_start = time.time() - 0.5  # blink lasted 0.5s

    left_eye = np.random.rand(6, 2)
    right_eye = np.random.rand(6, 2)

    # Force average EAR of 0.3
    with patch("util.calculate_EAR", side_effect=[0.3, 0.3]):
        avg_ear = handle_blink_detection(left_eye, right_eye, state, ear_threshold=0.25, consec_frames=3)

    # Should detect a blink
    assert isclose(avg_ear, 0.3)
    assert state.blink_counter == 1
    assert state.total_blinks == 1
    # We expect 1 timestamp in blink_timestamps
    assert len(state.blink_timestamps) == 1
    # We expect the blink duration in blink_durations
    assert len(state.blink_durations) == 1
    assert state.EAR_consec_frames == 0
    assert state.is_blinking is False


def test_handle_drowsiness_detection_not_drowsy():
    """
    If is_drowsy() returns (False, ""), handle_drowsiness_detection
    should do nothing (no video saved, no alert).
    """
    state = MockAppState()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    handle_drowsiness_detection(avg_EAR=0.3, state=state, frame=frame)

    # No threads for video or alert should be started
    # The last_video_time and last_alert_time remain 0
    assert state.last_video_time == 0.0
    assert state.last_alert_time == 0.0


def test_handle_drowsiness_detection_drowsy():
    """
    If is_drowsy() returns True, handle_drowsiness_detection
    should attempt to start a video-saving thread and an alert-sending thread.
    """
    state = MockAppState()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Force is_drowsy to return True
    state.is_drowsy.return_value = (True, "low_average_ear")

    # We'll patch out the video saving and the WebSocket send call
    with patch("util.save_video_sync") as mock_save_video:
        with patch("util.faceController.send_drowsiness_alert") as mock_send_alert:
            handle_drowsiness_detection(avg_EAR=0.25, state=state, frame=frame)

    # If drowsy, we expect we set last_video_time and last_alert_time
    assert state.last_video_time != 0.0
    assert state.last_alert_time != 0.0
    # Additional checks (mock_save_video or mock_send_alert call_count) are possible,
    # but they might be running in threads, so you might need to join them or wait a bit.


@pytest.mark.asyncio
async def test_send_drowsiness_alert():
    """
    Tests that send_drowsiness_alert calls initialize_websocket, then attempts to send a JSON alert.
    We'll mock out initialize_websocket and the ws_client.websocket.
    """
    mock_ws_client = MagicMock()
    mock_ws_client.websocket = MagicMock()

    with patch("util.faceController.initialize_websocket", return_value=mock_ws_client):
        # We also patch asyncio.run, though you might not need to if your code calls it directly
        with patch("asyncio.run") as mock_run:
            send_drowsiness_alert(WS_URL, RECONNECT_INTERVAL, "high_blink_rate")

            # ensure asyncio.run was called with a coroutine
            assert mock_run.call_count == 1
