import time
import pytest
from unittest.mock import patch, MagicMock
from collections import deque

from . import AppState

EAR_THRESHOLD = 0.25

@pytest.fixture
def app_state():
    """
    Fixture to create a fresh AppState instance for each test.
    """
    return AppState()


def test_default_initialization(app_state):
    """
    Test default initialization of the AppState class.
    """
    assert app_state.blink_counter == 0
    assert app_state.total_blinks == 0
    assert app_state.EAR_consec_frames == 0
    # assert isinstance(app_state.frame_buffer, deque)
    assert app_state.buffer_size == 300
    assert app_state.fps == 0.0
    assert app_state.blink_durations == []
    assert app_state.is_blinking is False
    assert app_state.current_blink_start == 0.0

    assert app_state.analysis_window == 20
    assert app_state.blink_rate_threshold == 90.0
    assert app_state.prolonged_EAR_duration_threshold == 1

    assert isinstance(app_state.blink_timestamps, deque)

    assert app_state.last_alert_time == 0.0
    assert app_state.debounce_time_alert == 4
    # assert app_state.last_video_time == 0.0
    # assert app_state.debounce_time_video == 30

    # Check locks
    assert app_state.alert_lock.locked() is False
    # assert app_state.video_lock.locked() is False

    # EAR measurements
    assert isinstance(app_state.ear_measurements, deque)
    assert app_state.ear_sum == 0.0

    assert app_state.last_ear_reset_time == 0.0
    assert app_state.ear_reset_cooldown == 5.0
    assert app_state.avg_ear_threshold == 0.25


def test_update_blink_rate_empty(app_state):
    """
    Test update_blink_rate() with no recent blinks (empty blink_timestamps).
    Should return 0.0 if no blinks in the analysis window.
    """
    with patch("time.time", return_value=100.0):
        rate = app_state.update_blink_rate()
    assert rate == 0.0
    assert len(app_state.blink_timestamps) == 0


def test_update_blink_rate_with_data(app_state):
    """
    Test update_blink_rate() with some existing blink timestamps.
    Only the timestamps within the analysis window should remain,
    and blink_rate should be computed correctly.
    """
    # Suppose the current time is 100
    now = 100.0
    analysis_window = app_state.analysis_window  # default = 20 seconds
    # We'll inject some timestamps:
    # - 10 timestamps between time=81 and time=99 (within the window)
    # - 2 timestamps at time=60 (outside the window, older than 20 seconds).
    old_time = now - analysis_window - 1  # 79 (outside the window)
    newer_times = list(range(int(now - 19), int(now)))  # 81 to 99 (19 timestamps)
    app_state.blink_timestamps.extend([old_time, old_time] + newer_times)

    with patch("time.time", return_value=now):
        blink_rate = app_state.update_blink_rate()

    # After update, timestamps older than (now - 20) = 80 should be removed
    # so we should have 19 timestamps left.
    assert len(app_state.blink_timestamps) == 19

    # blink_rate = (19 blinks in last 20 seconds) * (60 / 20) = 19 * 3 = 57
    assert blink_rate == 19 * (60.0 / analysis_window)


def test_check_prolonged_EAR_false(app_state):
    """
    Test check_prolonged_EAR() when EAR is above threshold or not enough time has passed.
    Should return False in both cases.
    """
    # 1. EAR above threshold => immediate False
    current_ear = 0.3  # above EAR_THRESHOLD=0.25
    with patch("time.time", return_value=100.0):
        # Suppose we started measuring at 99.0
        assert app_state.check_prolonged_EAR(current_ear, 99.0) is False

    # 2. EAR below threshold but not enough time elapsed => False
    current_ear = 0.2  # below 0.25
    # If we started 0.5 seconds ago, but threshold requires 1 second
    with patch("time.time", return_value=100.5):
        assert app_state.check_prolonged_EAR(current_ear, 100.0) is False


def test_check_prolonged_EAR_true(app_state):
    """
    Test check_prolonged_EAR() returns True if EAR has been below the threshold
    for longer than the prolonged_EAR_duration_threshold.
    """
    current_ear = 0.2  # below threshold
    # If we started at time=98.0, and current time is 100.0 => 2 seconds elapsed
    # default prolonged_EAR_duration_threshold = 1 second
    with patch("time.time", return_value=100.0):
        assert app_state.check_prolonged_EAR(current_ear, 98.0) is True


def test_add_ear_measurement(app_state):
    """
    Test that add_ear_measurement() appends the new measurement,
    updates the running sum, and removes old measurements outside the analysis window.
    """
    now = 100.0
    analysis_window = app_state.analysis_window  # 20 seconds
    # Pre-populate ear_measurements with some old data at time=79 (outside the window).
    old_time = now - analysis_window - 1  # 79
    app_state.ear_measurements.append((old_time, 0.1))
    app_state.ear_sum = 0.1

    with patch("time.time", return_value=now):
        app_state.add_ear_measurement(ear_value=0.3)

    # The old measurement at time=79 should be removed (outside [80..100]).
    assert len(app_state.ear_measurements) == 1
    assert app_state.ear_measurements[0][1] == 0.3
    # ear_sum should be 0.3 now
    assert app_state.ear_sum == pytest.approx(0.3, abs=1e-9)


def test_get_average_ear_empty(app_state):
    """
    Test get_average_ear() returns 0.0 when there are no measurements.
    """
    avg_ear = app_state.get_average_ear()
    assert avg_ear == 0.0


def test_get_average_ear_non_empty(app_state):
    """
    Test get_average_ear() with some EAR measurements.
    Uses the running sum for O(1) average computation.
    """
    # Add a few measurements
    app_state.ear_measurements.append((10, 0.2))
    app_state.ear_measurements.append((11, 0.3))
    app_state.ear_measurements.append((12, 0.4))
    app_state.ear_sum = 0.2 + 0.3 + 0.4  # 0.9

    avg_ear = app_state.get_average_ear()
    # Average should be 0.9 / 3 = 0.3
    assert pytest.approx(avg_ear, 0.001) == 0.3


def test_is_drowsy_high_blink_rate(app_state):
    """
    Test is_drowsy() triggers "high_blink_rate" if blink_rate > blink_rate_threshold.
    """
    # For simplicity, let's just mock update_blink_rate() to return 100.0
    with patch.object(app_state, 'update_blink_rate', return_value=100.0) as mock_rate:
        with patch.object(app_state, 'check_prolonged_EAR', return_value=False):
            with patch.object(app_state, 'get_average_ear', return_value=0.3):
                drowsy, reason = app_state.is_drowsy(current_EAR=0.3, ear_start_time=99.0)

    assert drowsy is True
    assert reason == "high_blink_rate"


def test_is_drowsy_prolonged_eye_closure(app_state):
    """
    Test is_drowsy() triggers "Prolonged_eye_closure" if check_prolonged_EAR() is True.
    """
    with patch.object(app_state, 'update_blink_rate', return_value=10.0):
        with patch.object(app_state, 'check_prolonged_EAR', return_value=True):
            with patch.object(app_state, 'get_average_ear', return_value=0.3):
                drowsy, reason = app_state.is_drowsy(current_EAR=0.2, ear_start_time=95.0)

    assert drowsy is True
    assert reason == "Prolonged_eye_closure"


def test_is_drowsy_low_average_ear(app_state):
    """
    Test is_drowsy() triggers "low_average_ear" if average_ear < 0.29
    and the ear_reset_cooldown has elapsed.
    """
    # Make sure time has advanced enough past last_ear_reset_time.
    app_state.last_ear_reset_time = 90.0
    app_state.ear_reset_cooldown = 5.0

    with patch.object(app_state, 'update_blink_rate', return_value=10.0):
        with patch.object(app_state, 'check_prolonged_EAR', return_value=False):
            # Let's say get_average_ear() returns 0.28, which is < 0.29
            with patch.object(app_state, 'get_average_ear', return_value=0.28):
                with patch("time.time", return_value=100.0):
                    drowsy, reason = app_state.is_drowsy(current_EAR=0.28, ear_start_time=95.0)

    assert drowsy is True
    assert reason == "low_average_ear"
    # The ear_measurements deque should be cleared and ear_sum reset
    assert len(app_state.ear_measurements) == 0
    assert app_state.ear_sum == 0.0
    assert app_state.last_ear_reset_time == 100.0


def test_is_drowsy_no_condition_met(app_state):
    """
    Test is_drowsy() returns (False, "") if no drowsiness condition is met.
    """
    with patch.object(app_state, 'update_blink_rate', return_value=10.0):
        with patch.object(app_state, 'check_prolonged_EAR', return_value=False):
            with patch.object(app_state, 'get_average_ear', return_value=0.31):
                drowsy, reason = app_state.is_drowsy(current_EAR=0.31, ear_start_time=99.0)

    assert drowsy is False
    assert reason == ""
