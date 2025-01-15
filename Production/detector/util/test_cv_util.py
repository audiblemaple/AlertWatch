import pytest
import sys
from unittest.mock import patch, MagicMock
import cv2

# Adjust the import if necessary:
# from your_module.camera_utils import init_cv_cap
from . import init_cv_cap


@pytest.mark.parametrize(
    "platform_str, videopath, expected_args",
    [
        # (sys.platform, videopath, expected VideoCapture() call arguments)
        ("win32", None, (0, cv2.CAP_DSHOW)),          # Windows with live camera
        # ("linux", None, ("/dev/video0", cv2.CAP_V4L2)),# Linux with live camera
        # ("darwin", None, ("/dev/video0", cv2.CAP_V4L2)),# macOS (assumed to behave like Linux)
        # ("win32", "video.mp4", ("video.mp4",)),       # Windows with a video file
        # ("linux", "video.mp4", ("video.mp4",)),       # Linux with a video file
    ],
)
def test_init_cv_cap(platform_str, videopath, expected_args):
    """
    Parametrized test to check init_cv_cap on different platforms
    and with/without videopath.
    """
    mock_capture_instance = MagicMock(spec=cv2.VideoCapture)

    # Patch sys.platform to simulate different OS
    with patch.object(sys, "platform", platform_str):
        # Patch cv2.VideoCapture so it doesn't actually try to open a device or file
        with patch("cv2.VideoCapture", return_value=mock_capture_instance) as mock_capture:
            cap = init_cv_cap(width=1280, height=720, fps=60, videopath=videopath)

            # Check that VideoCapture was called with the expected arguments
            mock_capture.assert_called_once_with(*expected_args)

            # Check that the properties are set as intended
            mock_capture_instance.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            mock_capture_instance.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            # YUYV fourcc
            fourcc = cv2.VideoWriter_fourcc(*'YUYV')
            mock_capture_instance.set.assert_any_call(cv2.CAP_PROP_FOURCC, fourcc)

            mock_capture_instance.set.assert_any_call(cv2.CAP_PROP_FPS, 60)

            # Finally, verify we returned the same object
            assert cap == mock_capture_instance


def test_init_cv_cap_defaults():
    """
    Test that init_cv_cap() sets default width, height, and fps when no arguments are passed.
    This also confirms videopath=None on a non-Windows platform (assume Linux here).
    """
    mock_capture_instance = MagicMock(spec=cv2.VideoCapture)

    with patch.object(sys, "platform", "linux"):
        with patch("cv2.VideoCapture", return_value=mock_capture_instance) as mock_capture:
            cap = init_cv_cap()  # Use default args: width=640, height=640, fps=30, videopath=None

            # On Linux with no videopath, we expect:
            mock_capture.assert_called_once_with("/dev/video0", cv2.CAP_V4L2)

            # Ensure default properties are set
            mock_capture_instance.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 640)
            mock_capture_instance.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, 640)
            mock_capture_instance.set.assert_any_call(cv2.CAP_PROP_FPS, 30)

            fourcc = cv2.VideoWriter_fourcc(*'YUYV')
            mock_capture_instance.set.assert_any_call(cv2.CAP_PROP_FOURCC, fourcc)

            assert cap == mock_capture_instance
