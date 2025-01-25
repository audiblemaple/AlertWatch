"""
Application State Management for Drowsiness Detection

This module defines the `AppState` class, which manages the application state
for blink detection, drowsiness analysis, and related metrics. It includes methods
to compute blink rates, evaluate drowsiness, and maintain a buffer of EAR (Eye Aspect Ratio) measurements.

Constants:
    - EAR_THRESHOLD (float): Threshold below which the EAR indicates eye closure.

Classes:
    - AppState: Manages application state for blink detection and drowsiness monitoring.

Dependencies:
    - dataclasses: For defining the `AppState` class with default values and field types.
    - collections.deque: For maintaining rolling buffers of EAR measurements and timestamps.
    - threading: For managing locks to synchronize alerts and video processing.
    - time: For time-based computations such as timestamps and durations.

Usage:
    Create an `AppState` instance and use its methods to track drowsiness-related metrics.

Author:
    Lior Jigalo

License:
    MIT
"""

from dataclasses import dataclass, field
from collections import deque
import threading
import time

EAR_THRESHOLD = 0.20

@dataclass
class AppState:
    """
     Manages the application state for drowsiness detection and blink analysis.

     Attributes:
         blink_counter (int): Count of consecutive EAR frames below the threshold.
         total_blinks (int): Total number of blinks detected.
         EAR_consec_frames (int): Consecutive frames with EAR below the threshold.
         frame_buffer (deque): Buffer for storing recent video frames.
         buffer_size (int): Maximum size of the frame buffer.
         fps (float): Current frames per second.
         blink_durations (list): List of durations for individual blinks.
         is_blinking (bool): Indicates if a blink is currently happening.
         current_blink_start (float): Timestamp for when the current blink started.

         analysis_window (int): Time window (in seconds) for computing blink rates and EAR averages.
         blink_rate_threshold (float): Threshold for high blink rate (blinks per minute).
         prolonged_EAR_duration_threshold (float): Duration threshold for prolonged low EAR.

         blink_timestamps (deque): Timestamps of recent blinks for rate computation.

         last_alert_time (float): Timestamp for the last alert.
         debounce_time_alert (float): Minimum time between consecutive alerts.
         last_video_time (float): Timestamp for the last video notification.
         debounce_time_video (float): Minimum time between video notifications.
         alert_lock (threading.Lock): Lock to synchronize alert generation.
         video_lock (threading.Lock): Lock to synchronize video processing.

         ear_measurements (deque): Rolling buffer of EAR measurements and timestamps.
         ear_sum (float): Running sum of EAR measurements for fast average computation.

         last_ear_reset_time (float): Timestamp for the last EAR measurement reset.
         ear_reset_cooldown (float): Cooldown duration for resetting EAR measurements.
         avg_ear_threshold (float): Threshold for average EAR indicating drowsiness.
     """
    blink_counter: int = 0
    total_blinks: int = 0
    EAR_consec_frames: int = 0
    frame_buffer: deque = field(default_factory=lambda: deque())
    buffer_size: int = 300
    fps: float = 0.0
    blink_durations: list = field(default_factory=list)  # Stores duration of each blink
    is_blinking: bool = False         # Indicates if a blink is ongoing
    current_blink_start: float = 0.0  # Timestamp when the current blink started

    # Analysis Parameters
    analysis_window: int = 20               # seconds (for blink rate + EAR over time)
    blink_rate_threshold: float = 100.0      # blinks per minute
    prolonged_EAR_duration_threshold: float = 1.1  # seconds

    # Timestamps of blinks for rate calculation
    blink_timestamps: deque = field(default_factory=lambda: deque())

    # Debounce Parameters
    last_alert_time: float = 0.0
    debounce_time_alert: float = 4
    last_video_time: float = 0.0
    debounce_time_video: float = 30
    alert_lock: threading.Lock = field(default_factory=threading.Lock)
    video_lock: threading.Lock = field(default_factory=threading.Lock)

    # EAR measurements (time-stamped) for analyzing drowsiness
    ear_measurements: deque = field(default_factory=lambda: deque())
    # Keep a running sum for fast average computation
    ear_sum: float = 0.0

    # Debounce or cooldown for average EAR
    last_ear_reset_time: float = 0.0
    ear_reset_cooldown: float = 5.0   # seconds between avg EAR alerts
    avg_ear_threshold: float = 0.25

    def update_blink_rate(self) -> float:
        """
        Removes blink timestamps outside the analysis window and
        returns the current blink rate (blinks per minute).
        """
        now = time.time()
        cutoff = now - self.analysis_window
        # Remove blinks older than 'analysis_window' in one pass
        while self.blink_timestamps and self.blink_timestamps[0] < cutoff:
            self.blink_timestamps.popleft()

        # Compute blink rate: (#blinks in analysis_window) * (60 / analysis_window)
        count = len(self.blink_timestamps)
        return count * (60.0 / self.analysis_window) if count > 0 else 0.0

    def check_prolonged_EAR(self, current_EAR, ear_start_time) -> bool:
        """
        Checks if the eye aspect ratio has been continuously below EAR_THRESHOLD
        for longer than 'prolonged_EAR_duration_threshold' seconds.
        """
        if current_EAR < EAR_THRESHOLD:
            elapsed = time.time() - ear_start_time
            return elapsed > self.prolonged_EAR_duration_threshold
        return False

    def add_ear_measurement(self, ear_value: float) -> None:
        """
        Adds a new EAR measurement and removes those older than 'analysis_window'.
        Maintains a running sum for quick average computation.
        """
        now = time.time()
        self.ear_measurements.append((now, ear_value))
        self.ear_sum += ear_value

        # Remove measurements older than the analysis_window
        cutoff = now - self.analysis_window
        while self.ear_measurements and self.ear_measurements[0][0] < cutoff:
            old_time, old_val = self.ear_measurements.popleft()
            self.ear_sum -= old_val

    def get_average_ear(self) -> float:
        """
        Returns the average EAR over the last 'analysis_window' seconds.
        Computation is O(1) due to the running sum.
        """
        count = len(self.ear_measurements)
        return (self.ear_sum / count) if count > 0 else 0.0

    def is_drowsy(self, current_EAR, ear_start_time) -> tuple[bool, str]:
        """
        Evaluates multiple drowsiness conditions:
          1. High blink rate.
          2. Prolonged eye closure (EAR < threshold for too long).
          3. Low average EAR over 'analysis_window'.

        Resets the EAR measurements if a 'low average EAR' alert occurs
        and a cooldown has elapsed.

        Returns:
            (drowsy: bool, reason: str)
        """
        now = time.time()

        # 1. Compute blink rate (blinks/min)
        blink_rate = self.update_blink_rate()

        # 2. Check if current EAR is below threshold for too long
        prolonged_EAR = self.check_prolonged_EAR(current_EAR, ear_start_time)

        # 3. Check average EAR
        average_ear = self.get_average_ear()

        drowsy = False
        reason = ""

        # Condition #1: High blink rate
        if blink_rate > self.blink_rate_threshold:
            drowsy = True
            reason = "high_blink_rate"

        # Condition #2: Prolonged eye closure
        if prolonged_EAR:
            drowsy = True
            reason = "Prolonged_eye_closure"

        # Condition #3: Low average EAR
        # Only trigger if we've passed the cooldown time.
        if average_ear < 0.20 and (now - self.last_ear_reset_time) > self.ear_reset_cooldown:
            print(average_ear)
            self.ear_measurements.clear()
            self.ear_sum = 0.0  # reset running sum
            self.last_ear_reset_time = now
            drowsy = True
            reason = "low_average_ear"

        return drowsy, reason






''' working for production but unoptimized '''
# @dataclass
# class AppState:
#     blink_counter: int  = 0
#     total_blinks: int   = 0
#     EAR_consec_frames: int = 0
#     frame_buffer: deque = field(default_factory=lambda: deque())
#     buffer_size: int = 300
#     fps: float = 0.0
#     blink_durations: list = field(default_factory=list)  # Stores duration of each blink
#     is_blinking: bool = False         # Indicates if a blink is ongoing
#     current_blink_start: float = 0.0  # Timestamp when the current blink started
#
#     # Analysis Parameters
#     analysis_window: int = 20               # seconds (used for blink rate + EAR over time)
#     blink_rate_threshold: float = 90.0      # blinks per minute
#     prolonged_EAR_duration_threshold: float = 1  # seconds
#
#     # Timestamps of blinks for rate calculation
#     blink_timestamps: deque = field(default_factory=lambda: deque())
#
#     # Debounce Parameters
#     last_alert_time: float      = 0.0   # Timestamp of the last alert
#     debounce_time_alert: float  = 4     # Minimum seconds between alerts
#     last_video_time: float      = 0.0   # Timestamp of the last video
#     debounce_time_video: float  = 30    # Minimum seconds between video savings
#     alert_lock: threading.Lock  = field(default_factory=threading.Lock)
#     video_lock: threading.Lock  = field(default_factory=threading.Lock)
#
#     ear_measurements: deque = field(default_factory=lambda: deque())
#     # Debounce or cooldown for average EAR
#     last_ear_reset_time: float = 0.0
#     ear_reset_cooldown: float  = 5.0  # in seconds, example value
#
#     # Threshold for average EAR alert
#     avg_ear_threshold: float = 0.25
#
#     def update_blink_rate(self) -> float:
#         current_time = time.time()
#         # Remove blinks outside the analysis window
#         while self.blink_timestamps and current_time - self.blink_timestamps[0] > self.analysis_window:
#             self.blink_timestamps.popleft()
#
#         blink_rate = len(self.blink_timestamps) * (60 / self.analysis_window)  # blinks per minute
#         return blink_rate
#
#     def check_prolonged_EAR(self, current_EAR, ear_start_time) -> bool :
#         if current_EAR < EAR_THRESHOLD:
#             elapsed = time.time() - ear_start_time
#             if elapsed > self.prolonged_EAR_duration_threshold:
#                 return True
#         return False
#
#     #### NEW METHODS FOR AVERAGE EAR TRACKING ####
#     def add_ear_measurement(self, ear_value: float) -> None:
#         """Add a new EAR measurement with a timestamp and remove old entries."""
#         now = time.time()
#         self.ear_measurements.append((now, ear_value))
#
#         # Remove EAR measurements older than analysis_window seconds
#         while self.ear_measurements and (now - self.ear_measurements[0][0]) > self.analysis_window:
#             self.ear_measurements.popleft()
#
#     def get_average_ear(self) -> float:
#         """Compute the average EAR over the last 'analysis_window' seconds."""
#         if not self.ear_measurements:
#             return 0.0
#         total_ear = sum(val for _, val in self.ear_measurements)
#         return total_ear / len(self.ear_measurements)
#
#     def is_drowsy(self, current_EAR, ear_start_time) -> tuple[bool, str]:
#         blink_rate = self.update_blink_rate()
#         prolonged_EAR = self.check_prolonged_EAR(current_EAR, ear_start_time)
#
#         # NEW: Calculate average EAR over the analysis window
#         average_ear = self.get_average_ear()
#
#         drowsy = False
#         reason = ""
#
#         # Condition #1: High blink rate
#         if blink_rate > self.blink_rate_threshold:
#             drowsy = True
#             # reasons.append(f"High blink rate: {blink_rate:.2f} blinks/min")
#             reason = "high_blink_rate"
#
#         # Condition #2: Prolonged eye closure
#         if prolonged_EAR:
#             drowsy = True
#             # reasons.append("Prolonged eye closure detected")
#             reason = "Prolonged_eye_closure"
#
#         # Condition #3: Low average EAR
#         now = time.time()
#         if average_ear < 0.29:
#             # Only trigger if we've passed the cooldown time
#             if (now - self.last_ear_reset_time) > self.ear_reset_cooldown:
#                 # Clear the EAR measurements
#                 self.ear_measurements.clear()
#                 self.last_ear_reset_time = now
#                 drowsy = True
#                 # reasons.append(f"Low average EAR over {self.analysis_window}s: {average_ear:.2f}")
#                 reason = "low_average_ear"
#
#         return drowsy, reason















# Original code before the ear monitoring over time.
# import time
#
# import threading
# from dataclasses import dataclass, field
# from collections import deque
#
# EAR_THRESHOLD = 0.23
#
# @dataclass
# class AppState:
#     blink_counter: int  = 0
#     total_blinks: int   = 0
#     EAR_consec_frames: int = 0
#     frame_buffer: deque = field(default_factory=lambda: deque())
#     buffer_size: int = 300
#     fps: float = 0.0
#     blink_durations: list = field(default_factory=list)  # Stores duration of each blink
#     is_blinking: bool = False         # Indicates if a blink is ongoing
#     current_blink_start: float = 0.0  # Timestamp when the current blink started
#
#     # Analysis Parameters
#     analysis_window: int = 60              # seconds
#     blink_rate_threshold: float = 75.0     # blinks per minute
#     prolonged_EAR_duration_threshold: float = 1  # seconds
#
#     # Timestamps of blinks for rate calculation
#     blink_timestamps: deque = field(default_factory=lambda: deque())
#
#     # Debounce Parameters
#     last_alert_time: float      = 0.0   # Timestamp of the last alert
#     debounce_time_alert: float  = 4   # Minimum seconds between alerts
#     last_video_time: float      = 0.0   # Timestamp of the last video
#     debounce_time_video: float  = 30    # Minimum seconds between video savings
#     alert_lock: threading.Lock  = field(default_factory=threading.Lock)
#     video_lock: threading.Lock  = field(default_factory=threading.Lock)
#
#     def update_blink_rate(self) -> float:
#         current_time = time.time()
#         # Remove blinks outside the analysis window
#         while self.blink_timestamps and current_time - self.blink_timestamps[0] > self.analysis_window:
#             self.blink_timestamps.popleft()
#         blink_rate = len(self.blink_timestamps) * (60 / self.analysis_window)  # blinks per minute
#         return blink_rate
#
#     def check_prolonged_EAR(self, current_EAR, ear_start_time) -> bool :
#         if current_EAR < EAR_THRESHOLD:
#             elapsed = time.time() - ear_start_time
#             if elapsed > self.prolonged_EAR_duration_threshold:
#                 return True
#         return False
#
#     def is_drowsy(self, current_EAR, ear_start_time) -> tuple[bool, list[str]]:
#         blink_rate = self.update_blink_rate()
#         prolonged_EAR = self.check_prolonged_EAR(current_EAR, ear_start_time)
#
#         drowsy = False
#         reasons = []
#
#         if blink_rate > self.blink_rate_threshold:
#             drowsy = True
#             reasons.append(f"High blink rate: {blink_rate:.2f} blinks/min")
#
#         if prolonged_EAR:
#             drowsy = True
#             reasons.append("Prolonged eye closure detected")
#
#         return drowsy, reasons