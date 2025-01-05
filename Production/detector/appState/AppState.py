from dataclasses import dataclass, field
from collections import deque
import threading
import time

EAR_THRESHOLD = 0.25

@dataclass
class AppState:
    blink_counter: int  = 0
    total_blinks: int   = 0
    EAR_consec_frames: int = 0
    frame_buffer: deque = field(default_factory=lambda: deque())
    buffer_size: int = 300
    fps: float = 0.0
    blink_durations: list = field(default_factory=list)  # Stores duration of each blink
    is_blinking: bool = False         # Indicates if a blink is ongoing
    current_blink_start: float = 0.0  # Timestamp when the current blink started

    # Analysis Parameters
    analysis_window: int = 20               # seconds (used for blink rate + EAR over time)
    blink_rate_threshold: float = 90.0      # blinks per minute
    prolonged_EAR_duration_threshold: float = 1  # seconds

    # Timestamps of blinks for rate calculation
    blink_timestamps: deque = field(default_factory=lambda: deque())

    # Debounce Parameters
    last_alert_time: float      = 0.0   # Timestamp of the last alert
    debounce_time_alert: float  = 4     # Minimum seconds between alerts
    last_video_time: float      = 0.0   # Timestamp of the last video
    debounce_time_video: float  = 30    # Minimum seconds between video savings
    alert_lock: threading.Lock  = field(default_factory=threading.Lock)
    video_lock: threading.Lock  = field(default_factory=threading.Lock)

    ear_measurements: deque = field(default_factory=lambda: deque())
    # Debounce or cooldown for average EAR
    last_ear_reset_time: float = 0.0
    ear_reset_cooldown: float  = 5.0  # in seconds, example value

    # Threshold for average EAR alert
    avg_ear_threshold: float = 0.25

    def update_blink_rate(self) -> float:
        current_time = time.time()
        # Remove blinks outside the analysis window
        while self.blink_timestamps and current_time - self.blink_timestamps[0] > self.analysis_window:
            self.blink_timestamps.popleft()

        blink_rate = len(self.blink_timestamps) * (60 / self.analysis_window)  # blinks per minute
        return blink_rate

    def check_prolonged_EAR(self, current_EAR, ear_start_time) -> bool :
        if current_EAR < EAR_THRESHOLD:
            elapsed = time.time() - ear_start_time
            if elapsed > self.prolonged_EAR_duration_threshold:
                return True
        return False

    #### NEW METHODS FOR AVERAGE EAR TRACKING ####
    def add_ear_measurement(self, ear_value: float) -> None:
        """Add a new EAR measurement with a timestamp and remove old entries."""
        now = time.time()
        self.ear_measurements.append((now, ear_value))

        # Remove EAR measurements older than analysis_window seconds
        while self.ear_measurements and (now - self.ear_measurements[0][0]) > self.analysis_window:
            self.ear_measurements.popleft()

    def get_average_ear(self) -> float:
        """Compute the average EAR over the last 'analysis_window' seconds."""
        if not self.ear_measurements:
            return 0.0
        total_ear = sum(val for _, val in self.ear_measurements)
        return total_ear / len(self.ear_measurements)

    def is_drowsy(self, current_EAR, ear_start_time) -> tuple[bool, str]:
        blink_rate = self.update_blink_rate()
        prolonged_EAR = self.check_prolonged_EAR(current_EAR, ear_start_time)

        # NEW: Calculate average EAR over the analysis window
        average_ear = self.get_average_ear()

        drowsy = False
        reason = ""

        # Condition #1: High blink rate
        if blink_rate > self.blink_rate_threshold:
            drowsy = True
            # reasons.append(f"High blink rate: {blink_rate:.2f} blinks/min")
            reason = "high_blink_rate"

        # Condition #2: Prolonged eye closure
        # if prolonged_EAR:
        #     drowsy = True
        #     # reasons.append("Prolonged eye closure detected")
        #     reason = "Prolonged_eye_closure"

        # Condition #3: Low average EAR
        now = time.time()
        if average_ear < 0.29:
            # Only trigger if we've passed the cooldown time
            if (now - self.last_ear_reset_time) > self.ear_reset_cooldown:
                # Clear the EAR measurements
                self.ear_measurements.clear()
                self.last_ear_reset_time = now
                drowsy = True
                # reasons.append(f"Low average EAR over {self.analysis_window}s: {average_ear:.2f}")
                reason = "low_average_ear"

        return drowsy, reason


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