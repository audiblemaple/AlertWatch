import time

import threading
from dataclasses import dataclass, field
from collections import deque

EAR_THRESHOLD = 0.2

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
    analysis_window: int = 60              # seconds
    blink_rate_threshold: float = 70.0     # blinks per minute
    blink_duration_threshold: float = 0.4  # seconds
    prolonged_EAR_duration_threshold: float = 0.8  # seconds

    # Timestamps of blinks for rate calculation
    blink_timestamps: deque = field(default_factory=lambda: deque())

    # Debounce Parameters
    last_alert_time: float      = 0.0   # Timestamp of the last alert
    debounce_time_alert: float  = 1.5   # Minimum seconds between alerts
    last_video_time: float      = 0.0   # Timestamp of the last video
    debounce_time_video: float  = 30    # Minimum seconds between video savings
    alert_lock: threading.Lock  = field(default_factory=threading.Lock)
    video_lock: threading.Lock  = field(default_factory=threading.Lock)

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

    def is_drowsy(self, current_EAR, ear_start_time) -> tuple[bool, list[str]]:
        blink_rate = self.update_blink_rate()
        # long_blinks = self.check_blink_duration()
        prolonged_EAR = self.check_prolonged_EAR(current_EAR, ear_start_time)

        drowsy = False
        reasons = []

        if blink_rate > self.blink_rate_threshold:
            drowsy = True
            reasons.append(f"High blink rate: {blink_rate:.2f} blinks/min")

        if prolonged_EAR:
            drowsy = True
            reasons.append("Prolonged eye closure detected")

        return drowsy, reasons


# # from dataclasses import dataclass, field
# # from collections import deque
# #
# # @dataclass
# # class AppState:
# #     blink_counter: int = 0
# #     total_blinks: int = 0
# #     EAR_consec_frames: int = 0
# #     frame_buffer: deque = field(default_factory=lambda: deque())
# #     buffer_size: int = 300
# #     fps: float = 0.0
# #     blink_durations: list = field(default_factory=list)  # Stores duration of each blink
# #     is_blinking: bool = False  # Indicates if a blink is ongoing
# #     current_blink_start: float = 0.0  # Timestamp when the current blink started
#
#
#
# from dataclasses import dataclass, field
# from collections import deque
# import time
#
# EAR_THRESHOLD = 0.21
#
# @dataclass
# class AppState:
#     blink_counter: int = 0
#     total_blinks: int = 0
#     EAR_consec_frames: int = 0
#     frame_buffer: deque = field(default_factory=lambda: deque())
#     buffer_size: int = 300
#     fps: float = 0.0
#     blink_durations: list = field(default_factory=list)  # Stores duration of each blink
#     is_blinking: bool = False  # Indicates if a blink is ongoing
#     current_blink_start: float = 0.0  # Timestamp when the current blink started
#
#     # Analysis Parameters
#     analysis_window: int = 60  # seconds
#     blink_rate_threshold: int = 15  # blinks / minute
#     blink_duration_threshold: float = 0.5  # seconds
#     prolonged_EAR_duration_threshold: float= 0.8  # seconds
#     lastalert: int = 0 # timestamp
#     debounce: int = 0.5 # seconds
#
#     # Timestamps of blinks for rate calculation
#     blink_timestamps: deque = field(default_factory=lambda: deque())
#
#     def update_blink_rate(self) -> float:
#         current_time = time.time()
#         # Remove blinks outside the analysis window
#         while self.blink_timestamps and current_time - self.blink_timestamps[0] > self.analysis_window:
#             self.blink_timestamps.popleft()
#         blink_rate = len(self.blink_timestamps) * (60 / self.analysis_window)  # blinks per minute
#         return blink_rate
#
#     # def check_blink_duration(self):
#     #     # Check if any blink duration exceeds the threshold
#     #     for duration in self.blink_durations[-10:]:  # Check last 10 blinks
#     #         if duration > self.blink_duration_threshold:
#     #             return True
#     #     return False
#
#     def check_prolonged_EAR(self, current_EAR, ear_start_time) -> bool:
#         if current_EAR < EAR_THRESHOLD:
#             elapsed = time.time() - ear_start_time
#             if elapsed > self.prolonged_EAR_duration_threshold:
#                 return True
#         return False
#
#     def is_drowsy(self, current_EAR, ear_start_time) -> tuple:
#         blink_rate = self.update_blink_rate()
#         # long_blinks = self.check_blink_duration()
#         prolonged_EAR = self.check_prolonged_EAR(current_EAR, ear_start_time)
#
#         drowsy = False
#         reasons = []
#
#         if blink_rate > self.blink_rate_threshold:
#             drowsy = True
#             reasons.append(f"High blink rate: {blink_rate:.2f} blinks/min")
#
#         # if long_blinks:
#         #     drowsy = True
#         #     reasons.append("Prolonged blink duration detected")
#
#         if prolonged_EAR:
#             drowsy = True
#             reasons.append("Prolonged eye closure detected")
#
#         return drowsy, reasons