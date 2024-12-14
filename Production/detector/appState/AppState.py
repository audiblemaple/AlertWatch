from dataclasses import dataclass, field
from collections import deque

@dataclass
class AppState:
    blink_counter: int = 0
    total_blinks: int = 0
    EAR_consec_frames: int = 0
    frame_buffer: deque = field(default_factory=lambda: deque())
    buffer_size: int = 300
    fps: float = 0.0
    blink_durations: list = field(default_factory=list)  # Stores duration of each blink
    is_blinking: bool = False  # Indicates if a blink is ongoing
    current_blink_start: float = 0.0  # Timestamp when the current blink started