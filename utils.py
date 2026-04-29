# ============================================================
# utils.py — Shared Helper / Utility Functions
# Used by both lane_detection.py and object_detection.py
# ============================================================

import cv2
import time
import numpy as np
from src import config


# ── FPS Counter ──────────────────────────────────────────────
class FPSCounter:
    """
    Tracks real-time FPS using a rolling average.
    Usage:
        fps = FPSCounter()
        fps.start()
        while True:
            fps.update()
            print(fps.get())
    """
    def __init__(self, avg_window: int = 30):
        self._avg_window = avg_window
        self._timestamps: list[float] = []
        self._start_time: float = 0.0

    def start(self) -> "FPSCounter":
        self._start_time = time.perf_counter()
        return self

    def update(self) -> None:
        """Call once per processed frame."""
        now = time.perf_counter()
        self._timestamps.append(now)
        # Keep only the last N timestamps
        if len(self._timestamps) > self._avg_window:
            self._timestamps.pop(0)

    def get(self) -> float:
        """Return rolling-average FPS."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0


# ── Frame Utilities ──────────────────────────────────────────
def resize_frame(frame: np.ndarray,
                 width: int = config.RESIZE_WIDTH,
                 height: int = config.RESIZE_HEIGHT) -> np.ndarray:
    """Resize a frame to the configured resolution."""
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Overlay FPS counter on the top-left of the frame."""
    if not config.SHOW_FPS:
        return frame
    text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame, text,
        config.FPS_POSITION,
        cv2.FONT_HERSHEY_SIMPLEX,
        config.FPS_FONT_SCALE,
        config.FPS_COLOR,
        config.FPS_THICKNESS,
        cv2.LINE_AA,
    )
    return frame


def draw_label(frame: np.ndarray,
               text: str,
               x1: int, y1: int,
               color: tuple) -> None:
    """
    Draw a filled label box above a bounding box.
    Automatically handles edge cases (label near top of frame).
    """
    (tw, th), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        config.LABEL_FONT_SCALE,
        config.LABEL_THICKNESS,
    )
    # Position label — push down if near top
    label_y = max(y1 - th - baseline - 4, th + baseline + 4)
    # Filled background rectangle
    cv2.rectangle(
        frame,
        (x1, label_y - th - baseline - 2),
        (x1 + tw + 4, label_y + 2),
        color,
        cv2.FILLED,
    )
    # White text on colored bg
    cv2.putText(
        frame, text,
        (x1 + 2, label_y - baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        config.LABEL_FONT_SCALE,
        (255, 255, 255),
        config.LABEL_THICKNESS,
        cv2.LINE_AA,
    )


def build_video_writer(cap: cv2.VideoCapture,
                       output_path: str = config.OUTPUT_PATH) -> cv2.VideoWriter:
    """
    Create a VideoWriter matching the input capture's resolution.
    Returns None if saving is disabled in config.
    """
    if not config.SAVE_OUTPUT:
        return None

    fourcc = cv2.VideoWriter_fourcc(*config.OUTPUT_CODEC)
    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        config.OUTPUT_FPS,
        (config.RESIZE_WIDTH, config.RESIZE_HEIGHT),
    )
    return writer


def validate_frame(frame) -> bool:
    """Return True only if frame is a non-empty numpy array."""
    return frame is not None and isinstance(frame, np.ndarray) and frame.size > 0
