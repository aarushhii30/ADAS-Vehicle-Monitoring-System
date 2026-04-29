# ============================================================
# lane_detection.py — Lane Detection Module
#
# Pipeline:
#   Raw frame
#     → Grayscale conversion
#     → Gaussian blur  (reduce noise)
#     → Canny edge detection  (find edges)
#     → Region of Interest mask  (focus on road)
#     → Probabilistic Hough Transform  (detect line segments)
#     → Line averaging  (left lane / right lane)
#     → Draw overlay on frame
# ============================================================

import cv2
import numpy as np
import math
from src import config


# ── Preprocessing ────────────────────────────────────────────
def preprocess(frame: np.ndarray) -> np.ndarray:
    """
    Convert frame to grayscale, blur it, then detect edges.

    Why Gaussian blur?  Canny is sensitive to noise; blurring smooths
    small intensity fluctuations so only real edges survive.

    Returns: edge map (single-channel uint8)
    """
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, config.GAUSSIAN_BLUR_K, sigmaX=0)
    edges = cv2.Canny(blur, config.CANNY_LOW_THRESH, config.CANNY_HIGH_THRESH)
    return edges


# ── Region of Interest ───────────────────────────────────────
def region_of_interest(edges: np.ndarray) -> np.ndarray:
    """
    Apply a trapezoidal mask so we only look at the road ahead.

    The trapezoid covers the lower portion of the frame where lanes
    are visible — ignoring sky, trees, and buildings above.

    Returns: masked edge map
    """
    h, w = edges.shape
    top_y    = int(h * config.ROI_TOP_RATIO)
    bottom_y = int(h * config.ROI_BOTTOM_RATIO)

    # Trapezoid vertices (bottom-left, top-left, top-right, bottom-right)
    polygon = np.array([[
        (int(w * 0.05), bottom_y),        # bottom-left
        (int(w * 0.40), top_y),            # top-left
        (int(w * 0.60), top_y),            # top-right
        (int(w * 0.95), bottom_y),         # bottom-right
    ]], dtype=np.int32)

    mask        = np.zeros_like(edges)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    return masked_edges


# ── Hough Line Detection ─────────────────────────────────────
def detect_lines(masked_edges: np.ndarray):
    """
    Run Probabilistic Hough Transform to find line segments.

    Returns: array of (x1,y1,x2,y2) segments, or empty array.
    """
    lines = cv2.HoughLinesP(
        masked_edges,
        rho=config.HOUGH_RHO,
        theta=np.deg2rad(config.HOUGH_THETA),
        threshold=config.HOUGH_THRESHOLD,
        minLineLength=config.HOUGH_MIN_LENGTH,
        maxLineGap=config.HOUGH_MAX_GAP,
    )
    return lines if lines is not None else []


# ── Line Classification & Averaging ──────────────────────────
def _slope_intercept(line) -> tuple[float, float] | None:
    """
    Return (slope, intercept) for a line segment.
    Filters nearly-horizontal lines (|slope| < 0.4) — they are
    road markings, not lane boundaries.
    """
    x1, y1, x2, y2 = line[0]
    if x2 == x1:               # vertical line — skip
        return None
    slope     = (y2 - y1) / (x2 - x1)
    if abs(slope) < 0.4:       # too flat — not a lane line
        return None
    intercept = y1 - slope * x1
    return slope, intercept


def average_lines(frame: np.ndarray, lines) -> tuple:
    """
    Separate detected segments into left / right lanes by slope sign:
      - Negative slope → left lane  (y decreases as x increases in image coords)
      - Positive slope → right lane

    Average all segments per side into a single representative line.

    Returns: (left_line, right_line) each as (x1,y1,x2,y2) or None
    """
    left_fits, right_fits = [], []

    for line in lines:
        params = _slope_intercept(line)
        if params is None:
            continue
        slope, intercept = params
        if slope < 0:
            left_fits.append((slope, intercept))
        else:
            right_fits.append((slope, intercept))

    def make_line(fits) -> tuple | None:
        if not fits:
            return None
        avg_slope, avg_intercept = np.mean(fits, axis=0)
        h = frame.shape[0]
        y1 = h                                    # bottom of frame
        y2 = int(h * config.ROI_TOP_RATIO)        # top of ROI
        if avg_slope == 0:
            return None
        x1 = int((y1 - avg_intercept) / avg_slope)
        x2 = int((y2 - avg_intercept) / avg_slope)
        return (x1, y1, x2, y2)

    return make_line(left_fits), make_line(right_fits)


# ── Drawing ──────────────────────────────────────────────────
def draw_lanes(frame: np.ndarray,
               left_line,
               right_line) -> np.ndarray:
    """
    Draw left/right lane lines on a transparent overlay and
    fill the drivable area between them in semi-transparent green.

    Returns: annotated frame (BGR)
    """
    overlay = frame.copy()
    h, w    = frame.shape[:2]

    # Draw individual lane lines
    for line in [left_line, right_line]:
        if line is not None:
            x1, y1, x2, y2 = line
            cv2.line(overlay, (x1, y1), (x2, y2),
                     config.LANE_COLOR, config.LANE_THICKNESS)

    # Fill drivable corridor between lanes
    if left_line is not None and right_line is not None:
        lx1, ly1, lx2, ly2 = left_line
        rx1, ry1, rx2, ry2 = right_line
        poly_pts = np.array([[lx1, ly1], [lx2, ly2],
                             [rx2, ry2], [rx1, ry1]], dtype=np.int32)
        cv2.fillPoly(overlay, [poly_pts], (0, 180, 0))  # green fill

    # Blend overlay with original frame for transparency
    result = cv2.addWeighted(overlay, config.LANE_OVERLAY_ALPHA,
                             frame,  1 - config.LANE_OVERLAY_ALPHA, 0)
    return result


# ── Public API ───────────────────────────────────────────────
def detect_and_draw_lanes(frame: np.ndarray) -> np.ndarray:
    """
    Full lane detection pipeline in one call.
    Input:  BGR frame
    Output: BGR frame with lane overlay
    """
    edges        = preprocess(frame)
    masked_edges = region_of_interest(edges)
    lines        = detect_lines(masked_edges)
    left, right  = average_lines(frame, lines)
    result       = draw_lanes(frame, left, right)
    return result
