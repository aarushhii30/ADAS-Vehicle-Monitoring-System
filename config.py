# ============================================================
# config.py — Central Configuration for ADAS Project
# All tunable parameters live here. Change values here ONLY.
# ============================================================

# ── Input Source ─────────────────────────────────────────────
# Use 0 for webcam, or provide a file path string
INPUT_SOURCE = "videos/sample.mp4"   # e.g. 0 for webcam

# ── Output Settings ──────────────────────────────────────────
SAVE_OUTPUT       = True
OUTPUT_PATH       = "output/result.mp4"
OUTPUT_FPS        = 30
OUTPUT_CODEC      = "mp4v"           # codec for cv2.VideoWriter

# ── Frame Display ────────────────────────────────────────────
DISPLAY_WINDOW    = True             # show live window
WINDOW_NAME       = "ADAS System — Lane & Object Detection"
RESIZE_WIDTH      = 1280             # resize input frame width
RESIZE_HEIGHT     = 720              # resize input frame height

# ── Lane Detection (Canny + Hough) ───────────────────────────
CANNY_LOW_THRESH  = 50               # lower threshold for Canny edge detector
CANNY_HIGH_THRESH = 150              # upper threshold for Canny edge detector
GAUSSIAN_BLUR_K   = (5, 5)          # kernel size for Gaussian blur (must be odd)

# Hough Transform parameters
HOUGH_RHO         = 1                # distance resolution (pixels)
HOUGH_THETA       = 1                # angle resolution (degrees, converted to radians in code)
HOUGH_THRESHOLD   = 50              # min votes to detect a line
HOUGH_MIN_LENGTH  = 100              # min line length (pixels)
HOUGH_MAX_GAP     = 50              # max gap between line segments (pixels)

# Region of Interest (trapezoid covering road area)
ROI_TOP_RATIO     = 0.60             # top of ROI as fraction of frame height
ROI_BOTTOM_RATIO  = 0.95             # bottom of ROI as fraction of frame height

# Lane line drawing
LANE_COLOR        = (0, 255, 0)     # BGR green
LANE_THICKNESS    = 4
LANE_OVERLAY_ALPHA = 0.4            # transparency of filled lane overlay

# ── Object Detection (YOLOv8) ────────────────────────────────
YOLO_MODEL_PATH   = "models/yolov8n.pt"   # 'n' = nano (fastest), swap for yolov8s/m for better accuracy
YOLO_CONF_THRESH  = 0.40            # minimum confidence to show a detection
YOLO_IOU_THRESH   = 0.45            # NMS IoU threshold
YOLO_IMG_SIZE     = 640             # inference image size (pixels)
YOLO_DEVICE       = "cpu"           # "cpu" or "cuda" if GPU available

# Classes to detect (COCO class IDs). None = detect all.
# 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck, 9=traffic light, 11=stop sign
YOLO_CLASSES      = [0, 1, 2, 3, 5, 7, 9, 11]

# Bounding box colours per class (BGR)
CLASS_COLORS = {
    0:  (255, 100, 100),   # person   — blue
    1:  (255, 200, 50),    # bicycle  — sky
    2:  (50,  200, 255),   # car      — yellow-orange
    3:  (50,  255, 150),   # motorcycle — green
    5:  (50,  50,  255),   # bus      — red
    7:  (200, 50,  255),   # truck    — purple
    9:  (0,   220, 220),   # light    — yellow
    11: (0,   128, 255),   # stop     — orange
}
DEFAULT_BOX_COLOR = (200, 200, 200)  # fallback grey

BOX_THICKNESS     = 2
LABEL_FONT_SCALE  = 0.55
LABEL_THICKNESS   = 1

# ── FPS Display ──────────────────────────────────────────────
SHOW_FPS          = True
FPS_COLOR         = (0, 255, 255)   # cyan
FPS_POSITION      = (20, 40)
FPS_FONT_SCALE    = 1.0
FPS_THICKNESS     = 2
