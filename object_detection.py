# ============================================================
# object_detection.py — Object Detection Module (YOLOv8)
#
# Pipeline:
#   BGR frame
#     → YOLOv8 inference (ultralytics)
#     → Filter by class & confidence
#     → Return frame (UNMODIFIED) + detection list
#
# ⚠️  Drawing is intentionally REMOVED from this module.
#     All boxes / labels are drawn by main.py via the tracker,
#     so there are no duplicate or conflicting overlays.
# ============================================================

import cv2
import numpy as np
from ultralytics import YOLO
from src import config


# ── COCO Class Names (indices used by YOLOv8) ────────────────
COCO_NAMES = {
    0: "person",    1: "bicycle",   2: "car",        3: "motorcycle",
    4: "airplane",  5: "bus",       6: "train",      7: "truck",
    8: "boat",      9: "traffic light", 10: "fire hydrant",
    11: "stop sign", 12: "parking meter",
}


# ── Detector Class ───────────────────────────────────────────
class ObjectDetector:
    """
    Wraps a YOLOv8 model with filtered inference.

    detect() now returns raw detections WITHOUT drawing anything
    on the frame. main.py passes a throwaway copy anyway, but
    returning clean data is the correct contract.

    Usage:
        detector = ObjectDetector()
        frame, detections = detector.detect(frame)
        # detections → list of dicts:
        #   { "class_id", "class_name", "confidence",
        #     "box": (x1, y1, x2, y2) }
    """

    def __init__(self):
        print(f"[ObjectDetector] Loading model: {config.YOLO_MODEL_PATH}")
        try:
            self.model = YOLO(config.YOLO_MODEL_PATH)
            print("[ObjectDetector] Model loaded successfully.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load YOLO model from '{config.YOLO_MODEL_PATH}'.\n"
                f"Run: python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')\" "
                f"to auto-download.\nOriginal error: {e}"
            )

    # ── Core Inference ───────────────────────────────────────
    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, list[dict]]:
        """
        Run YOLOv8 inference on a single BGR frame.

        Args:
            frame: BGR image as numpy array (may be a throwaway copy
                   from main.py — we never draw on it regardless).

        Returns:
            frame       — returned as-is, completely unmodified
            detections  — list of dicts:
                          { "class_id":   int,
                            "class_name": str,
                            "confidence": float,
                            "box":        (x1, y1, x2, y2) }
        """
        results = self.model.predict(
            source=frame,
            conf=config.YOLO_CONF_THRESH,
            iou=config.YOLO_IOU_THRESH,
            imgsz=config.YOLO_IMG_SIZE,
            device=config.YOLO_DEVICE,
            classes=config.YOLO_CLASSES,
            verbose=False,
        )

        detections: list[dict] = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf     = float(box.conf[0])
                class_id = int(box.cls[0])

                if (config.YOLO_CLASSES is not None
                        and class_id not in config.YOLO_CLASSES):
                    continue

                class_name = COCO_NAMES.get(class_id, f"cls_{class_id}")

                # ✅ NO cv2.rectangle / draw_label calls here
                detections.append({
                    "class_id":   class_id,
                    "class_name": class_name,
                    "confidence": round(conf, 3),
                    "box":        (x1, y1, x2, y2),
                })

        return frame, detections

    # ── Optional: Danger Zone ────────────────────────────────
    def flag_close_objects(self,
                           detections: list[dict],
                           frame_height: int,
                           frame_width: int,
                           frame: np.ndarray,
                           threshold_ratio: float = 0.25) -> np.ndarray:
        """
        Highlight objects whose bounding box area exceeds a threshold —
        a simple proxy for proximity. Draws a red WARNING banner.
        Still usable from main.py if needed; untouched by the refactor.
        """
        min_box_area = (frame_height * threshold_ratio) * (frame_width * 0.1)

        warnings = []
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            box_area = (x2 - x1) * (y2 - y1)
            if box_area >= min_box_area:
                warnings.append(det["class_name"])

        if warnings:
            warning_text = f"WARNING: Close {', '.join(set(warnings))}"
            cv2.putText(frame, warning_text,
                        (20, frame_height - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.85, (0, 0, 255), 2, cv2.LINE_AA)

        return frame