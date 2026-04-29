#!/usr/bin/env python3

import sys
import argparse
import cv2
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deep_sort_realtime.deepsort_tracker import DeepSort
from src import config
from src.utils import FPSCounter, resize_frame, draw_fps, build_video_writer, validate_frame
from src.lane_detection import detect_and_draw_lanes
from src.object_detection import ObjectDetector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


def open_capture(source):
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source {source}")

    print("[main] Source opened:", source)
    return cap


def run(args):

    source      = args.source if args.source else config.INPUT_SOURCE
    save_output = not args.no_save and config.SAVE_OUTPUT
    display     = not args.no_display and config.DISPLAY_WINDOW

    cap    = open_capture(source)
    writer = build_video_writer(cap) if save_output else None

    detector = ObjectDetector()
    tracker  = DeepSort(max_age=30)

    prev_positions: dict = {}
    prev_time:      dict = {}

    SPEED_LIMIT        = 40   # km/h
    DISTANCE_THRESHOLD = 10   # metres
    PIXELS_PER_METER   = 8    # tune per camera height/angle

    fps_counter = FPSCounter(avg_window=30).start()
    print("[main] Running...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if not validate_frame(frame):
            continue

        frame = resize_frame(frame)

        # ── 1. Lane detection ─────────────────────────────────────────────────
        frame = detect_and_draw_lanes(frame)

        # ── 2. Object detection ───────────────────────────────────────────────
        # detect() no longer draws anything (fixed in object_detection.py).
        # Passing frame.copy() is kept as an extra safety net.
        _, detections = detector.detect(frame.copy())

        # ── 3. Build DeepSORT input ───────────────────────────────────────────
        # detect() returns dicts: {"box": (x1,y1,x2,y2), "confidence": float, ...}
        # DeepSORT expects:       ([x, y, w, h], confidence, class_id)
        dets = []
        for det in detections:
            try:
                x1, y1, x2, y2 = det["box"]           # ← unpack dict correctly
                conf            = det["confidence"]
                class_id        = det["class_id"]

                w = x2 - x1
                h = y2 - y1

                if w <= 2 or h <= 2:                   # skip degenerate boxes
                    continue

                dets.append(([x1, y1, w, h], conf, class_id))
            except Exception as e:
                print(f"[warn] bad detection skipped: {e}")
                continue

        # ── 4. Update tracker ─────────────────────────────────────────────────
        tracks = tracker.update_tracks(dets, frame=frame)

        # ── 5. Draw per-vehicle overlays ──────────────────────────────────────
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            l, t, r, b = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

            # Clamp to frame boundaries
            fh, fw = frame.shape[:2]
            l = max(0, l);      t = max(0, t)
            r = min(fw - 1, r); b = min(fh - 1, b)
            if r <= l or b <= t:
                continue

            cx  = (l + r) // 2
            cy  = (t + b) // 2
            now = time.time()

            # ── Speed ──────────────────────────────────────────────────────
            speed_kmh = 0.0
            if track_id in prev_positions:
                px, py = prev_positions[track_id]
                dt = now - prev_time[track_id]
                if dt > 0:
                    dist_px   = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                    speed_kmh = (dist_px / dt / PIXELS_PER_METER) * 3.6

            prev_positions[track_id] = (cx, cy)
            prev_time[track_id]      = now

            # ── Distance ───────────────────────────────────────────────────
            box_h      = b - t
            distance_m = 5000 / box_h if box_h > 0 else 0.0

            # ── Alert flags ────────────────────────────────────────────────
            is_overspeed      = speed_kmh  > SPEED_LIMIT
            is_collision_risk = distance_m < DISTANCE_THRESHOLD and is_overspeed

            # ── Bounding box ───────────────────────────────────────────────
            box_color = (0, 0, 255) if is_overspeed else (0, 255, 0)
            cv2.rectangle(frame, (l, t), (r, b), box_color, 2)

            # ── Labels stacked upward, no overlap ─────────────────────────
            font   = cv2.FONT_HERSHEY_SIMPLEX
            LINE_H = 26   # vertical gap between stacked lines (px)

            # Line 1 (closest to box): ID | speed | distance
            y_info = t - 8
            cv2.putText(
                frame,
                f"ID {track_id} | {int(speed_kmh)} km/h | {int(distance_m)} m",
                (l, y_info), font, 0.60, (0, 255, 255), 2, cv2.LINE_AA,
            )

            # Line 2: OVERSPEED
            if is_overspeed:
                y_over = y_info - LINE_H
                cv2.putText(
                    frame, "OVERSPEED!",
                    (l, y_over), font, 0.70, (0, 0, 255), 2, cv2.LINE_AA,
                )

                # Line 3: COLLISION RISK (only shown when overspeed is also active)
                if is_collision_risk:
                    y_coll = y_over - LINE_H
                    cv2.putText(
                        frame, "COLLISION RISK!",
                        (l, y_coll), font, 0.70, (0, 0, 255), 2, cv2.LINE_AA,
                    )

        # ── 6. FPS display ────────────────────────────────────────────────────
        fps_counter.update()
        frame = draw_fps(frame, fps_counter.get())

        if writer:
            writer.write(frame)

        if display:
            cv2.imshow(config.WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    run(args)