#!/usr/bin/env python3
"""palm_normalization.py
Phase 4 – Milestone 4.2
Palm orientation normalization + ROI crop.

This script can be run directly (CLI) or imported for reuse.
"""

import argparse
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

# -----------------------------------------------------------------------------
# Core processing function
# -----------------------------------------------------------------------------

def run_normalization(
    input_image: Path,
    output_dir: Path,
    model_path: Path = None,
) -> Path:
    """Normalize a palm image and save results.

    Parameters
    ----------
    input_image: Path to the raw palm image.
    output_dir: Directory where all outputs will be written.
    model_path: Optional custom MediaPipe model path. If None, defaults to
        ``<project_root>/models/hand_landmarker.task``.

    Returns
    -------
    Path to the normalized 512×512 palm image.
    """
    input_image = input_image.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve defaults
    if model_path is None:
        model_path = Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"
    else:
        model_path = Path(model_path).resolve()

    # ---------------------------------------------------------------------
    # STEP 1 — LOAD IMAGE
    # ---------------------------------------------------------------------
    image = cv2.imread(str(input_image))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {input_image}")
    height, width = image.shape[:2]
    print(f"[INFO] Loaded {input_image.name}: {width}×{height}")

    # ---------------------------------------------------------------------
    # STEP 2 — MEDIA PIPE HAND DETECTION
    # ---------------------------------------------------------------------
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    with HandLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        raise RuntimeError("No hand detected.")
    hand_landmarks = result.hand_landmarks[0]

    # ---------------------------------------------------------------------
    # STEP 3 — CONVERT LANDMARKS TO PIXEL COORDINATES
    # ---------------------------------------------------------------------
    def landmark_to_pixel(lmk, w, h):
        x = int(lmk.x * w)
        y = int(lmk.y * h)
        return max(0, min(x, w - 1)), max(0, min(y, h - 1))

    pixel_points = [landmark_to_pixel(lmk, width, height) for lmk in hand_landmarks]

    WRIST = 0
    MIDDLE_MCP = 9
    wrist = np.array(pixel_points[WRIST], dtype=np.float32)
    middle_mcp = np.array(pixel_points[MIDDLE_MCP], dtype=np.float32)

    # ---------------------------------------------------------------------
    # STEP 4 — CALCULATE PALM ORIENTATION
    # ---------------------------------------------------------------------
    vector = middle_mcp - wrist
    current_angle = np.degrees(np.arctan2(vector[1], vector[0]))
    target_angle = -90.0
    rotation_angle = target_angle - current_angle
    print(f"[INFO] Rotation required: {rotation_angle:.2f}°")

    # ---------------------------------------------------------------------
    # STEP 5 — ROTATE IMAGE
    # ---------------------------------------------------------------------
    center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    rotated_image = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # ---------------------------------------------------------------------
    # STEP 6 — TRANSFORM LANDMARK COORDINATES
    # ---------------------------------------------------------------------
    def transform_points(points, matrix):
        out = []
        for x, y in points:
            pt = np.array([x, y, 1.0])
            nx, ny = matrix @ pt
            out.append((int(nx), int(ny)))
        return out

    rotated_points = transform_points(pixel_points, rotation_matrix)

    # ---------------------------------------------------------------------
    # STEP 7 — CALCULATE PALM ROI
    # ---------------------------------------------------------------------
    palm_indices = [0, 1, 2, 5, 9, 13, 17]
    palm_pts = np.array([rotated_points[i] for i in palm_indices])
    min_x, min_y = palm_pts.min(axis=0)
    max_x, max_y = palm_pts.max(axis=0)
    palm_width = np.linalg.norm(np.array(rotated_points[5]) - np.array(rotated_points[17]))
    side_padding = int(palm_width * 0.30)
    top_padding = int(palm_width * 0.25)
    bottom_padding = int(palm_width * 0.45)
    x1 = max(0, min_x - side_padding)
    y1 = max(0, min_y - top_padding)
    x2 = min(width, max_x + side_padding)
    y2 = min(height, max_y + bottom_padding)

    # ---------------------------------------------------------------------
    # STEP 8 — CROP PALM (square)
    # ---------------------------------------------------------------------
    def square_crop(img, x1, y1, x2, y2):
        h, w = img.shape[:2]
        size = max(x2 - x1, y2 - y1)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        x1 = cx - size // 2
        y1 = cy - size // 2
        x2 = x1 + size
        y2 = y1 + size
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        return img[y1:y2, x1:x2], (x1, y1, x2, y2)

    palm_crop, _ = square_crop(rotated_image, x1, y1, x2, y2)

    # ---------------------------------------------------------------------
    # STEP 9 — RESIZE TO 512×512
    # ---------------------------------------------------------------------
    normalized = cv2.resize(palm_crop, (512, 512), interpolation=cv2.INTER_AREA)
    normalized_path = output_dir / "palm_512.jpg"
    cv2.imwrite(str(normalized_path), normalized)
    print(f"[INFO] Normalized palm saved to {normalized_path}")

    # ---------------------------------------------------------------------
    # STEP 10 — OPTIONAL VISUALISATION
    # ---------------------------------------------------------------------
    original_vis = image.copy()
    for i, (x, y) in enumerate(pixel_points):
        cv2.circle(original_vis, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(original_vis, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    roi_vis = rotated_image.copy()
    cv2.rectangle(roi_vis, (x1, y1), (x2, y2), (0, 255, 0), 5)
    for idx in palm_indices:
        x, y = rotated_points[idx]
        if 0 <= x < width and 0 <= y < height:
            cv2.circle(roi_vis, (x, y), 8, (0, 0, 255), -1)

    (output_dir / "01_original_landmarks.jpg").write_bytes(cv2.imencode('.jpg', original_vis)[1].tobytes())
    (output_dir / "02_rotated_roi.jpg").write_bytes(cv2.imencode('.jpg', roi_vis)[1].tobytes())
    (output_dir / "03_palm_crop.jpg").write_bytes(cv2.imencode('.jpg', palm_crop)[1].tobytes())
    (output_dir / "04_normalized_palm_512.jpg").write_bytes(cv2.imencode('.jpg', normalized)[1].tobytes())

    # Comparison grid (optional)
    def resize_for_display(img, size=(500, 500)):
        r = cv2.resize(img, size)
        if len(r.shape) == 2:
            r = cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)
        return r

    def add_label(img, text):
        lbl = img.copy()
        cv2.rectangle(lbl, (0, 0), (lbl.shape[1], 45), (0, 0, 0), -1)
        cv2.putText(lbl, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return lbl

    panels = [
        add_label(resize_for_display(original_vis), "1. Original + Landmarks"),
        add_label(resize_for_display(roi_vis), "2. Rotated + ROI"),
        add_label(resize_for_display(palm_crop), "3. Palm Crop"),
        add_label(resize_for_display(normalized), "4. Normalized 512x512"),
    ]
    top = np.hstack(panels[:2])
    bottom = np.hstack(panels[2:])
    comparison = np.vstack([top, bottom])
    (output_dir / "palm_normalization_comparison.jpg").write_bytes(cv2.imencode('.jpg', comparison)[1].tobytes())

    return normalized_path

# -----------------------------------------------------------------------------
# CLI entry point – retains original behaviour
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Palm orientation normalization + ROI crop.")
    parser.add_argument("--input", type=str, help="Path to raw palm image.")
    parser.add_argument("--output-dir", type=str, default="data/processed/normalized", help="Directory to store outputs.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    input_path = Path(args.input) if args.input else Path(__file__).resolve().parents[1] / "data" / "raw" / "images" / "test_palm.jpg"
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parents[1] / "data" / "processed" / "normalized"
    run_normalization(input_path, output_dir)