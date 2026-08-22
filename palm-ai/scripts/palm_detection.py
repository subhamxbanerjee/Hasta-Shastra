"""
palm_detection.py — Phase 4, Milestone 4.1
MediaPipe Hand Landmarks & Palm Crop

WHAT WE ARE BUILDING:
  A script that uses MediaPipe to detect hand landmarks and isolate
  the palm region (removing most fingers and background).

WHY WE ARE BUILDING IT:
  For palm line detection (Life, Head, Heart lines), feeding an entire image
  with background and fingers into a model creates noise. By consistently 
  detecting the hand and cropping only the palm, we standardize the input
  for future machine learning models.

CURRENT STATUS:
  Image → Hand Landmarks → Palm Crop → Normalized 512x512

NOT YET:
  Image → Life Line / Head Line / Heart Line
  (This script ONLY localizes the palm; it does not trace palm lines).

HOW TO RUN:
  From the palm-ai/ folder, with .venv active:
    python scripts/palm_detection.py
"""

import cv2
import mediapipe as mp
import numpy as np
import sys
import os
import urllib.request

# ── Central config ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import RAW_IMAGES_DIR, PROCESSED_DIR, MODELS_DIR, IMAGE_SIZE

IMAGE_FILENAME = "test_palm.jpg"
IMAGE_PATH = os.path.join(RAW_IMAGES_DIR, IMAGE_FILENAME)
MODEL_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")

print("=" * 68)
print("  PalmVerse — Phase 4, Milestone 4.1: Palm Localization")
print("=" * 68)

# =============================================================================
# STEP 0 — DOWNLOAD MEDIAPIPE MODEL BUNDLE
# =============================================================================
# Modern MediaPipe uses the "Tasks API", which requires a model bundle file.
os.makedirs(MODELS_DIR, exist_ok=True)
if not os.path.exists(MODEL_PATH):
    print("─── Step 0: Download Hand Landmarker Model ────────────────────────")
    print("  Downloading model (this only happens once)...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("  ✅ Downloaded hand_landmarker.task")

# =============================================================================
# STEP 1 — LOAD IMAGE
# =============================================================================
print("\n─── Step 1: Load Image ──────────────────────────────────────────────")

if not os.path.exists(IMAGE_PATH):
    print(f"❌ File not found: {IMAGE_PATH}")
    sys.exit(1)

image_bgr = cv2.imread(IMAGE_PATH)
if image_bgr is None:
    print("❌ OpenCV could not read the image.")
    sys.exit(1)

orig_h, orig_w, orig_c = image_bgr.shape
print(f"  Width    : {orig_w} px")
print(f"  Height   : {orig_h} px")
print(f"  Channels : {orig_c}")
print("  ✅ Image loaded successfully.")

# =============================================================================
# STEP 2 — CONVERT BGR TO RGB
# =============================================================================
# EXPLANATION:
# OpenCV historically uses BGR (Blue, Green, Red) channel ordering because
# early camera manufacturers used that format. MediaPipe (and most modern
# ML libraries like PyTorch/TensorFlow) expect standard RGB ordering.
# We MUST convert the image to RGB before passing it to MediaPipe, otherwise
# the model will process inverted colors and may fail to detect the hand.

print("\n─── Step 2: Convert to RGB ──────────────────────────────────────────")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
print("  ✅ Converted BGR to RGB for MediaPipe compatibility.")


# =============================================================================
# STEP 3 — INITIALIZE MEDIAPIPE HAND DETECTOR
# =============================================================================
print("\n─── Step 3: Initialize MediaPipe Tasks API ──────────────────────────")

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5
)

# We initialize the landmarker in a context manager so it cleans up automatically
landmarker = HandLandmarker.create_from_options(options)
print("  ✅ MediaPipe HandLandmarker initialized (Tasks API).")


# =============================================================================
# STEP 4 — DETECT 21 HAND LANDMARKS
# =============================================================================
print("\n─── Step 4: Detect Landmarks ────────────────────────────────────────")

# Convert numpy array to MediaPipe Image object
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

# Run detection
results = landmarker.detect(mp_image)

if not results.hand_landmarks:
    print("  ❌ No hand detected in the image.")
    print("  Exiting safely...")
    sys.exit(0)

# We requested num_hands=1, so we take the first detected hand.
hand_landmarks = results.hand_landmarks[0]
num_landmarks = len(hand_landmarks)

print(f"  ✅ Hand detected! Found {num_landmarks} landmarks.")
print("  (Showing first 5 landmarks as an example)")
print(f"  {'ID':<5} {'X (normalized)':<18} {'Y (normalized)':<18}")
print("  " + "-" * 45)

for i in range(5):
    lm = hand_landmarks[i]
    print(f"  {i:<5} {lm.x:<18.4f} {lm.y:<18.4f}")

# =============================================================================
# STEP 5 — DRAW LANDMARK VISUALIZATION
# =============================================================================
print("\n─── Step 5: Visualization ───────────────────────────────────────────")

# We will manually draw circles and connections since the Tasks API
# doesn't include the old drawing_utils natively in the same way.
image_annotated = image_bgr.copy()

# A minimal set of connections for a hand
CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),       # Thumb
    (0,5), (5,6), (6,7), (7,8),       # Index
    (5,9), (9,10), (10,11), (11,12),  # Middle
    (9,13), (13,14), (14,15), (15,16),# Ring
    (13,17), (0,17), (17,18), (18,19), (19,20) # Pinky & Palm base
]

# Get pixel coordinates for all landmarks
px_points = []
for lm in hand_landmarks:
    px_x = int(lm.x * orig_w)
    px_y = int(lm.y * orig_h)
    px_points.append((px_x, px_y))

# Draw connections
for p1_idx, p2_idx in CONNECTIONS:
    cv2.line(image_annotated, px_points[p1_idx], px_points[p2_idx], (0, 255, 0), 2)

# Draw landmark dots
for (x, y) in px_points:
    cv2.circle(image_annotated, (x, y), 5, (0, 0, 255), -1)

viz_path = os.path.join(PROCESSED_DIR, "palm_landmarks.jpg")
cv2.imwrite(viz_path, image_annotated)
print(f"  ✅ Landmarks drawn and saved to: {viz_path}")

# =============================================================================
# STEP 6 — EXPLAIN PALM REGION SELECTION
# =============================================================================
print("\n─── Step 6 & 7: Palm Region Bounding Box ────────────────────────────")
# EXPLANATION:
# A standard bounding box would enclose the entire hand, including all fingers.
# Fingers contain knuckle wrinkles that can confuse a palm-line detection model.
# To focus specifically on the Life, Head, and Heart lines, we want a tighter box.
# 
# We can form a "palm-only" bounding box by looking only at specific landmarks:
# 0: Wrist
# 1, 2: Base of thumb
# 5, 9, 13, 17: The MCP joints (knuckles at the base of the fingers)
# 
# By finding the min and max coordinates of ONLY these points, we exclude 
# the upper fingers completely.

PALM_LANDMARK_IDS = [0, 1, 2, 5, 9, 13, 17]

palm_points = [px_points[i] for i in PALM_LANDMARK_IDS]

# =============================================================================
# STEP 7 — CALCULATE PALM BOUNDING BOX
# =============================================================================
x_coords = [p[0] for p in palm_points]
y_coords = [p[1] for p in palm_points]

# Find the min/max to form a bounding box
raw_x_min, raw_x_max = min(x_coords), max(x_coords)
raw_y_min, raw_y_max = min(y_coords), max(y_coords)

# Add padding (e.g., 20% of the width/height of the raw box)
# This ensures we don't accidentally cut off the edges of the major lines.
box_w = raw_x_max - raw_x_min
box_h = raw_y_max - raw_y_min
pad_x = int(box_w * 0.20)
pad_y = int(box_h * 0.20)

# Clamp coordinates so they don't go outside the image boundaries
x_min = max(0, raw_x_min - pad_x)
y_min = max(0, raw_y_min - pad_y)
x_max = min(orig_w, raw_x_max + pad_x)
y_max = min(orig_h, raw_y_max + pad_y)

final_box_w = x_max - x_min
final_box_h = y_max - y_min

print(f"  Calculated Box : X:[{x_min} - {x_max}], Y:[{y_min} - {y_max}]")
print(f"  Crop Dimensions: {final_box_w} × {final_box_h} px")


# =============================================================================
# STEP 8 — CREATE PALM CROP
# =============================================================================
print("\n─── Step 8: Create Palm Crop ────────────────────────────────────────")

# NumPy arrays are sliced as [y_start:y_end, x_start:x_end]
palm_crop = image_bgr[y_min:y_max, x_min:x_max]

crop_path = os.path.join(PROCESSED_DIR, "palm_crop.jpg")
cv2.imwrite(crop_path, palm_crop)
print(f"  ✅ Cropped palm saved to: {crop_path}")


# =============================================================================
# STEP 9 — NORMALIZE FOR FUTURE ML
# =============================================================================
# EXPLANATION:
# Machine Learning models (like PyTorch CNNs) expect all inputs to be the exact
# same size. We resize our varied-size crops to a consistent 512x512.
# We use cv2.INTER_AREA interpolation because we are shrinking (downsampling) 
# the image; INTER_AREA averages pixels together and prevents moiré patterns/artifacts.

print("\n─── Step 9: Normalize to 512x512 ────────────────────────────────────")

palm_512 = cv2.resize(palm_crop, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

norm_path = os.path.join(PROCESSED_DIR, "palm_512.jpg")
cv2.imwrite(norm_path, palm_512)
print(f"  ✅ Normalized image saved to: {norm_path}")


# =============================================================================
# STEP 10 — CREATE COMPARISON VISUALIZATION
# =============================================================================
print("\n─── Step 10: Comparison Grid ────────────────────────────────────────")

# Draw the bounding box on a fresh copy of the original image
image_with_box = image_bgr.copy()
cv2.rectangle(image_with_box, (x_min, y_min), (x_max, y_max), (0, 255, 0), 6)

def resize_for_grid(img, target_height=600):
    """Helper to scale images to uniform height for side-by-side concatenation."""
    h, w = img.shape[:2]
    scale = target_height / h
    return cv2.resize(img, (int(w * scale), target_height), interpolation=cv2.INTER_AREA)

# Resize all panels to the same height so hstack works cleanly
panel1 = resize_for_grid(image_bgr)
panel2 = resize_for_grid(image_annotated)
panel3 = resize_for_grid(image_with_box)
panel4 = resize_for_grid(palm_crop)
panel5 = resize_for_grid(palm_512)

# Create two rows to keep it readable (3 on top, 2 on bottom)
row1 = np.hstack([panel1, panel2, panel3])

# For row 2, we need to match row1's width. We'll pad the remaining space with black.
row2_content = np.hstack([panel4, panel5])
padding = np.zeros((600, row1.shape[1] - row2_content.shape[1], 3), dtype=np.uint8)
row2 = np.hstack([row2_content, padding])

grid = np.vstack([row1, row2])

grid_path = os.path.join(PROCESSED_DIR, "palm_detection_comparison.jpg")
cv2.imwrite(grid_path, grid)
print(f"  ✅ Comparison grid saved to: {grid_path}")

# =============================================================================
# STEP 11 — TERMINAL LEARNING SUMMARY
# =============================================================================
print("\n=================================================")
print("MILESTONE 4.1 COMPLETE")
print("=================================================")
print("""
What I learned:
1. MediaPipe detects landmarks rather than directly understanding
   palmistry.
2. Landmark coordinates are normalized values between 0 and 1.
3. We convert normalized coordinates to pixel coordinates using
   image width and height.
4. A palm crop reduces irrelevant background and finger information.
5. Normalizing to 512×512 gives future ML models consistent input.
6. This does NOT yet detect Life, Head or Heart Lines.
7. Palm line detection will require either a trained segmentation
   model or a carefully designed CV pipeline.

CURRENT STATUS:
Image → Hand Landmarks → Palm Crop → 512×512

NOT YET:
Image → Life Line / Head Line / Heart Line
""")

# Display the grid
# Resize window to fit screen better if it's too huge
cv2.namedWindow("Milestone 4.1 Comparison", cv2.WINDOW_NORMAL)
cv2.imshow("Milestone 4.1 Comparison", grid)
print("\n  🖼️  Display window open. Press ANY KEY in the window to close and exit.")
cv2.waitKey(0)
cv2.destroyAllWindows()
