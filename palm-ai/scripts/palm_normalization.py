import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path


# ============================================================
# PalmVerse — Phase 4, Milestone 4.2
# Palm Orientation Normalization + ROI Crop
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

IMAGE_PATH = ROOT_DIR / "data" / "raw" / "images" / "test_palm.jpg"
MODEL_PATH = ROOT_DIR / "models" / "hand_landmarker.task"
OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "normalized"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Helper: convert normalized landmark to pixel coordinates
# ------------------------------------------------------------
def landmark_to_pixel(landmark, width, height):
    x = int(landmark.x * width)
    y = int(landmark.y * height)

    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))

    return x, y


# ------------------------------------------------------------
# Helper: transform points after rotation
# ------------------------------------------------------------
def transform_points(points, matrix):
    transformed = []

    for x, y in points:
        point = np.array([x, y, 1.0])
        new_point = matrix @ point

        transformed.append(
            (int(new_point[0]), int(new_point[1]))
        )

    return transformed


# ------------------------------------------------------------
# Helper: create a square crop
# ------------------------------------------------------------
def square_crop(image, x1, y1, x2, y2):

    h, w = image.shape[:2]

    crop_w = x2 - x1
    crop_h = y2 - y1

    size = max(crop_w, crop_h)

    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    x1 = center_x - size // 2
    y1 = center_y - size // 2

    x2 = x1 + size
    y2 = y1 + size

    # Keep crop inside image
    if x1 < 0:
        x2 -= x1
        x1 = 0

    if y1 < 0:
        y2 -= y1
        y1 = 0

    if x2 > w:
        shift = x2 - w
        x1 -= shift
        x2 = w

    if y2 > h:
        shift = y2 - h
        y1 -= shift
        y2 = h

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    return image[y1:y2, x1:x2], (x1, y1, x2, y2)


# ============================================================
# STEP 1 — LOAD IMAGE
# ============================================================

print("=" * 68)
print("  PalmVerse — Phase 4, Milestone 4.2")
print("  Palm Orientation Normalization + ROI Crop")
print("=" * 68)

print("\n─── Step 1: Load Image ─────────────────────────────────────")

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(
        f"Could not load image:\n{IMAGE_PATH}"
    )

height, width = image.shape[:2]

print(f"  Image width  : {width}")
print(f"  Image height : {height}")
print("  ✅ Image loaded successfully.")


# ============================================================
# STEP 2 — MEDIAPIPE HAND DETECTION
# ============================================================

print("\n─── Step 2: Detect Hand Landmarks ──────────────────────────")

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=rgb_image
)

with HandLandmarker.create_from_options(options) as landmarker:

    result = landmarker.detect(mp_image)


if not result.hand_landmarks:
    raise RuntimeError(
        "\n❌ No hand detected.\n"
        "Try using a clearer palm image with the full hand visible."
    )


hand_landmarks = result.hand_landmarks[0]

print(f"  Landmarks detected : {len(hand_landmarks)}")
print("  ✅ Hand detected.")


# ============================================================
# STEP 3 — CONVERT LANDMARKS TO PIXELS
# ============================================================

print("\n─── Step 3: Convert Landmarks to Pixels ───────────────────")

pixel_points = []

for landmark in hand_landmarks:
    pixel_points.append(
        landmark_to_pixel(landmark, width, height)
    )

print("  ✅ Converted 21 normalized landmarks to pixel coordinates.")


# Important landmarks
WRIST = 0
MIDDLE_MCP = 9

wrist = np.array(pixel_points[WRIST], dtype=np.float32)
middle_mcp = np.array(pixel_points[MIDDLE_MCP], dtype=np.float32)


# ============================================================
# STEP 4 — CALCULATE PALM ORIENTATION
# ============================================================

print("\n─── Step 4: Calculate Palm Orientation ────────────────────")

vector = middle_mcp - wrist

dx = vector[0]
dy = vector[1]

current_angle = np.degrees(np.arctan2(dy, dx))

# We want wrist → middle finger direction to point upward.
target_angle = -90.0

rotation_angle = target_angle - current_angle

print(f"  Wrist position      : {tuple(wrist.astype(int))}")
print(f"  Middle MCP position : {tuple(middle_mcp.astype(int))}")
print(f"  Current angle       : {current_angle:.2f}°")
print(f"  Rotation required   : {rotation_angle:.2f}°")


# ============================================================
# STEP 5 — ROTATE IMAGE
# ============================================================

print("\n─── Step 5: Rotate Palm to Standard Orientation ───────────")

# Rotate around the middle of the image
center = (width / 2, height / 2)

rotation_matrix = cv2.getRotationMatrix2D(
    center,
    rotation_angle,
    1.0
)

rotated_image = cv2.warpAffine(
    image,
    rotation_matrix,
    (width, height),
    flags=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=(0, 0, 0)
)

print("  ✅ Image rotated.")


# ============================================================
# STEP 6 — ROTATE LANDMARK COORDINATES TOO
# ============================================================

print("\n─── Step 6: Transform Landmark Coordinates ────────────────")

rotated_points = transform_points(
    pixel_points,
    rotation_matrix
)

print("  ✅ All 21 landmark coordinates transformed.")


# ============================================================
# STEP 7 — CALCULATE PALM ROI
# ============================================================

print("\n─── Step 7: Calculate Palm Region of Interest ─────────────")

# Palm-related landmarks:
# wrist, thumb base, index MCP, middle MCP,
# ring MCP, pinky MCP

palm_indices = [
    0,   # wrist
    1,   # thumb CMC
    2,   # thumb MCP
    5,   # index MCP
    9,   # middle MCP
    13,  # ring MCP
    17   # pinky MCP
]

palm_points = np.array(
    [rotated_points[i] for i in palm_indices]
)

min_x = int(np.min(palm_points[:, 0]))
max_x = int(np.max(palm_points[:, 0]))

min_y = int(np.min(palm_points[:, 1]))
max_y = int(np.max(palm_points[:, 1]))


# Estimate palm width using index MCP ↔ pinky MCP
index_mcp = np.array(rotated_points[5])
pinky_mcp = np.array(rotated_points[17])

palm_width = np.linalg.norm(
    index_mcp - pinky_mcp
)


# Dynamic padding based on hand size
side_padding = int(palm_width * 0.30)
top_padding = int(palm_width * 0.25)
bottom_padding = int(palm_width * 0.45)


x1 = min_x - side_padding
x2 = max_x + side_padding

y1 = min_y - top_padding
y2 = max_y + bottom_padding


# Keep coordinates inside image
x1 = max(0, x1)
y1 = max(0, y1)

x2 = min(width, x2)
y2 = min(height, y2)


print(f"  Estimated palm width : {palm_width:.1f} px")
print(f"  Palm ROI             : ({x1}, {y1}) → ({x2}, {y2})")


# ============================================================
# STEP 8 — CROP PALM
# ============================================================

print("\n─── Step 8: Crop Palm ROI ─────────────────────────────────")

palm_crop, crop_box = square_crop(
    rotated_image,
    x1, y1, x2, y2
)

crop_x1, crop_y1, crop_x2, crop_y2 = crop_box

print(
    f"  Square crop : "
    f"({crop_x1}, {crop_y1}) → ({crop_x2}, {crop_y2})"
)

print(f"  Crop shape  : {palm_crop.shape[1]} × {palm_crop.shape[0]}")
print("  ✅ Palm cropped.")


# ============================================================
# STEP 9 — RESIZE TO STANDARD MODEL INPUT
# ============================================================

print("\n─── Step 9: Normalize to 512 × 512 ────────────────────────")

normalized_palm = cv2.resize(
    palm_crop,
    (512, 512),
    interpolation=cv2.INTER_AREA
)

print("  Final size : 512 × 512")
print("  ✅ Palm normalized.")


# ============================================================
# STEP 10 — CREATE VISUALIZATIONS
# ============================================================

print("\n─── Step 10: Create Visualizations ────────────────────────")

# Original with landmarks
original_vis = image.copy()

for i, (x, y) in enumerate(pixel_points):

    cv2.circle(
        original_vis,
        (x, y),
        8,
        (0, 255, 0),
        -1
    )

    cv2.putText(
        original_vis,
        str(i),
        (x + 5, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 255, 0),
        1
    )


# Rotated image with ROI box
roi_vis = rotated_image.copy()

cv2.rectangle(
    roi_vis,
    (crop_x1, crop_y1),
    (crop_x2, crop_y2),
    (0, 255, 0),
    5
)


# Draw rotated palm landmarks
for index in palm_indices:

    x, y = rotated_points[index]

    if 0 <= x < width and 0 <= y < height:

        cv2.circle(
            roi_vis,
            (x, y),
            8,
            (0, 0, 255),
            -1
        )


# Save individual stages
original_output = OUTPUT_DIR / "01_original_landmarks.jpg"
rotated_output = OUTPUT_DIR / "02_rotated_roi.jpg"
crop_output = OUTPUT_DIR / "03_palm_crop.jpg"
normalized_output = OUTPUT_DIR / "04_normalized_palm_512.jpg"

cv2.imwrite(str(original_output), original_vis)
cv2.imwrite(str(rotated_output), roi_vis)
cv2.imwrite(str(crop_output), palm_crop)
cv2.imwrite(str(normalized_output), normalized_palm)

print(f"  ✅ {original_output.name}")
print(f"  ✅ {rotated_output.name}")
print(f"  ✅ {crop_output.name}")
print(f"  ✅ {normalized_output.name}")


# ============================================================
# STEP 11 — BUILD COMPARISON GRID
# ============================================================

print("\n─── Step 11: Build Comparison Grid ────────────────────────")


def resize_for_display(img, size=(500, 500)):
    return cv2.resize(img, size)


panel1 = resize_for_display(original_vis)
panel2 = resize_for_display(roi_vis)
panel3 = resize_for_display(palm_crop)
panel4 = resize_for_display(normalized_palm)


def add_label(img, text):

    labeled = img.copy()

    cv2.rectangle(
        labeled,
        (0, 0),
        (labeled.shape[1], 45),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        labeled,
        text,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    return labeled


panel1 = add_label(panel1, "1. Original + Landmarks")
panel2 = add_label(panel2, "2. Rotated + Palm ROI")
panel3 = add_label(panel3, "3. Palm Crop")
panel4 = add_label(panel4, "4. Normalized 512x512")


top_row = np.hstack([panel1, panel2])
bottom_row = np.hstack([panel3, panel4])

comparison = np.vstack([top_row, bottom_row])


comparison_output = OUTPUT_DIR / "palm_normalization_comparison.jpg"

cv2.imwrite(
    str(comparison_output),
    comparison
)

print(f"  ✅ {comparison_output.name}")


# ============================================================
# STEP 12 — DISPLAY
# ============================================================

print("\n─── Step 12: Display ──────────────────────────────────────")

cv2.namedWindow(
    "PalmVerse — Milestone 4.2: Palm Normalization",
    cv2.WINDOW_NORMAL
)

cv2.imshow(
    "PalmVerse — Milestone 4.2: Palm Normalization",
    comparison
)

print("  Opening comparison window...")
print("  👉 Press any key to close.")


cv2.waitKey(0)
cv2.destroyAllWindows()


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 68)
print("  ✅ MILESTONE 4.2 COMPLETE!")
print("=" * 68)

print("""
  Pipeline achieved:

      Palm Image
          ↓
      Hand Detection
          ↓
      21 MediaPipe Landmarks
          ↓
      Calculate Orientation
          ↓
      Rotate Palm
          ↓
      Dynamic Palm ROI
          ↓
      Square Crop
          ↓
      Resize to 512 × 512
          ↓
      STANDARDIZED PALM INPUT

  This means future CV / AI models receive a much more
  consistent palm image.

  IMPORTANT:
  This still does NOT identify Life, Head, Heart or Fate lines.
  The next stages will use this normalized palm as the input
  for actual palm-line enhancement and detection.

  Output directory:
""")

print(f"  {OUTPUT_DIR}")

print("=" * 68)