"""
preprocess.py — Phase 1, Milestone 1.2: Grayscale + Contrast Enhancement

WHAT WE ARE BUILDING:
  A script that takes our palm image through three processing experiments:
    1. Grayscale conversion
    2. Global histogram equalization
    3. CLAHE (Contrast Limited Adaptive Histogram Equalization)

  We then save each result and build a side-by-side comparison image.

WHY WE ARE BUILDING IT:
  Palm lines are regions of brightness contrast — not colour.
  Before we can detect or segment lines, we need to understand how
  to enhance an image so that those lines are clearly visible.
  This milestone is about building intuition, not about final results.

WHAT YOU ARE LEARNING:
  - cv2.cvtColor()         — colour space conversion
  - cv2.equalizeHist()    — global histogram equalization
  - cv2.createCLAHE()     — adaptive local contrast enhancement
  - numpy concatenation    — building comparison grids
  - cv2.imwrite()          — saving images to disk

HOW TO RUN:
  From the palm-ai/ folder, with .venv active:
    python scripts/preprocess.py
"""

import cv2
import numpy as np
import sys
import os

# ── Import our central config ─────────────────────────────────────────────────
# We built config.py specifically so we never hardcode paths.
# Any script just imports and uses the constants.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import RAW_IMAGES_DIR, PROCESSED_DIR

# ── Settings ──────────────────────────────────────────────────────────────────
IMAGE_FILENAME = "test_palm.jpg"
IMAGE_PATH     = os.path.join(RAW_IMAGES_DIR, IMAGE_FILENAME)

print("=" * 65)
print("  PalmVerse — Phase 1, Milestone 1.2: Grayscale + Contrast")
print("=" * 65)

# =============================================================================
# STEP 1 — LOAD THE IMAGE
# =============================================================================
# Same pattern as test_image.py: check path exists, imread(), check for None.
# This is our standard loading block — you will write this from memory soon.
print("\n📂 Loading image...")

if not os.path.exists(IMAGE_PATH):
    print(f"❌ Image not found: {IMAGE_PATH}")
    sys.exit(1)

image_bgr = cv2.imread(IMAGE_PATH)

if image_bgr is None:
    print("❌ OpenCV could not read the image. File may be corrupted.")
    sys.exit(1)

height, width, channels = image_bgr.shape
print(f"✅ Loaded: {IMAGE_FILENAME}  |  Shape: {image_bgr.shape}  |  dtype: {image_bgr.dtype}")

# =============================================================================
# STEP 2 — CONVERT TO GRAYSCALE
# =============================================================================
# cv2.cvtColor(image, conversion_code) converts between colour spaces.
#
# cv2.COLOR_BGR2GRAY applies the formula:
#     Gray = 0.114×B + 0.587×G + 0.299×R
#
# Result: a 2D array of shape (H, W) — the channel dimension is gone.
# Each value is still uint8 (0–255), but now it represents brightness only.

print("\n─── Step 2: Grayscale Conversion ───────────────────────────────────")

image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

print(f"  Original shape  : {image_bgr.shape}   (H, W, C)")
print(f"  Grayscale shape : {image_gray.shape}  (H, W)  ← no channel dimension")
print(f"  dtype           : {image_gray.dtype}")

# Look at a sample pixel in both versions
r, c = height // 2, width // 2  # centre pixel
bgr_val  = image_bgr[r, c]
gray_val = image_gray[r, c]

print(f"\n  Centre pixel BGR  : {bgr_val}")
print(f"  Centre pixel Gray : {gray_val}")
print(f"  Calculated check  : {int(0.114*bgr_val[0] + 0.587*bgr_val[1] + 0.299*bgr_val[2])}  (approx, due to rounding)")

# =============================================================================
# STEP 3 — GLOBAL HISTOGRAM EQUALIZATION
# =============================================================================
# cv2.equalizeHist(gray_image) redistributes pixel values across 0–255.
#
# HOW IT WORKS (conceptually):
#   Imagine the histogram is a frequency chart: x-axis = brightness (0–255),
#   y-axis = how many pixels have that brightness.
#   If most pixels are clustered between 100–180 (medium brightness),
#   the image looks flat. equalizeHist stretches those values to use the
#   full 0–255 range, increasing perceived contrast.
#
# LIMITATION:
#   It processes the entire image with one global adjustment.
#   If lighting is uneven (e.g., the palm centre is bright, edges are dark),
#   a single global equalization won't do well on every region.
#
# INPUT:  Must be a single-channel (grayscale) uint8 image.
# OUTPUT: A grayscale uint8 image with equalized histogram.

print("\n─── Step 3: Global Histogram Equalization ───────────────────────────")

image_eq = cv2.equalizeHist(image_gray)

# Let's print the min/max brightness before and after to see the effect
print(f"  Before equalization — min: {image_gray.min()}, max: {image_gray.max()}, mean: {image_gray.mean():.1f}")
print(f"  After  equalization — min: {image_eq.min()},   max: {image_eq.max()},   mean: {image_eq.mean():.1f}")
print("  → The range should now be closer to 0–255 (full range utilised)")

# =============================================================================
# STEP 4 — CLAHE (Contrast Limited Adaptive Histogram Equalization)
# =============================================================================
# cv2.createCLAHE() builds a CLAHE object with two key parameters:
#
#   clipLimit:
#     Controls how aggressively contrast is boosted in any local tile.
#     Higher = stronger enhancement, but also more noise amplification.
#     2.0 is a standard starting point for palm/skin images.
#
#   tileGridSize:
#     The image is divided into a grid of tiles. Each tile is equalized
#     independently, then blended with its neighbours.
#     (8, 8) means 8×8 pixel tiles. Smaller tiles = more local enhancement.
#     (8, 8) is a typical default; we may tune this later.
#
# Then clahe.apply(gray_image) runs the enhancement.
# INPUT:  Single-channel (grayscale) uint8 image.
# OUTPUT: Grayscale uint8 image with locally enhanced contrast.

print("\n─── Step 4: CLAHE ───────────────────────────────────────────────────")

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
image_clahe = clahe.apply(image_gray)

print(f"  clipLimit   : 2.0  (controls noise suppression — tune later)")
print(f"  tileGridSize: (8, 8)  (local tile size for adaptive processing)")
print(f"  Before CLAHE — min: {image_gray.min()}, max: {image_gray.max()}, mean: {image_gray.mean():.1f}")
print(f"  After  CLAHE — min: {image_clahe.min()},   max: {image_clahe.max()},   mean: {image_clahe.mean():.1f}")
print("  → CLAHE enhances local regions; values stay within 0–255")

# =============================================================================
# STEP 5 — SAVE INDIVIDUAL PROCESSED IMAGES
# =============================================================================
# cv2.imwrite(filepath, image) saves a NumPy array as an image file.
# The extension in the filename (.jpg, .png) determines the format.
#
# We save each experiment separately in data/processed/ with clear names.
# This is good practice: never overwrite your originals.

print("\n─── Step 5: Saving Processed Images ────────────────────────────────")

# Make sure the output folder exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

def save_image(img, filename):
    """
    Save an image to the processed directory and print confirmation.
    
    Parameters:
      img      : NumPy array (the image to save)
      filename : string (just the filename, not full path)
    """
    path = os.path.join(PROCESSED_DIR, filename)
    success = cv2.imwrite(path, img)
    status = "✅" if success else "❌"
    print(f"  {status} {filename}  →  {path}")
    return success

save_image(image_gray,  "palm_grayscale.jpg")
save_image(image_eq,    "palm_hist_eq.jpg")
save_image(image_clahe, "palm_clahe.jpg")

# =============================================================================
# STEP 6 — BUILD A COMPARISON GRID
# =============================================================================
# We want to display all four versions (original, gray, eq, clahe) side by side.
#
# PROBLEM: The original is BGR (3 channels); the others are grayscale (1 channel).
# We cannot horizontally concatenate arrays with different channel counts.
#
# SOLUTION: Convert all grayscale images back to 3-channel BGR using
# cv2.COLOR_GRAY2BGR. This does NOT add colour — it just copies the single
# channel into all three slots so the shape is (H, W, 3) again.
# That makes concatenation possible.
#
# numpy.hstack([a, b, c, d]) stacks arrays horizontally (side by side).
# All arrays must have the same height and dtype.
#
# We also resize the images for the comparison view, since your original
# is 3046×3736 which is enormous. A 400px height makes a readable grid.

print("\n─── Step 6: Building Comparison Grid ───────────────────────────────")

COMPARE_HEIGHT = 400   # pixels — height of each panel in the comparison

def resize_to_height(img, target_h):
    """
    Resize an image to a target height, maintaining aspect ratio.
    
    cv2.resize(src, (width, height), interpolation=...)
      INTER_AREA is the best interpolation method when SHRINKING images.
      It averages pixel regions, preserving overall appearance.
      (INTER_LINEAR or INTER_CUBIC are better when ENLARGING.)
    """
    h, w = img.shape[:2]                          # current height, width
    scale     = target_h / h                      # how much to shrink
    new_w     = int(w * scale)                    # maintain aspect ratio
    resized   = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)
    return resized

# Resize all four versions
orig_small  = resize_to_height(image_bgr,   COMPARE_HEIGHT)
gray_small  = resize_to_height(image_gray,  COMPARE_HEIGHT)
eq_small    = resize_to_height(image_eq,    COMPARE_HEIGHT)
clahe_small = resize_to_height(image_clahe, COMPARE_HEIGHT)

# Convert grayscale panels to 3-channel so hstack can combine them
gray_3ch  = cv2.cvtColor(gray_small,  cv2.COLOR_GRAY2BGR)
eq_3ch    = cv2.cvtColor(eq_small,    cv2.COLOR_GRAY2BGR)
clahe_3ch = cv2.cvtColor(clahe_small, cv2.COLOR_GRAY2BGR)

# Add text labels to each panel
# cv2.putText(img, text, origin, font, scale, color, thickness)
# origin = (x, y) of the bottom-left corner of the text
def add_label(img, label):
    """Overlay a white label with dark border onto an image panel."""
    labeled = img.copy()
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness  = 2
    color      = (255, 255, 255)   # white text
    shadow     = (0, 0, 0)         # black shadow for readability

    pos = (10, 30)
    # Draw shadow first (offset by 1 pixel)
    cv2.putText(labeled, label, (pos[0]+1, pos[1]+1), font, font_scale, shadow, thickness + 1)
    # Draw white text on top
    cv2.putText(labeled, label, pos, font, font_scale, color, thickness)
    return labeled

orig_panel  = add_label(orig_small,  "Original (BGR)")
gray_panel  = add_label(gray_3ch,    "Grayscale")
eq_panel    = add_label(eq_3ch,      "Hist. Equalization")
clahe_panel = add_label(clahe_3ch,   "CLAHE (clip=2.0)")

# Concatenate all four panels side by side
comparison = np.hstack([orig_panel, gray_panel, eq_panel, clahe_panel])

# Save the comparison grid
comp_path = os.path.join(PROCESSED_DIR, "palm_comparison.jpg")
cv2.imwrite(comp_path, comparison)
print(f"  ✅ Comparison grid  →  {comp_path}")
print(f"  Grid size: {comparison.shape[1]}×{comparison.shape[0]} pixels (W×H)")

# =============================================================================
# STEP 7 — DISPLAY THE COMPARISON GRID
# =============================================================================
print("\n─── Step 7: Displaying Comparison ──────────────────────────────────")
print("  Opening comparison window...")
print("  👉 Press any key to close.\n")

cv2.imshow("PalmVerse — Grayscale + Contrast Comparison", comparison)
cv2.waitKey(0)
cv2.destroyAllWindows()

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 65)
print("  ✅ Milestone 1.2 Complete!")
print("=" * 65)
print("\n  Files saved to data/processed/:")
print("    palm_grayscale.jpg  — single-channel brightness image")
print("    palm_hist_eq.jpg    — globally equalised contrast")
print("    palm_clahe.jpg      — locally adaptive contrast (CLAHE)")
print("    palm_comparison.jpg — side-by-side comparison grid")

print("""
📚 What you learned:
   1. cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) collapses 3 channels → 1
   2. Grayscale shape is (H, W) — no channel dimension
   3. cv2.equalizeHist() boosts contrast globally across the whole image
   4. cv2.createCLAHE() boosts contrast locally per tile — better for
      uneven lighting (like palm images shot in natural light)
   5. cv2.COLOR_GRAY2BGR lets you "promote" a grayscale image to 3
      channels so you can combine it with colour images in a grid
   6. np.hstack() stacks arrays side-by-side (same height required)
   7. cv2.resize() with INTER_AREA is the right choice when shrinking
   8. cv2.imwrite() saves any NumPy array as an image file
""")
print("=" * 65)
