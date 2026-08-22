"""
threshold_experiment.py — Phase 2, Milestone 2.1: Image Thresholding

WHAT WE ARE BUILDING:
  A script that applies six different thresholding methods to our
  palm image, saves each result, and builds a labeled comparison grid.

WHY WE ARE BUILDING IT:
  Thresholding is the classical first step in separating image regions
  by brightness. Understanding it — and its limits — builds the intuition
  for why we later need a trained segmentation model.

  IMPORTANT HONESTY: Thresholding will show us dark structures in the palm,
  but it cannot reliably identify or classify specific palm lines.
  It has no concept of "Life Line" vs "Head Line" vs "shadow".
  This phase is about learning, not solving.

WHAT YOU ARE LEARNING:
  - cv2.threshold()          — global binary / inverse / Otsu thresholding
  - cv2.adaptiveThreshold()  — local neighbourhood-based thresholding
  - Why global methods fail under uneven illumination
  - How Otsu finds an optimal threshold automatically
  - The fundamental limits of brightness-only segmentation

HOW TO RUN:
  From the palm-ai/ folder, with .venv active:
    python scripts/threshold_experiment.py
"""

import cv2
import numpy as np
import sys
import os

# ── Import central config ─────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import RAW_IMAGES_DIR, THRESHOLDS_DIR

# =============================================================================
# PARAMETERS — Edit these at the top; never bury magic numbers in code.
# =============================================================================

IMAGE_FILENAME = "test_palm.jpg"

# ── Global threshold parameters ───────────────────────────────────────────────
# MANUAL_THRESH: the brightness value you choose by hand.
# Pixels above this → white. Below → black (for THRESH_BINARY).
# Try values like 100, 127, 150 and compare results.
MANUAL_THRESH  = 127

# ── Adaptive threshold parameters ─────────────────────────────────────────────
# BLOCK_SIZE: the size (in pixels) of each local neighbourhood.
# Must be an ODD number greater than 1.
# Larger blocks = smoother, less sensitive to fine local variation.
# Smaller blocks = more detail but more noise.
# 51 is a reasonable starting point for high-res palm images.
BLOCK_SIZE     = 51

# C (constant subtracted from the computed local mean):
# Positive C → threshold is LOWER than local mean → more pixels turn white.
# Negative C → threshold is HIGHER than local mean → fewer pixels turn white.
# 10 is a standard starting value. Experiment with 5, 10, 15.
ADAPTIVE_C     = 10

# ── Comparison grid height (per panel, in pixels) ─────────────────────────────
COMPARE_HEIGHT = 420


# =============================================================================
# SETUP
# =============================================================================

IMAGE_PATH = os.path.join(RAW_IMAGES_DIR, IMAGE_FILENAME)

print("=" * 65)
print("  PalmVerse — Phase 2, Milestone 2.1: Thresholding Experiments")
print("=" * 65)
print(f"\n  Manual threshold : {MANUAL_THRESH}")
print(f"  Adaptive block   : {BLOCK_SIZE}×{BLOCK_SIZE} px")
print(f"  Adaptive C       : {ADAPTIVE_C}\n")

os.makedirs(THRESHOLDS_DIR, exist_ok=True)

# =============================================================================
# STEP 1 — LOAD IMAGE
# =============================================================================
if not os.path.exists(IMAGE_PATH):
    print(f"❌ File not found: {IMAGE_PATH}")
    sys.exit(1)

image_bgr = cv2.imread(IMAGE_PATH)
if image_bgr is None:
    print("❌ OpenCV could not read the image.")
    sys.exit(1)

print("✅ Image loaded.")

# =============================================================================
# STEP 2 — PRE-PROCESS BEFORE THRESHOLDING
# =============================================================================
# We apply CLAHE before thresholding for a specific reason:
#
# WHY CLAHE HERE:
#   Palm images typically have uneven illumination across the hand.
#   A global threshold applied to an unprocessed grayscale image will
#   produce poor results in shadowed or overlit regions.
#   CLAHE normalises local contrast so that bright and dark regions of
#   the palm are treated more fairly by the threshold logic.
#
# This is a deliberate preprocessing decision, not just habit.
# In Phase 3 we will also add a palm-crop step before thresholding.

print("\n─── Step 2: Preprocessing ───────────────────────────────────────────")

image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

clahe        = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
image_clahe  = clahe.apply(image_gray)

print(f"  Grayscale shape  : {image_gray.shape}")
print(f"  Applied CLAHE    : clipLimit=2.0, tileGridSize=(8,8)")
print(f"  Mean before CLAHE: {image_gray.mean():.1f}")
print(f"  Mean after CLAHE : {image_clahe.mean():.1f}")
print("  ✅ Using CLAHE image as base for all threshold experiments.")

# =============================================================================
# STEP 3 — THRESHOLDING METHODS
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# METHOD 1: Global Binary Threshold (manual)
# ─────────────────────────────────────────────────────────────────────────────
# cv2.threshold(src, thresh, maxval, type) → (retval, dst)
#
#   src     : input grayscale image
#   thresh  : the threshold value T you choose
#   maxval  : the value assigned to pixels that PASS the threshold (255 = white)
#   type    : THRESH_BINARY → pixel > T becomes maxval, otherwise 0
#
#   Returns a TUPLE: (computed_threshold_used, binary_image)
#   For manual thresholding, retval == thresh (your chosen value).
#   For Otsu, retval == the auto-computed optimal threshold.
#
# The result: every pixel is either 0 (black) or 255 (white).
# No information is preserved between those two values.

print("\n─── Step 3: Applying Threshold Methods ──────────────────────────────")

_, thresh_binary = cv2.threshold(
    image_clahe, MANUAL_THRESH, 255, cv2.THRESH_BINARY
)
print(f"  [1] Binary thresh      : T={MANUAL_THRESH}  →  pixels > {MANUAL_THRESH} → white")

# ─────────────────────────────────────────────────────────────────────────────
# METHOD 2: Global Binary INVERSE
# ─────────────────────────────────────────────────────────────────────────────
# THRESH_BINARY_INV reverses the logic:
#   pixel > T → 0 (BLACK)
#   pixel ≤ T → 255 (WHITE)
#
# WHY THIS MATTERS FOR PALM LINES:
#   Palm lines are darker than surrounding skin.
#   In THRESH_BINARY, dark lines fall BELOW the threshold → they become BLACK.
#   In THRESH_BINARY_INV, dark lines still fall below → they become WHITE.
#   Depending on the downstream task, one orientation may be more useful.
#   Morphological operations, contour detection, and skeletonisation
#   typically expect the FOREGROUND to be white and background black.

_, thresh_binary_inv = cv2.threshold(
    image_clahe, MANUAL_THRESH, 255, cv2.THRESH_BINARY_INV
)
print(f"  [2] Binary inv thresh  : T={MANUAL_THRESH}  →  pixels ≤ {MANUAL_THRESH} → white (inverted)")

# ─────────────────────────────────────────────────────────────────────────────
# METHOD 3: Otsu's Thresholding
# ─────────────────────────────────────────────────────────────────────────────
# We pass thresh=0 (ignored), and add cv2.THRESH_OTSU to the type flags.
# OpenCV computes the optimal T from the histogram.
#
# HOW OTSU WORKS:
#   It assumes the image has two classes of pixels (dark and bright).
#   It finds the T that MINIMISES the weighted sum of variances within
#   each class — equivalently, it MAXIMISES the variance BETWEEN classes.
#   This is called maximising inter-class variance.
#
# The returned retval IS the computed optimal threshold.
# This is the most important return value to check.
#
# LIMITATION: Otsu assumes a bi-modal histogram (two humps — one for
# background, one for foreground). Palm images may not always have this
# clean separation, so Otsu's result won't always be perfect.

otsu_retval, thresh_otsu = cv2.threshold(
    image_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
print(f"  [3] Otsu thresh        : Auto-computed T={otsu_retval:.0f}  (Otsu found this value)")

# ─────────────────────────────────────────────────────────────────────────────
# METHOD 4: Adaptive Mean Threshold
# ─────────────────────────────────────────────────────────────────────────────
# cv2.adaptiveThreshold(src, maxval, method, type, blockSize, C) → dst
#
#   method = cv2.ADAPTIVE_THRESH_MEAN_C
#     For each pixel, the threshold T is:
#       T = mean of the (blockSize × blockSize) neighbourhood − C
#
#   type = cv2.THRESH_BINARY_INV
#     We use inverse so palm lines (dark) → white foreground.
#
#   blockSize = must be ODD. Controls how local the adaptation is.
#   C         = fine-tuning constant subtracted from the local mean.
#               Positive C makes the threshold lower than the local mean,
#               meaning more pixels will exceed T → more white pixels.
#
# WHY MEAN vs GAUSSIAN:
#   Mean treats all neighbours equally within the block.
#   This can cause sharp transitions at block boundaries if there's noise.

thresh_adapt_mean = cv2.adaptiveThreshold(
    image_clahe, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY_INV,
    BLOCK_SIZE, ADAPTIVE_C
)
print(f"  [4] Adaptive Mean      : block={BLOCK_SIZE}, C={ADAPTIVE_C}")

# ─────────────────────────────────────────────────────────────────────────────
# METHOD 5: Adaptive Gaussian Threshold
# ─────────────────────────────────────────────────────────────────────────────
# Same as Adaptive Mean, but the neighbourhood average is WEIGHTED:
#   Pixels closer to the centre of the block contribute MORE to the mean.
#   Pixels at the edges of the block contribute LESS.
#
# The weighting follows a Gaussian (bell-curve) distribution.
#
# WHY THIS IS USUALLY BETTER THAN MEAN FOR PALM IMAGES:
#   Gaussian weighting smooths the threshold transitions across block
#   boundaries, producing less blocky, more natural-looking results.
#   For fine structures like palm lines, Gaussian typically retains
#   more continuity. This is usually the preferred default.
#
# The parameters (blockSize, C) mean the same thing as for mean thresholding.

thresh_adapt_gauss = cv2.adaptiveThreshold(
    image_clahe, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    BLOCK_SIZE, ADAPTIVE_C
)
print(f"  [5] Adaptive Gaussian  : block={BLOCK_SIZE}, C={ADAPTIVE_C}")

# =============================================================================
# STEP 4 — SAVE INDIVIDUAL RESULTS
# =============================================================================
print("\n─── Step 4: Saving Results ──────────────────────────────────────────")

def save(img, name):
    path = os.path.join(THRESHOLDS_DIR, name)
    ok   = cv2.imwrite(path, img)
    print(f"  {'✅' if ok else '❌'}  {name}")
    return path

save(image_clahe,        "00_clahe_base.jpg")
save(thresh_binary,      "01_binary.jpg")
save(thresh_binary_inv,  "02_binary_inv.jpg")
save(thresh_otsu,        "03_otsu.jpg")
save(thresh_adapt_mean,  "04_adaptive_mean.jpg")
save(thresh_adapt_gauss, "05_adaptive_gaussian.jpg")

# =============================================================================
# STEP 5 — BUILD LABELED COMPARISON GRID (2 rows × 3 columns)
# =============================================================================
# All threshold outputs are single-channel (grayscale).
# We convert each to 3-channel BGR so we can add colour labels and hstack/vstack.
#
# np.vstack([row1, row2]) stacks arrays VERTICALLY (one on top of the other).
# Combined with np.hstack for rows, this creates a 2×3 grid.
# Both require matching width; we resize to a fixed height to guarantee this.

print("\n─── Step 5: Building Comparison Grid (2×3) ──────────────────────────")

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
THICKNESS  = 2

def make_panel(gray_img, label, target_h=COMPARE_HEIGHT):
    """
    Resize a grayscale image to target_h, convert to BGR, add a text label.
    Returns a 3-channel BGR panel ready for grid assembly.
    """
    h, w   = gray_img.shape[:2]
    scale  = target_h / h
    new_w  = int(w * scale)
    small  = cv2.resize(gray_img, (new_w, target_h), interpolation=cv2.INTER_AREA)
    panel  = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)

    # Draw shadow for readability on any background
    cv2.putText(panel, label, (11, 31), FONT, FONT_SCALE, (0, 0, 0),     THICKNESS + 2)
    cv2.putText(panel, label, (10, 30), FONT, FONT_SCALE, (255, 255, 255), THICKNESS)
    return panel

# Build six panels
panels = [
    make_panel(image_clahe,        "CLAHE base"),
    make_panel(thresh_binary,      f"Binary  T={MANUAL_THRESH}"),
    make_panel(thresh_binary_inv,  f"Inv.Binary T={MANUAL_THRESH}"),
    make_panel(thresh_otsu,        f"Otsu  T={otsu_retval:.0f} (auto)"),
    make_panel(thresh_adapt_mean,  f"Adapt.Mean  B={BLOCK_SIZE} C={ADAPTIVE_C}"),
    make_panel(thresh_adapt_gauss, f"Adapt.Gaussian B={BLOCK_SIZE} C={ADAPTIVE_C}"),
]

# Ensure all panels have the same width for clean stacking.
# Find the minimum width to avoid mismatches from aspect ratio rounding.
min_w = min(p.shape[1] for p in panels)
panels = [p[:, :min_w] for p in panels]

# Arrange into two rows of three
row1 = np.hstack(panels[:3])
row2 = np.hstack(panels[3:])
grid = np.vstack([row1, row2])

grid_path = os.path.join(THRESHOLDS_DIR, "threshold_comparison.jpg")
cv2.imwrite(grid_path, grid)
print(f"  ✅ threshold_comparison.jpg  ({grid.shape[1]}×{grid.shape[0]} px)")

# =============================================================================
# STEP 6 — DISPLAY
# =============================================================================
print("\n─── Step 6: Display ─────────────────────────────────────────────────")
print("  Opening comparison grid...")
print("  👉 Press any key to close.\n")

cv2.imshow("PalmVerse — Threshold Comparison (6 methods)", grid)
cv2.waitKey(0)
cv2.destroyAllWindows()

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 65)
print("  ✅ Milestone 2.1 Complete!")
print("=" * 65)
print(f"""
  Key result to notice:
  ┌──────────────────────────────────────────────────────────┐
  │  Otsu auto-selected threshold: T = {otsu_retval:.0f}               │
  │  Your manual threshold was:        T = {MANUAL_THRESH}               │
  │  Compare panels 2 and 4 to see the difference.          │
  └──────────────────────────────────────────────────────────┘

  IMPORTANT LESSON:
  Look at all six panels carefully. You will likely see:
    - Dark structures appear in the palm region
    - Many of those dark structures are NOT palm lines:
        shadows, skin texture, hair, noise
    - Adaptive methods preserve more local detail
    - No method can LABEL which structure is the Life Line

  This is why we need semantic segmentation (Phase 6):
  a model that learns the SHAPE and POSITION of each line,
  not just its brightness.

📚 What you learned:
   1. cv2.threshold() → global: one T for the whole image
   2. Otsu: auto-computes the optimal T from the histogram
   3. cv2.adaptiveThreshold() → local T per neighbourhood block
   4. THRESH_BINARY_INV makes dark regions white (good for
      downstream contour detection & morphological ops)
   5. Gaussian adaptive weighting produces smoother results
      than mean weighting for fine structures
   6. CLAHE before thresholding helps uneven illumination
   7. No thresholding method can classify which line is which
""")
print("=" * 65)
print("\n  Outputs saved to:")
print(f"  {THRESHOLDS_DIR}")
