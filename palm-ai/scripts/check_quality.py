"""
check_quality.py — Phase 1, Milestone 1.3: Basic Image Quality Assessment

WHAT WE ARE BUILDING:
  A script that evaluates whether a palm image is "good enough" to
  feed into the CV/ML pipeline. It checks resolution, brightness,
  and sharpness, then prints a clear PASS / WARNING / ACCEPT decision.

WHY WE ARE BUILDING IT:
  In the final app, users will upload their own photos — taken in
  different lighting, with different phones, at different distances.
  Some will be too dark, too blurry, or too small to process reliably.
  We need to catch these problems BEFORE they silently corrupt our
  CV or ML results.

WHAT YOU ARE LEARNING:
  - Image resolution as pixel count
  - Mean pixel value as a brightness estimate
  - Laplacian variance as a sharpness/blur metric
  - How to set and apply threshold-based quality gates
  - Why these thresholds are heuristics (educated guesses to tune)

HOW TO RUN:
  From the palm-ai/ folder, with .venv active:
    python scripts/check_quality.py
"""

import cv2
import numpy as np
import sys
import os

# ── Import our central config ─────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import RAW_IMAGES_DIR

# =============================================================================
# QUALITY THRESHOLDS — Change these values here to tune the checker.
#
# These are heuristics: reasonable starting values based on common
# phone photography. As we see more palm images, we will adjust them.
#
# WHY THEY ARE AT THE TOP:
#   If thresholds were buried in the middle of the code, you'd have to
#   hunt for them when tuning. Keeping config at the top is a clean habit.
# =============================================================================

# --- Resolution ---------------------------------------------------------------
# We need enough pixels to see fine palm line detail.
# A 300×300 thumbnail is too small. 400×400 is a conservative minimum.
MIN_WIDTH      = 400    # pixels
MIN_HEIGHT     = 400    # pixels

# --- Brightness ---------------------------------------------------------------
# These are applied to the grayscale mean (0–255).
# Below MIN_BRIGHTNESS → too dark, detail hidden in shadow.
# Above MAX_BRIGHTNESS → too bright, overexposed, detail blown out.
MIN_BRIGHTNESS = 60     # mean pixel value — below this = too dark
MAX_BRIGHTNESS = 220    # mean pixel value — above this = too bright/overexposed

# --- Blur (Laplacian Variance) ------------------------------------------------
# This is the trickiest threshold because it depends on image resolution.
# A high-resolution sharp image will have a much higher score than a
# low-resolution sharp image. We will refine this in Phase 3.
# For now, < 50 is a practical "likely blurry" threshold for phone photos.
MIN_BLUR_SCORE = 50     # Laplacian variance — below this = likely blurry


# =============================================================================
# HELPER: Print a formatted check result
# =============================================================================

def print_check(label, value, passed, detail=""):
    """
    Print one quality check result in a consistent format.

    Parameters:
      label  : name of the check (e.g. "Width")
      value  : the measured value (e.g. "3736 px")
      passed : True = PASS (green-ish), False = WARNING
      detail : optional extra context printed after the result
    """
    status = "✅ PASS   " if passed else "⚠️  WARNING"
    print(f"  {status}  {label:<22} {str(value):<20}  {detail}")


# =============================================================================
# MAIN SCRIPT
# =============================================================================

IMAGE_FILENAME = "test_palm.jpg"
IMAGE_PATH     = os.path.join(RAW_IMAGES_DIR, IMAGE_FILENAME)

print("=" * 65)
print("  PalmVerse — Phase 1, Milestone 1.3: Image Quality Check")
print("=" * 65)
print(f"\n📂 Checking: {IMAGE_PATH}\n")

# ── Step 1: Load the image ────────────────────────────────────────────────────
if not os.path.exists(IMAGE_PATH):
    print(f"❌ File not found: {IMAGE_PATH}")
    sys.exit(1)

image_bgr = cv2.imread(IMAGE_PATH)

if image_bgr is None:
    print("❌ OpenCV could not read this file (may be corrupted or unsupported format).")
    sys.exit(1)

# ── Step 2: Convert to grayscale ──────────────────────────────────────────────
# All quality metrics below work on brightness, so we use grayscale.
# We already know this conversion from Milestone 1.2.
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

# ── Step 3: Extract basic measurements ───────────────────────────────────────
height, width = image_gray.shape   # grayscale: shape is (H, W), no channels

total_pixels = height * width
file_size_kb = os.path.getsize(IMAGE_PATH) / 1024   # bytes → kilobytes

print("─── Raw Measurements ────────────────────────────────────────────")
print(f"  Width        : {width} px")
print(f"  Height       : {height} px")
print(f"  Total pixels : {total_pixels:,}  ({total_pixels / 1_000_000:.1f} megapixels)")
print(f"  File size    : {file_size_kb:.1f} KB  ({file_size_kb/1024:.2f} MB)")

# ── Step 4: Brightness ────────────────────────────────────────────────────────
# numpy's .mean() computes the average of every element in the array.
# Since image_gray is a 2D array of values 0–255, this gives us the
# average pixel brightness across the entire image.
#
# np.mean() returns a float64 — we round it for readability.
mean_brightness = np.mean(image_gray)

# Standard deviation tells us the spread of brightness values.
# A low std = many pixels at similar brightness = flat/low-contrast image.
# A high std = wide range of brightnesses = more detail/contrast.
std_brightness = np.std(image_gray)

print(f"\n  Mean brightness : {mean_brightness:.1f}  (0=black, 255=white)")
print(f"  Std deviation   : {std_brightness:.1f}  (spread of brightness values)")
print(f"  Min pixel value : {int(image_gray.min())}")
print(f"  Max pixel value : {int(image_gray.max())}")

# ── Step 5: Blur detection via Laplacian Variance ─────────────────────────────
#
# HOW THE LAPLACIAN WORKS:
#   The Laplacian is a second-order derivative filter. It measures how quickly
#   brightness is changing in all directions at each pixel.
#   At an edge, brightness changes sharply → large Laplacian value.
#   In a smooth/blurry region, brightness changes slowly → small value.
#
# cv2.Laplacian(src, ddepth):
#   src    = the source image (grayscale)
#   ddepth = output image depth (bit depth)
#            cv2.CV_64F means 64-bit float
#            We use float because the Laplacian can produce negative values
#            (a uint8 would clip negatives to 0, losing information)
#
# .var() computes the statistical variance of all Laplacian values.
#   Variance = average of squared deviations from the mean.
#   High variance → lots of strong edges → SHARP image
#   Low variance  → weak, few edges      → BLURRY image
#
# This is one of the most reliable simple blur metrics in computer vision.
# It was proposed by Pech-Pacheco et al. (2000) and is widely used.

laplacian         = cv2.Laplacian(image_gray, cv2.CV_64F)
blur_score        = laplacian.var()

# Also collect the min/max of Laplacian for insight
lap_min = laplacian.min()
lap_max = laplacian.max()

print(f"\n  Laplacian variance (blur score) : {blur_score:.2f}")
print(f"  Laplacian min / max             : {lap_min:.1f} / {lap_max:.1f}")
print(f"  (Higher variance = sharper image; lower = more blurry)")

# ── Step 6: Run quality checks ────────────────────────────────────────────────
print("\n─── Quality Gate Results ────────────────────────────────────────────")
print(f"  {'Check':<22} {'Value':<20}  Result")
print("  " + "-" * 60)

# Each check compares a measured value against a threshold.
# We collect True/False results so we can compute the overall decision.
checks = []

# CHECK 1 — Minimum width
passed = width >= MIN_WIDTH
checks.append(passed)
print_check("Width", f"{width} px",
            passed,
            f"(min: {MIN_WIDTH} px)")

# CHECK 2 — Minimum height
passed = height >= MIN_HEIGHT
checks.append(passed)
print_check("Height", f"{height} px",
            passed,
            f"(min: {MIN_HEIGHT} px)")

# CHECK 3 — Not too dark
passed = mean_brightness >= MIN_BRIGHTNESS
checks.append(passed)
print_check("Brightness (low)",
            f"{mean_brightness:.1f}",
            passed,
            f"(min: {MIN_BRIGHTNESS}  — below = too dark)")

# CHECK 4 — Not too bright / overexposed
passed = mean_brightness <= MAX_BRIGHTNESS
checks.append(passed)
print_check("Brightness (high)",
            f"{mean_brightness:.1f}",
            passed,
            f"(max: {MAX_BRIGHTNESS}  — above = overexposed)")

# CHECK 5 — Not too blurry
passed = blur_score >= MIN_BLUR_SCORE
checks.append(passed)
print_check("Sharpness (blur)",
            f"{blur_score:.1f}",
            passed,
            f"(min: {MIN_BLUR_SCORE}  — below = too blurry)")

# ── Step 7: Overall quality decision ─────────────────────────────────────────
# all(checks) returns True only if EVERY check passed.
# One failure = overall rejection.
print("\n─── Overall Decision ────────────────────────────────────────────────")

all_passed = all(checks)
failed_count = checks.count(False)

if all_passed:
    print("\n  ✅✅  ACCEPT — Image passes all quality checks.")
    print("       Safe to proceed with hand detection and preprocessing.")
else:
    print(f"\n  ❌   NEEDS BETTER IMAGE — {failed_count} check(s) failed.")
    print("       Fix the flagged issues before running the CV pipeline.")

# ── Step 8: Actionable advice for common failures ─────────────────────────────
print("\n─── Guidance ────────────────────────────────────────────────────────")
if mean_brightness < MIN_BRIGHTNESS:
    print("  💡 Image is too dark: take the photo in better lighting,")
    print("     or move closer to a light source.")
if mean_brightness > MAX_BRIGHTNESS:
    print("  💡 Image is overexposed: avoid direct sunlight or flash")
    print("     pointing directly at the palm.")
if blur_score < MIN_BLUR_SCORE:
    print("  💡 Image is blurry: hold the camera steady, ensure")
    print("     autofocus has locked, and retake the photo.")
if width < MIN_WIDTH or height < MIN_HEIGHT:
    print("  💡 Image is too small: use the full camera resolution,")
    print("     and make sure the palm fills most of the frame.")
if all_passed:
    print("  All checks passed — no action required.")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  ✅ Milestone 1.3 Complete!")
print("=" * 65)
print("""
📚 What you learned:
   1. Resolution = width × height in pixels — more pixels = more detail
   2. Mean brightness = np.mean(gray_image) — average of all pixel values
   3. Laplacian variance measures sharpness — sharp images have high
      variance because edges produce large Laplacian responses
   4. cv2.Laplacian(img, cv2.CV_64F) must use float output to preserve
      negative values (a uint8 would clip them and lose edge information)
   5. Quality thresholds are heuristics — educated starting points
      that must be tuned as you see more real-world images
   6. Validating quality BEFORE inference prevents silent bad outputs
""")
print("=" * 65)
