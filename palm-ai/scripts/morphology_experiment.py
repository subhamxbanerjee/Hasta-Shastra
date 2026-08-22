"""
morphology_experiment.py — Phase 2, Milestone 2.2
Gaussian Blur + Morphological Operations

WHAT WE ARE BUILDING:
  A pipeline that takes a palm image through:
    grayscale → CLAHE → Gaussian Blur → Threshold → Morphological ops
  We compare blur kernel sizes side-by-side, then compare the four
  morphological operations (erosion, dilation, opening, closing)
  on a binary thresholded image.

WHY WE ARE BUILDING IT:
  Real palm images contain noise, skin texture, and shadows that
  create unwanted structures in a binary threshold result.
  Gaussian blur reduces noise BEFORE thresholding.
  Morphological operations clean up the binary result AFTER thresholding.
  Together they are the classical pre/post-processing stack for
  binary image analysis.

IMPORTANT HONESTY:
  These operations manipulate pixel structures based on shape and size.
  They do NOT understand what a Life Line, Head Line, or Heart Line is.
  We are learning the tools — we are not solving the segmentation problem.

WHAT YOU ARE LEARNING:
  - cv2.GaussianBlur()           — weighted neighbourhood smoothing
  - How kernel size and sigma affect blur strength
  - cv2.getStructuringElement()  — defining the morphology neighbourhood
  - cv2.erode()                  — shrinks white regions
  - cv2.dilate()                 — grows white regions
  - cv2.morphologyEx(MORPH_OPEN) — erosion then dilation
  - cv2.morphologyEx(MORPH_CLOSE)— dilation then erosion

  PERFORMANCE NOTE:
  Your image is 11.4 megapixels. Morphological operations on full-res
  images are slow. We resize to a working resolution before processing —
  this is standard practice in production CV pipelines.

HOW TO RUN:
  From the palm-ai/ folder, with .venv active:
    python scripts/morphology_experiment.py
"""

import cv2
import numpy as np
import sys
import os

# ── Central config ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import RAW_IMAGES_DIR, MORPHOLOGY_DIR

# =============================================================================
# PARAMETERS — Tune these to explore different effects.
# =============================================================================

IMAGE_FILENAME = "test_palm.jpg"

# ── Working resolution ────────────────────────────────────────────────────────
# We resize the image to this width before processing.
# Reason: morphological operations on 11.4MP images are very slow.
# 1024px wide is large enough to see fine palm line detail while being
# fast enough to experiment interactively.
# In Phase 3, MediaPipe will give us a cropped palm region at our
# chosen resolution — this manual resize is a placeholder for that.
WORKING_WIDTH = 1024

# ── Gaussian Blur experiments ─────────────────────────────────────────────────
# We run two blur kernels so you can compare the difference visually.
# Kernel size MUST be ODD and positive.
# sigmaX=0 → OpenCV auto-computes sigma from kernel size.
BLUR_KERNEL_SMALL = (5, 5)    # subtle — removes fine noise, preserves edges
BLUR_KERNEL_LARGE = (15, 15)  # strong — may soften line edges

# ── Threshold settings ────────────────────────────────────────────────────────
# We use Adaptive Gaussian after blur — it handles remaining illumination
# variation better than global Otsu for palm images.
THRESH_BLOCK_SIZE = 35   # local neighbourhood (ODD number)
THRESH_C          = 8    # constant subtracted from local mean

# ── Morphology settings ───────────────────────────────────────────────────────
# MORPH_SHAPE options: cv2.MORPH_RECT | MORPH_ELLIPSE | MORPH_CROSS
MORPH_SHAPE      = cv2.MORPH_ELLIPSE
MORPH_KSIZE      = (3, 3)   # structuring element size — try (3,3), (5,5)
MORPH_ITERATIONS = 1        # how many times to apply erosion/dilation

# ── Display ───────────────────────────────────────────────────────────────────
PANEL_HEIGHT = 400    # height of each panel in comparison grids

# =============================================================================
# SETUP
# =============================================================================
IMAGE_PATH = os.path.join(RAW_IMAGES_DIR, IMAGE_FILENAME)

print("=" * 68)
print("  PalmVerse — Phase 2, Milestone 2.2: Blur + Morphology")
print("=" * 68)
print(f"\n  Working width      : {WORKING_WIDTH} px")
print(f"  Blur kernels       : {BLUR_KERNEL_SMALL}  and  {BLUR_KERNEL_LARGE}")
print(f"  Threshold block    : {THRESH_BLOCK_SIZE},  C={THRESH_C}")
print(f"  Morph shape/ksize  : ELLIPSE {MORPH_KSIZE},  iterations={MORPH_ITERATIONS}\n")

os.makedirs(MORPHOLOGY_DIR, exist_ok=True)

# =============================================================================
# STEP 1 — LOAD + RESIZE TO WORKING RESOLUTION
# =============================================================================
# WHY RESIZE FIRST:
#   Every operation below runs on every pixel. For an 11MP image that is
#   11,379,856 operations per step. At 1024px wide it is ~640,000 — 18× faster.
#   We always resize with INTER_AREA when shrinking (best quality).
#
# We record the scale factor so we can describe it in output messages.

print("─── Step 1: Load + Resize ───────────────────────────────────────────")

if not os.path.exists(IMAGE_PATH):
    print(f"❌ File not found: {IMAGE_PATH}")
    sys.exit(1)

image_bgr = cv2.imread(IMAGE_PATH)
if image_bgr is None:
    print("❌ OpenCV could not read the image.")
    sys.exit(1)

orig_h, orig_w = image_bgr.shape[:2]
scale_factor   = WORKING_WIDTH / orig_w
working_h      = int(orig_h * scale_factor)

image_bgr = cv2.resize(
    image_bgr, (WORKING_WIDTH, working_h), interpolation=cv2.INTER_AREA
)

print(f"  Original size : {orig_w} × {orig_h} px")
print(f"  Working size  : {WORKING_WIDTH} × {working_h} px  (scale: {scale_factor:.3f})")
print(f"  ✅ Image loaded and resized.")

# =============================================================================
# STEP 2 — GRAYSCALE + CLAHE
# =============================================================================
# Standard pipeline from Phase 1 — you should recognise this now.
# CLAHE normalises local contrast BEFORE blurring.
# Order matters: CLAHE enhances local detail; blur then softens fine noise.
# If you blurred first, CLAHE would re-enhance the blurred (noisy) image.

print("\n─── Step 2: Grayscale + CLAHE ───────────────────────────────────────")

image_gray  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
clahe       = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
image_clahe = clahe.apply(image_gray)

print(f"  Grayscale shape : {image_gray.shape}")
print(f"  CLAHE applied   : clipLimit=2.0, tileGridSize=(8,8)")

# =============================================================================
# STEP 3 — GAUSSIAN BLUR (two kernel sizes for comparison)
# =============================================================================
# cv2.GaussianBlur(src, ksize, sigmaX, sigmaY=0)
#
#   src    : input image (grayscale here)
#   ksize  : (width, height) of the kernel — MUST be ODD positive integers
#   sigmaX : standard deviation in X direction
#            0 → computed automatically from ksize:
#                σ = 0.3 × ((ksize − 1) × 0.5 − 1) + 0.8
#   sigmaY : if 0, uses sigmaX value
#
# HOW IT WORKS PER PIXEL:
#   For each output pixel, a Gaussian-shaped weight matrix is placed over
#   the input image, centred on that pixel. The output value is the
#   weighted average of all covered input pixels. Centre-heavy weighting
#   means nearby pixels matter more than far ones — this preserves more
#   structural shape than a simple box average.
#
# WHAT CHANGES WITH KERNEL SIZE:
#   Larger kernel → more pixels included in the average → stronger blur.
#   (3,3)  → barely visible blur
#   (5,5)  → gentle — removes fine grain, preserves edges
#   (15,15)→ noticeable softening — wider edges, some line loss
#   (25,25)→ heavy — palm lines may begin to disappear

print("\n─── Step 3: Gaussian Blur (two experiments) ─────────────────────────")

blur_small = cv2.GaussianBlur(image_clahe, BLUR_KERNEL_SMALL, sigmaX=0)
blur_large = cv2.GaussianBlur(image_clahe, BLUR_KERNEL_LARGE, sigmaX=0)

print(f"  Blur A (small kernel) : {BLUR_KERNEL_SMALL}, sigmaX=auto")
print(f"  Blur B (large kernel) : {BLUR_KERNEL_LARGE}, sigmaX=auto")

# Pixel-level difference between blur versions — shows what was removed
diff_blur = cv2.absdiff(blur_small, blur_large)
print(f"  Mean pixel difference A→B : {diff_blur.mean():.2f}")
print("  (Higher = larger kernel removed more detail from the image)")

# =============================================================================
# STEP 4 — THRESHOLD TO BINARY (on the small-kernel blur result)
# =============================================================================
# We use small-kernel blur as the base for morphology.
# Reason: we want to PRESERVE fine palm line detail while removing only
# the finest noise. A (5,5) kernel achieves this balance.
#
# We use Adaptive Gaussian (from Milestone 2.1) with THRESH_BINARY_INV
# so that dark palm lines → WHITE foreground, background → BLACK.
# White foreground = the convention expected by morphological operations.
#
# Remember from Milestone 2.1:
#   THRESH_BINARY_INV: pixel < local_threshold → 255 (white)
#                      pixel ≥ local_threshold → 0   (black)

print("\n─── Step 4: Adaptive Gaussian Threshold → Binary ────────────────────")

binary = cv2.adaptiveThreshold(
    blur_small,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    THRESH_BLOCK_SIZE,
    THRESH_C
)

white_pixels = int(np.sum(binary == 255))
total_pixels = binary.size
white_pct    = 100 * white_pixels / total_pixels

print(f"  Method     : Adaptive Gaussian")
print(f"  Block size : {THRESH_BLOCK_SIZE},  C = {THRESH_C}")
print(f"  White pixels (foreground) : {white_pixels:,}  ({white_pct:.1f}% of image)")
print("  ✅ Binary image ready for morphological operations.")

# =============================================================================
# STEP 5 — STRUCTURING ELEMENT
# =============================================================================
# cv2.getStructuringElement(shape, ksize) → a small NumPy matrix.
#
# The structuring element is the "brush" used during morphology.
# During erosion/dilation, it is placed over each pixel and determines
# which neighbours are considered in the operation.
#
# MORPH_RECT     → all cells in the rectangle are 1 (square brush)
# MORPH_ELLIPSE  → cells within an inscribed ellipse are 1 (round brush)
# MORPH_CROSS    → only the row and column through the centre are 1
#
# WHY ELLIPSE FOR PALM IMAGES:
#   Palm lines are curved, not square. An elliptical kernel applies the
#   operation more gently at the diagonal corners, producing smoother
#   results on curved structures than a sharp rectangular kernel.
#
# Let's print the kernel so you can see what it actually looks like.

print("\n─── Step 5: Structuring Element ─────────────────────────────────────")

kernel = cv2.getStructuringElement(MORPH_SHAPE, MORPH_KSIZE)

shape_name = {
    cv2.MORPH_RECT:    "RECT",
    cv2.MORPH_ELLIPSE: "ELLIPSE",
    cv2.MORPH_CROSS:   "CROSS",
}.get(MORPH_SHAPE, "UNKNOWN")

print(f"  Shape : {shape_name},  Size : {MORPH_KSIZE}")
print(f"  Kernel matrix (1=active, 0=inactive):")
for row in kernel:
    print("    " + "  ".join(str(v) for v in row))

# =============================================================================
# STEP 6 — MORPHOLOGICAL OPERATIONS
# =============================================================================

print("\n─── Step 6: Morphological Operations ───────────────────────────────")

# ── EROSION ──────────────────────────────────────────────────────────────────
# cv2.erode(src, kernel, iterations=N)
#
# HOW IT WORKS:
#   For each white pixel, place the structuring element centred on it.
#   If ALL pixels covered by the kernel's '1' cells are white → keep white.
#   If ANY covered pixel is black → this pixel becomes black.
#
# VISUAL EFFECT on binary palm image:
#   - Thin white structures (noise speckles, thin lines) disappear
#   - Thicker white structures shrink at their boundaries
#   - Small isolated white blobs (< kernel size) are completely removed
#   - Fine palm lines may thin out or break if kernel is too large
#
# iterations=1: apply once. iterations=2: apply twice (stronger effect).

eroded = cv2.erode(binary, kernel, iterations=MORPH_ITERATIONS)
eroded_white = int(np.sum(eroded == 255))
print(f"  Erosion   : {eroded_white:,} white px  (was {white_pixels:,})  "
      f"→ removed {white_pixels - eroded_white:,} px")

# ── DILATION ─────────────────────────────────────────────────────────────────
# cv2.dilate(src, kernel, iterations=N)
#
# HOW IT WORKS:
#   For each black pixel, place the structuring element centred on it.
#   If ANY pixel covered by the kernel's '1' cells is white → this becomes white.
#   Equivalently: any white pixel "infects" its neighbours.
#
# VISUAL EFFECT on binary palm image:
#   - White structures grow outward — their boundaries expand
#   - Small gaps between nearby white structures get filled
#   - Fine details that are close together may merge into one blob
#   - Useful to "fatten" thin detected line segments

dilated = cv2.dilate(binary, kernel, iterations=MORPH_ITERATIONS)
dilated_white = int(np.sum(dilated == 255))
print(f"  Dilation  : {dilated_white:,} white px  (was {white_pixels:,})  "
      f"→ added {dilated_white - white_pixels:,} px")

# ── OPENING = EROSION then DILATION ──────────────────────────────────────────
# cv2.morphologyEx(src, cv2.MORPH_OPEN, kernel, iterations=N)
#
# HOW IT WORKS:
#   1. Erode  → removes small speckles and thins structures
#   2. Dilate → restores the remaining (large enough) structures to near original size
#
# NET EFFECT:
#   Small isolated white blobs smaller than the kernel disappear.
#   Larger white structures (like palm line segments) survive mostly intact.
#   This is the primary tool for NOISE REMOVAL on binary images.
#
# Think of it as: "remove everything too small to be a meaningful structure."

opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=MORPH_ITERATIONS)
opened_white = int(np.sum(opened == 255))
print(f"  Opening   : {opened_white:,} white px  (was {white_pixels:,})  "
      f"→ net change {opened_white - white_pixels:+,} px  [erode→dilate]")

# ── CLOSING = DILATION then EROSION ──────────────────────────────────────────
# cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel, iterations=N)
#
# HOW IT WORKS:
#   1. Dilate → grows structures, fills small holes and gaps
#   2. Erode  → shrinks back to near original boundaries
#
# NET EFFECT:
#   Small black holes inside white regions get filled.
#   Thin gaps between nearby white regions get closed.
#   Useful when a palm line is detected as a dashed or broken line —
#   closing can reconnect the gaps.
#
# Think of it as: "fill gaps too small to be meaningful background."

closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITERATIONS)
closed_white = int(np.sum(closed == 255))
print(f"  Closing   : {closed_white:,} white px  (was {white_pixels:,})  "
      f"→ net change {closed_white - white_pixels:+,} px  [dilate→erode]")

# =============================================================================
# STEP 7 — SAVE INDIVIDUAL RESULTS
# =============================================================================
print("\n─── Step 7: Saving Results ──────────────────────────────────────────")

def save(img, name):
    path = os.path.join(MORPHOLOGY_DIR, name)
    ok   = cv2.imwrite(path, img)
    print(f"  {'✅' if ok else '❌'}  {name}")

# Save pipeline stages
save(image_clahe, "00_clahe.jpg")
save(blur_small,  f"01_blur_{BLUR_KERNEL_SMALL[0]}x{BLUR_KERNEL_SMALL[1]}.jpg")
save(blur_large,  f"02_blur_{BLUR_KERNEL_LARGE[0]}x{BLUR_KERNEL_LARGE[1]}.jpg")
save(diff_blur,   "03_blur_difference.jpg")
save(binary,      "04_binary.jpg")
save(eroded,      "05_eroded.jpg")
save(dilated,     "06_dilated.jpg")
save(opened,      "07_opened.jpg")
save(closed,      "08_closed.jpg")

# =============================================================================
# STEP 8 — COMPARISON GRIDS
# =============================================================================
print("\n─── Step 8: Building Comparison Grids ───────────────────────────────")

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
THICKNESS  = 2

def make_panel(img, label, h=PANEL_HEIGHT):
    """Resize to target height, ensure 3-channel, add white text label."""
    oh, ow = img.shape[:2]
    nw     = int(ow * h / oh)
    small  = cv2.resize(img, (nw, h), interpolation=cv2.INTER_AREA)
    if small.ndim == 2:                           # grayscale → 3 channel
        small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    cv2.putText(small, label, (11, 31), FONT, FONT_SCALE, (0,0,0),       THICKNESS+2)
    cv2.putText(small, label, (10, 30), FONT, FONT_SCALE, (255,255,255), THICKNESS)
    return small

def hstack_panels(panels):
    """Stack panels horizontally, trimming to the minimum width."""
    min_w = min(p.shape[1] for p in panels)
    return np.hstack([p[:, :min_w] for p in panels])

# ── Grid 1: Blur comparison ───────────────────────────────────────────────────
# Three panels: CLAHE base | small blur | large blur
# This lets you see exactly what each blur level removes.
blur_grid = hstack_panels([
    make_panel(image_clahe, "CLAHE (no blur)"),
    make_panel(blur_small,  f"Blur {BLUR_KERNEL_SMALL[0]}x{BLUR_KERNEL_SMALL[1]} (small)"),
    make_panel(blur_large,  f"Blur {BLUR_KERNEL_LARGE[0]}x{BLUR_KERNEL_LARGE[1]} (large)"),
])
save(blur_grid, "GRID_A_blur_comparison.jpg")
print(f"  ✅ Grid A: Blur comparison")

# ── Grid 2: Morphology comparison ─────────────────────────────────────────────
# Five panels: binary | eroded | dilated | opened | closed
morph_grid = hstack_panels([
    make_panel(binary,  "Binary (base)"),
    make_panel(eroded,  "Eroded"),
    make_panel(dilated, "Dilated"),
    make_panel(opened,  "Opened"),
    make_panel(closed,  "Closed"),
])
save(morph_grid, "GRID_B_morphology_comparison.jpg")
print(f"  ✅ Grid B: Morphology comparison")

# =============================================================================
# STEP 9 — DISPLAY BOTH GRIDS
# =============================================================================
print("\n─── Step 9: Display ─────────────────────────────────────────────────")
print("  Two windows will open — press any key in each to close.\n")

cv2.imshow("PalmVerse — Grid A: Blur Comparison", blur_grid)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("PalmVerse — Grid B: Morphology Comparison", morph_grid)
cv2.waitKey(0)
cv2.destroyAllWindows()

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 68)
print("  ✅ Milestone 2.2 Complete!")
print("=" * 68)
print(f"""
  Pipeline used:
  Original → resize {WORKING_WIDTH}px → gray → CLAHE → GaussianBlur{BLUR_KERNEL_SMALL}
           → AdaptiveGaussian(block={THRESH_BLOCK_SIZE}, C={THRESH_C}) → binary
           → erode / dilate / open / close

  White pixel counts (foreground structures):
    Binary (base) : {white_pixels:,}
    After erode   : {eroded_white:,}   (removed {white_pixels - eroded_white:,} px)
    After dilate  : {dilated_white:,}  (added   {dilated_white - white_pixels:,} px)
    After open    : {opened_white:,}   (net     {opened_white - white_pixels:+,} px)
    After close   : {closed_white:,}   (net     {closed_white - white_pixels:+,} px)

📚 What you learned:
   1. Resize BEFORE heavy operations — always work at the resolution
      you need, not the camera's full resolution
   2. GaussianBlur smooths noise BEFORE thresholding; kernel size
      controls the trade-off between noise removal and edge sharpness
   3. Morphological operations work on binary images only
   4. Structuring element shape affects how boundaries are treated
      (ELLIPSE is gentler on curves than RECT)
   5. Erosion shrinks → removes speckles; Dilation grows → fills gaps
   6. Opening  = erode then dilate → REMOVES small noise blobs
   7. Closing  = dilate then erode → FILLS small holes and gaps
   8. None of these operations understand palm lines — they only
      manipulate pixel shapes and sizes

  IMPORTANT: There is no universally best kernel size.
  The optimal kernel depends on:
    - Your image resolution (larger image = relatively smaller kernel needed)
    - The thickness of palm lines at that resolution
    - The size of the noise you want to remove
    - The gap size you want closing to bridge
  This is why we experiment and inspect visually.
""")
print("=" * 68)
print(f"\n  Outputs saved to: {MORPHOLOGY_DIR}")
