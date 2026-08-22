"""
canny_experiment.py — Phase 2, Milestone 2.3: Canny Edge Detection

WHAT WE ARE BUILDING:
  A script that runs Canny edge detection across multiple blur levels
  and threshold configurations, saves all results, and builds labeled
  comparison grids so we can visually evaluate how parameters change output.

WHY WE ARE BUILDING IT:
  Canny is the most widely used classical edge detection algorithm.
  Understanding how blur and thresholds interact gives us the intuition
  to later evaluate whether a model's predictions contain meaningful edges.
  It also clearly demonstrates why classical CV alone cannot reliably
  segment palm lines — setting the stage for Phase 6 (U-Net segmentation).

IMPORTANT HONESTY:
  Canny detects intensity boundaries — ALL of them.
  It cannot distinguish the Life Line from a shadow, a wrinkle,
  a hair, or the boundary between fingers. We are learning the
  tool and its limits, not solving the palm-line problem here.

WHAT YOU ARE LEARNING:
  - cv2.Canny()         — the full 5-stage edge detection algorithm
  - How Gaussian blur before Canny suppresses noise-driven false edges
  - How threshold1 (low) and threshold2 (high) control edge sensitivity
  - Hysteresis: weak edges survive only if connected to strong edges
  - Why a good threshold pair depends on your specific image content

HOW TO RUN:
  From the palm-ai/ folder, with .venv active:
    python scripts/canny_experiment.py
"""

import cv2
import numpy as np
import sys
import os

# ── Central config ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import RAW_IMAGES_DIR, EDGES_DIR

# =============================================================================
# PARAMETERS — Tune these freely; all important values live here.
# =============================================================================

IMAGE_FILENAME = "test_palm.jpg"

# ── Working resolution ────────────────────────────────────────────────────────
# Same rationale as Milestone 2.2: work at 1024px, not full 11MP.
WORKING_WIDTH  = 1024

# ── Blur kernels ──────────────────────────────────────────────────────────────
# We test three blur levels before Canny:
#   No blur  → maximum edges including all noise
#   Small    → subtle noise reduction, edges mostly intact
#   Large    → stronger smoothing, fewer edges, may lose fine lines
# All sigmaX=0 → OpenCV computes sigma automatically from ksize.
BLUR_NONE      = None       # sentinel: skip GaussianBlur step
BLUR_SMALL     = (5, 5)
BLUR_LARGE     = (11, 11)

# ── Canny threshold experiments ───────────────────────────────────────────────
# Each entry is a dict with a label and the (low, high) threshold pair.
#
# CHOOSING THRESHOLD PAIRS:
#   high threshold = the minimum gradient magnitude to be a strong edge.
#   low  threshold = the minimum gradient for a weak edge (needs strong neighbour).
#   A common starting rule: high = 2× to 3× low.
#
# LOW SENSITIVITY (high thresholds):
#   Only the strongest brightness boundaries survive.
#   Skin texture, fine wrinkles → mostly removed.
#   Risk: important but faint palm line segments may also disappear.
#
# MEDIUM SENSITIVITY (moderate thresholds):
#   A balance between structural edges and texture noise.
#   Often the most useful starting point for palm images.
#
# HIGH SENSITIVITY (low thresholds):
#   Even faint edges survive. Fine skin texture becomes visible.
#   Risk: so many edges that the actual palm lines are lost in noise.

CANNY_CONFIGS = [
    {"label": "Low sensitivity",    "low": 100, "high": 200},
    {"label": "Medium sensitivity", "low":  50, "high": 150},
    {"label": "High sensitivity",   "low":  20, "high": 100},
]

# ── Display ───────────────────────────────────────────────────────────────────
PANEL_HEIGHT = 380

# =============================================================================
# SETUP
# =============================================================================
IMAGE_PATH = os.path.join(RAW_IMAGES_DIR, IMAGE_FILENAME)

print("=" * 68)
print("  PalmVerse — Phase 2, Milestone 2.3: Canny Edge Detection")
print("=" * 68)
print(f"\n  Working width  : {WORKING_WIDTH} px")
print(f"  Blur sizes     : None  |  {BLUR_SMALL}  |  {BLUR_LARGE}")
for cfg in CANNY_CONFIGS:
    print(f"  Canny config   : [{cfg['label']:22s}] low={cfg['low']}, high={cfg['high']}")
print()

os.makedirs(EDGES_DIR, exist_ok=True)

# =============================================================================
# STEP 1 — LOAD + RESIZE
# =============================================================================
print("─── Step 1: Load + Resize ───────────────────────────────────────────")

if not os.path.exists(IMAGE_PATH):
    print(f"❌ File not found: {IMAGE_PATH}")
    sys.exit(1)

image_bgr = cv2.imread(IMAGE_PATH)
if image_bgr is None:
    print("❌ OpenCV could not read the image.")
    sys.exit(1)

orig_h, orig_w = image_bgr.shape[:2]
scale          = WORKING_WIDTH / orig_w
working_h      = int(orig_h * scale)

image_bgr = cv2.resize(image_bgr, (WORKING_WIDTH, working_h),
                        interpolation=cv2.INTER_AREA)

print(f"  {orig_w} × {orig_h} px  →  {WORKING_WIDTH} × {working_h} px  (scale {scale:.3f})")
print("  ✅ Loaded and resized.")

# =============================================================================
# STEP 2 — GRAYSCALE + CLAHE
# =============================================================================
# Standard pipeline. CLAHE normalises local contrast before any edge work.
# WHY CLAHE BEFORE CANNY:
#   Canny uses gradient magnitudes. Without CLAHE, a dimly-lit region of
#   the palm may have palm lines with only a 15-unit brightness difference
#   from surrounding skin — too faint for edges to survive thresholding.
#   CLAHE amplifies those local contrasts, making the edge detectable.

print("\n─── Step 2: Grayscale + CLAHE ───────────────────────────────────────")

image_gray  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
clahe       = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
image_clahe = clahe.apply(image_gray)

print(f"  Grayscale  : mean={image_gray.mean():.1f}")
print(f"  CLAHE      : mean={image_clahe.mean():.1f}  (local contrast enhanced)")

# =============================================================================
# STEP 3 — GAUSSIAN BLUR (three levels for comparison)
# =============================================================================
# cv2.GaussianBlur(src, ksize, sigmaX)
# We prepare three versions: raw CLAHE (no extra blur), small blur, large blur.
# Each will be fed into Canny to show how blur changes edge maps.
#
# WHY BLUR BEFORE CANNY:
#   The Canny algorithm's Stage 1 is noise suppression, but it uses only
#   a fixed small Gaussian internally. For fine-grained control over how
#   aggressively noise is smoothed before gradient computation, we apply
#   an explicit GaussianBlur first.
#   Rule of thumb: blur more → fewer false edges → may lose thin detail.

print("\n─── Step 3: Gaussian Blur (three levels) ────────────────────────────")

blur_none  = image_clahe   # no blur — used directly
blur_small = cv2.GaussianBlur(image_clahe, BLUR_SMALL,  sigmaX=0)
blur_large = cv2.GaussianBlur(image_clahe, BLUR_LARGE,  sigmaX=0)

diff_s = cv2.absdiff(blur_none, blur_small)
diff_l = cv2.absdiff(blur_none, blur_large)
print(f"  No blur → small  ({BLUR_SMALL})  : mean pixel change = {diff_s.mean():.2f}")
print(f"  No blur → large  ({BLUR_LARGE}) : mean pixel change = {diff_l.mean():.2f}")
print("  (More change = more detail removed by blur)")

# =============================================================================
# STEP 4 — SAVE PREPROCESSING PIPELINE IMAGES
# =============================================================================
print("\n─── Step 4: Saving Preprocessing Images ─────────────────────────────")

def save(img, name):
    path = os.path.join(EDGES_DIR, name)
    ok   = cv2.imwrite(path, img)
    print(f"  {'✅' if ok else '❌'}  {name}")
    return path

save(image_gray,    "01_gray.jpg")
save(image_clahe,   "02_clahe.jpg")
save(blur_small,    f"03_blur_small_{BLUR_SMALL[0]}x{BLUR_SMALL[1]}.jpg")
save(blur_large,    f"04_blur_large_{BLUR_LARGE[0]}x{BLUR_LARGE[1]}.jpg")

# =============================================================================
# STEP 5 — CANNY EDGE DETECTION
# =============================================================================
# cv2.Canny(image, threshold1, threshold2, apertureSize=3, L2gradient=False)
#
#   image        : single-channel uint8 input (our blurred grayscale)
#   threshold1   : the LOW threshold  — minimum for a weak edge to exist
#   threshold2   : the HIGH threshold — minimum to be classified as strong edge
#   apertureSize : size of the Sobel kernel used internally (3 is standard)
#   L2gradient   : False=L1 norm for gradient magnitude (faster, slight approx)
#                  True =L2 norm (sqrt of sum of squares — more accurate)
#                  For exploration, L2gradient=False is fine.
#
# Returns a binary image: 255 = edge pixel, 0 = not an edge.
#
# REMEMBER: Canny detects ALL intensity boundaries — palm lines, skin
# texture, hair, shadows, background. It cannot label which is which.
#
# We run each Canny config on our small-blur image (best trade-off
# between noise reduction and edge preservation from the comparison).
# We also run all three blur levels on the medium sensitivity config
# to isolate the effect of blur separately.

print("\n─── Step 5: Canny Edge Detection ────────────────────────────────────")
print(f"\n  {'Config':<25} {'Low':>5} {'High':>6}  {'Edge px':>10}  {'% of image':>11}")
print("  " + "-" * 63)

total_px = image_clahe.size   # H × W
canny_results = []

for cfg in CANNY_CONFIGS:
    # Use small-blur as the primary input
    edges = cv2.Canny(blur_small, cfg["low"], cfg["high"])

    edge_px  = int(np.sum(edges == 255))
    edge_pct = 100 * edge_px / total_px
    print(f"  {cfg['label']:<25} {cfg['low']:>5} {cfg['high']:>6}  {edge_px:>10,}  {edge_pct:>10.2f}%")

    canny_results.append({**cfg, "edges": edges, "px": edge_px, "pct": edge_pct})

# ── Blur impact on medium sensitivity ─────────────────────────────────────────
print(f"\n  Blur impact at medium sensitivity (low=50, high=150):")
print(f"  {'Blur':>25} {'Edge px':>10}  {'% of image':>11}")
print("  " + "-" * 50)

for blur_img, blur_label in [
    (blur_none,  "No blur (raw CLAHE)"),
    (blur_small, f"Small {BLUR_SMALL}"),
    (blur_large, f"Large {BLUR_LARGE}"),
]:
    e     = cv2.Canny(blur_img, 50, 150)
    e_px  = int(np.sum(e == 255))
    e_pct = 100 * e_px / total_px
    print(f"  {blur_label:>25} {e_px:>10,}  {e_pct:>10.2f}%")

# =============================================================================
# STEP 6 — SAVE CANNY OUTPUTS
# =============================================================================
print("\n─── Step 6: Saving Canny Outputs ────────────────────────────────────")

label_to_fname = {
    "Low sensitivity":    "05_canny_low_sensitivity.jpg",
    "Medium sensitivity": "06_canny_medium_sensitivity.jpg",
    "High sensitivity":   "07_canny_high_sensitivity.jpg",
}
for r in canny_results:
    save(r["edges"], label_to_fname[r["label"]])

# =============================================================================
# STEP 7 — BUILD COMPARISON GRIDS
# =============================================================================

print("\n─── Step 7: Building Comparison Grids ───────────────────────────────")

FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE  = 0.50
THICKNESS   = 2

def make_panel(img, label, h=PANEL_HEIGHT):
    """Resize to target height, ensure 3-channel BGR, overlay text label."""
    oh, ow = img.shape[:2]
    nw     = int(ow * h / oh)
    small  = cv2.resize(img, (nw, h), interpolation=cv2.INTER_AREA)
    if small.ndim == 2:
        small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    cv2.putText(small, label, (11, 28), FONT, FONT_SCALE, (0, 0, 0),       THICKNESS+2)
    cv2.putText(small, label, (10, 27), FONT, FONT_SCALE, (255, 255, 255), THICKNESS)
    return small

def hstack_trim(panels):
    min_w = min(p.shape[1] for p in panels)
    return np.hstack([p[:, :min_w] for p in panels])

def vstack_trim(rows):
    min_w = min(r.shape[1] for r in rows)
    return np.vstack([r[:, :min_w] for r in rows])

# ── Grid A: Preprocessing stages ──────────────────────────────────────────────
# Shows the pipeline: gray → CLAHE → blur_small → blur_large
# Helps you see how each step changes the input before Canny.
grid_a = hstack_trim([
    make_panel(image_gray,   "1. Grayscale"),
    make_panel(image_clahe,  "2. CLAHE"),
    make_panel(blur_small,   f"3. Blur {BLUR_SMALL}"),
    make_panel(blur_large,   f"4. Blur {BLUR_LARGE}"),
])
save(grid_a, "GRID_A_preprocessing.jpg")
print("  ✅ Grid A: Preprocessing stages")

# ── Grid B: Threshold sensitivity comparison ───────────────────────────────────
# Three Canny results on the same blur input — isolates threshold effect.
# Label includes threshold values so you can correlate visually with numbers.
grid_b = hstack_trim([
    make_panel(
        r["edges"],
        f"{r['label']} | {r['low']}/{r['high']} | {r['pct']:.1f}%"
    )
    for r in canny_results
])
save(grid_b, "GRID_B_threshold_comparison.jpg")
print("  ✅ Grid B: Threshold sensitivity comparison")

# ── Grid C: Blur impact on medium threshold ────────────────────────────────────
# Same thresholds (50/150), three blur levels — isolates blur effect.
canny_no_blur    = cv2.Canny(blur_none,  50, 150)
canny_blur_small = cv2.Canny(blur_small, 50, 150)
canny_blur_large = cv2.Canny(blur_large, 50, 150)

grid_c = hstack_trim([
    make_panel(canny_no_blur,    f"No blur | {np.sum(canny_no_blur==255):,}px"),
    make_panel(canny_blur_small, f"Blur{BLUR_SMALL} | {np.sum(canny_blur_small==255):,}px"),
    make_panel(canny_blur_large, f"Blur{BLUR_LARGE} | {np.sum(canny_blur_large==255):,}px"),
])
save(grid_c, "GRID_C_blur_impact_on_canny.jpg")
print("  ✅ Grid C: Blur impact on Canny (same threshold, three blur levels)")

# ── Grid D: Full overview (all grids stacked vertically) ──────────────────────
grid_d = vstack_trim([grid_a, grid_b, grid_c])
save(grid_d, "08_canny_comparison.jpg")
print("  ✅ Grid D: Full overview (A + B + C stacked)")

# =============================================================================
# STEP 8 — DISPLAY
# =============================================================================
print("\n─── Step 8: Display ─────────────────────────────────────────────────")
print("  Three windows will open. Press any key in each to advance.\n")

for title, grid in [
    ("PalmVerse — Grid A: Preprocessing Pipeline", grid_a),
    ("PalmVerse — Grid B: Canny Threshold Comparison", grid_b),
    ("PalmVerse — Grid C: Blur Impact on Canny", grid_c),
]:
    cv2.imshow(title, grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 68)
print("  ✅ Milestone 2.3 Complete!")
print("=" * 68)
print(f"""
  Results summary:
  ┌──────────────────────────────────────────────────────────────┐
  │  Config               Low   High   Edge px     % of image   │
  ├──────────────────────────────────────────────────────────────┤""")

for r in canny_results:
    print(f"  │  {r['label']:<22} {r['low']:>4}  {r['high']:>5}  {r['px']:>9,}  {r['pct']:>10.2f}%   │")

print(f"""  └──────────────────────────────────────────────────────────────┘

📚 What you learned:
   1. Edges = locations of rapid brightness change in an image
   2. Gradient = rate of change of brightness; measured by Sobel filters
   3. Canny's 5 stages: blur → gradient → NMS → double-threshold
      → hysteresis edge tracking
   4. threshold2 (high): minimum gradient to be a guaranteed strong edge
      threshold1 (low):  minimum for a weak edge if connected to a strong one
   5. Blur BEFORE Canny suppresses noise gradients → fewer false edges
   6. More blur → fewer edges; less blur → more edges (including noise)
   7. Lower thresholds → more sensitive → more edges including texture noise
   8. Higher thresholds → less sensitive → only the strongest boundaries
   9. Canny detects ALL brightness boundaries — palm lines, skin texture,
      hair, shadows, background. It cannot label any of them.
   10. There is no universal threshold pair: the optimal values depend on
       your specific image, lighting, and what you're trying to detect.

  IMPORTANT LESSON:
  Even your best Canny result contains hundreds or thousands of
  edge pixels that are NOT palm lines. Without a way to label which
  edges belong to which line, classical CV reaches its limit here.
  This is precisely why Phase 6 (semantic segmentation with U-Net)
  is the real solution — a model trained on annotated examples learns
  which spatial patterns correspond to Life, Head, and Heart lines.
""")
print("=" * 68)
print(f"\n  Outputs saved to: {EDGES_DIR}")
