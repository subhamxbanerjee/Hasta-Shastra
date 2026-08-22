# ============================================================
# PalmVerse - Phase 5, Milestone 5.1
# Multi-Stage Palm Crease Enhancement
# ============================================================

"""This script takes the normalized palm image (512×512) generated in
Phase 4 Milestone 4.2 and runs a series of classic OpenCV operations to
enhance faint palm creases and produce candidate line maps.

**IMPORTANT**: The output consists of *candidate* structures – we do NOT
claim to have identified the Life, Head, Heart, or Fate lines. Those
semantic detections will be added in later phases.
"""

import cv2
import numpy as np
from pathlib import Path
import os

# ------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------

def save_image(path: Path, img, description: str) -> bool:
    """Write ``img`` to ``path`` using ``cv2.imwrite``.
    Prints success/failure and returns the boolean status.
    """
    success = cv2.imwrite(str(path), img)
    if success:
        print(f"  [OK] {path.name} saved ({description})")
    else:
        print(f"  [WARN] Failed to save {path.name} ({description})")
    return success


def add_label(img, text: str):
    """Add a black banner with white text on the top of *img*.
    Returns a new image.
    """
    labeled = img.copy()
    cv2.rectangle(labeled, (0, 0), (labeled.shape[1], 45), (0, 0, 0), -1)
    cv2.putText(
        labeled,
        text,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    return labeled


def resize_for_display(img, size=(500, 500)):
    resized = cv2.resize(img, size)
    # Ensure output has 3 channels for consistent stacking
    if len(resized.shape) == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    return resized


def percentage_edge_pixels(edge_img: np.ndarray) -> float:
    """Return the percentage of non‑zero (edge) pixels in *edge_img*.
    ``edge_img`` is expected to be a binary (0/255) image.
    """
    edge_pixels = np.count_nonzero(edge_img)
    total = edge_img.size
    return (edge_pixels / total) * 100.0

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT_DIR / "data" / "processed" / "palm_512.jpg"
OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "line_enhancement"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# STEP 1 — LOAD IMAGE
# ============================================================
print("=" * 68)
print("  PalmVerse — Phase 5, Milestone 5.1")
print("  Multi-Stage Palm Crease Enhancement")
print("=" * 68)

print("\n--- Step 1: Load Image -------------------------------------")
image = cv2.imread(str(INPUT_PATH))
if image is None:
    raise FileNotFoundError(f"Could not load image: {INPUT_PATH}")
height, width = image.shape[:2]
print(f"  Image dimensions: {width} × {height}")

# ============================================================
# STEP 2 — GRAYSCALE
# ============================================================
print("\n--- Step 2: Grayscale Conversion -----------------------------")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
mean_brightness = gray.mean()
std_brightness = gray.std()
print(f"  Mean brightness: {mean_brightness:.2f}")
print(f"  Std deviation : {std_brightness:.2f}")
save_image(OUTPUT_DIR / "01_grayscale.jpg", gray, "grayscale")

# ============================================================
# STEP 3 — CLAHE COMPARISON
# ============================================================
print("\n--- Step 3: CLAHE (Local Contrast Enhancement) ---------------")
# CLAHE purpose: increase local contrast so faint creases become more visible
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_img = clahe.apply(gray)
mean_clahe = clahe_img.mean()
print(f"  Brightness before CLAHE : {mean_brightness:.2f}")
print(f"  Brightness after CLAHE  : {mean_clahe:.2f}")
save_image(OUTPUT_DIR / "02_clahe.jpg", clahe_img, "CLAHE")

# ============================================================
# STEP 4 — NOISE REDUCTION EXPERIMENT
# ============================================================
print("\n--- Step 4: Noise Reduction Experiments ----------------------")
# A. Gaussian Blur
gaussian = cv2.GaussianBlur(clahe_img, (5, 5), 0)
save_image(OUTPUT_DIR / "03_gaussian_blur.jpg", gaussian, "Gaussian blur (5x5)")
print("   Gaussian Blur – kernel=(5,5)")
# B. Bilateral Filter
bilateral = cv2.bilateralFilter(clahe_img, d=9, sigmaColor=75, sigmaSpace=75)
save_image(OUTPUT_DIR / "04_bilateral.jpg", bilateral, "Bilateral filter")
print("   Bilateral Filter – d=9, sigmaColor=75, sigmaSpace=75")

# ============================================================
# STEP 5 — ILLUMINATION / BACKGROUND NORMALIZATION
# ============================================================
print("\n--- Step 5: Illumination / Background Normalization -----------")
large_blur = cv2.GaussianBlur(clahe_img, (51, 51), 0)
# Subtract the large‑scale illumination estimate
norm = cv2.subtract(clahe_img.astype(np.float32), large_blur.astype(np.float32))
# Rescale to full 0‑255 range
norm = cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX)
norm_uint8 = norm.astype(np.uint8)
save_image(OUTPUT_DIR / "05_background_normalized.jpg", norm_uint8, "background normalized")

# ============================================================
# STEP 6 — DARK CREASE ENHANCEMENT (BLACK‑HAT)
# ============================================================
print("\n--- Step 6: Dark Crease Enhancement (Black-Hat) ---------------")
kernel_blackhat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
blackhat = cv2.morphologyEx(norm_uint8, cv2.MORPH_BLACKHAT, kernel_blackhat)
save_image(OUTPUT_DIR / "06_blackhat.jpg", blackhat, "black-hat result")
# Boost contrast of the black-hat output using CLAHE again
enhanced = clahe.apply(blackhat)
save_image(OUTPUT_DIR / "07_crease_enhanced.jpg", enhanced, "crease enhanced")

# ============================================================
# STEP 7 — LINE CANDIDATE THRESHOLDING
# ============================================================
print("\n--- Step 7: Line Candidate Thresholding ---------------------")
# A. Otsu threshold on the enhanced image
otsu_thresh, _ = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"  Otsu computed threshold value: {otsu_thresh:.2f}")
otsu_candidates = cv2.threshold(enhanced, otsu_thresh, 255, cv2.THRESH_BINARY)[1]
save_image(OUTPUT_DIR / "08_otsu_candidates.jpg", otsu_candidates, "Otsu candidates")
# B. Adaptive Gaussian threshold
adaptive = cv2.adaptiveThreshold(
    enhanced,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    blockSize=31,
    C=5,
)
save_image(OUTPUT_DIR / "09_adaptive_candidates.jpg", adaptive, "Adaptive Gaussian candidates")

# ============================================================
# STEP 8 — MORPHOLOGICAL CLEANUP
# ============================================================
print("\n--- Step 8: Morphological Cleanup -----------------------------")
kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
opened = cv2.morphologyEx(otsu_candidates, cv2.MORPH_OPEN, kernel_morph)
cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_morph)
save_image(OUTPUT_DIR / "10_cleaned_candidates.jpg", cleaned, "cleaned candidates")

# ============================================================
# STEP 9 — CANNY COMPARISON
# ============================================================
print("\n--- Step 9: Canny Edge Comparison -----------------------------")
# Canny on the CLAHE image
canny_clahe = cv2.Canny(clahe_img, 50, 150)
save_image(OUTPUT_DIR / "11_canny_clahe.jpg", canny_clahe, "Canny on CLAHE")
# Canny on the enhanced crease image
canny_enhanced = cv2.Canny(enhanced, 50, 150)
save_image(OUTPUT_DIR / "12_canny_enhanced.jpg", canny_enhanced, "Canny on enhanced")
perc_clahe = percentage_edge_pixels(canny_clahe)
perc_enh = percentage_edge_pixels(canny_enhanced)
print(f"  Edge-pixel % (CLAHE Canny)   : {perc_clahe:.2f}%")
print(f"  Edge-pixel % (Enhanced Canny): {perc_enh:.2f}%")

# ============================================================
# STEP 10 — CREATE FINAL OVERLAY
# ============================================================
print("\n--- Step 10: Final Overlay Visualization ---------------------")
overlay = image.copy()
# Green mask for cleaned candidate pixels
green_mask = np.zeros_like(overlay)
green_mask[cleaned == 255] = (0, 255, 0)
overlay = cv2.addWeighted(overlay, 1.0, green_mask, 0.7, 0)
save_image(OUTPUT_DIR / "13_line_candidate_overlay.jpg", overlay, "candidate overlay")

# ============================================================
# STEP 11 — BUILD COMPARISON GRIDS
# ============================================================
print("\n--- Step 11: Build Comparison Grids ---------------------------")
# Stage grid (8 panels)
stage_imgs = [
    add_label(resize_for_display(image), "1. Normalized Palm"),
    add_label(resize_for_display(gray), "2. Grayscale"),
    add_label(resize_for_display(clahe_img), "3. CLAHE"),
    add_label(resize_for_display(gaussian), "4. Gaussian Blur"),
    add_label(resize_for_display(bilateral), "5. Bilateral Filter"),
    add_label(resize_for_display(norm_uint8), "6. Background Normalized"),
    add_label(resize_for_display(blackhat), "7. Black‑Hat"),
    add_label(resize_for_display(enhanced), "8. Crease Enhanced"),
]
top = np.hstack(stage_imgs[:4])
bottom = np.hstack(stage_imgs[4:])
stages_grid = np.vstack([top, bottom])
save_image(OUTPUT_DIR / "line_enhancement_stages.jpg", stages_grid, "stages grid")

# Candidate comparison grid (6 panels)
candidate_imgs = [
    add_label(resize_for_display(otsu_candidates), "1. Otsu Candidates"),
    add_label(resize_for_display(adaptive), "2. Adaptive Candidates"),
    add_label(resize_for_display(cleaned), "3. Cleaned Candidates"),
    add_label(resize_for_display(canny_clahe), "4. Canny (CLAHE)"),
    add_label(resize_for_display(canny_enhanced), "5. Canny (Enhanced)"),
    add_label(resize_for_display(overlay), "6. Final Overlay"),
]
row1 = np.hstack(candidate_imgs[:3])
row2 = np.hstack(candidate_imgs[3:])
candidates_grid = np.vstack([row1, row2])
save_image(OUTPUT_DIR / "line_candidate_comparison.jpg", candidates_grid, "candidate comparison grid")

# ============================================================
# STEP 12 — DISPLAY
# ============================================================
print("\n--- Step 12: Display Comparison Grids ---------------------")
cv2.namedWindow("PalmVerse — Milestone 5.1: Enhancement Stages", cv2.WINDOW_NORMAL)
cv2.imshow("PalmVerse — Milestone 5.1: Enhancement Stages", stages_grid)
print("  Press any key to close the first window.")
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.namedWindow("PalmVerse — Milestone 5.1: Line Candidate Comparison", cv2.WINDOW_NORMAL)
cv2.imshow("PalmVerse — Milestone 5.1: Line Candidate Comparison", candidates_grid)
print("  Press any key to close the second window.")
cv2.waitKey(0)
cv2.destroyAllWindows()

# ============================================================
# STEP 13 — TERMINAL SUMMARY
# ============================================================
print("\n" + "=" * 68)
print("  MILESTONE 5.1 COMPLETE")
print("=" * 68)
print("\nPipeline completed:\n")
print("Normalized Palm")
print("    ↓")
print("CLAHE")
print("    ↓")
print("Noise Reduction")
print("    ↓")
print("Illumination Normalization")
print("    ↓")
print("Black‑Hat Crease Enhancement")
print("    ↓")
print("Threshold Candidate Maps")
print("    ↓")
print("Morphological Cleanup")
print("    ↓")
print("Edge Comparison")
print("    ↓")
print("Palm Crease Candidate Map")

print("\nWHAT THIS CAN DO:")
print("- Enhance visible dark crease structures")
print("- Reduce some illumination variation")
print("- Produce candidate line structures")
print("- Prepare the palm for later feature extraction")

print("\nWHAT THIS CANNOT DO YET:")
print("- Reliably identify the Life Line")
print("- Reliably identify the Head Line")
print("- Reliably identify the Heart Line")
print("- Reliably identify the Fate Line")
print("- Make personality or future predictions")

print("\nFuture work will require either a trained segmentation model or a geometric feature pipeline to map candidates to semantic palm lines.")

print("=" * 68)
