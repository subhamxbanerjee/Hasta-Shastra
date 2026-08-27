# ============================================================
# PalmVerse - Phase 5, Milestone 5.2
# Major Palm Line Candidate Extraction & Filtering
# ============================================================
"""This script processes the cleaned binary candidate image produced by
Phase 5 Milestone 5.1 and extracts major palm-line-like structures.

The module exposes ``run_candidate_extraction(enhanced_img_path, output_dir)``
for use by the batch pipeline. A standalone CLI entry point is retained
for backward compatibility. All CV algorithm parameters are kept as
internal constants (Phase 5.6 is a path/data-flow refactor only).
"""

import cv2
import numpy as np
import json
from pathlib import Path

# ------------------------------------------------------------
# Configurable constants – kept internal; do not expose via CLI
# ------------------------------------------------------------
MIN_COMPONENT_AREA = 40          # minimum pixel area for a component
MIN_COMPONENT_LENGTH = 20        # minimum length (max(bbox_w, bbox_h))
BORDER_MARGIN = 8                # pixels from image edge considered border zone
MIN_CONTOUR_LENGTH = 30          # minimum arc length to keep a contour
MIN_ASPECT_RATIO = 1.5           # favour elongated shapes (width/height)

# ------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------

def save_image(path: Path, img, description: str) -> bool:
    """Write *img* to *path* using cv2.imwrite.
    Prints a concise status message and returns success flag.
    """
    success = cv2.imwrite(str(path), img)
    if success:
        print(f"  [OK] {path.name} saved ({description})")
    else:
        print(f"  [WARN] Failed to save {path.name} ({description})")
    return success


def add_label(img, text: str):
    """Add a black banner with white text on the top of *img*.
    Returns a new image (does not modify the original).
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
    """Resize *img* for display/comparison grids.
    Guarantees a 3-channel BGR output regardless of input shape.
    """
    resized = cv2.resize(img, size)
    if len(resized.shape) == 2:  # grayscale
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    return resized


def percentage_foreground(img: np.ndarray) -> float:
    """Return percentage of non-zero (foreground) pixels in a binary image."""
    fg = np.count_nonzero(img)
    total = img.size
    return (fg / total) * 100.0


# ------------------------------------------------------------
# Core processing function (public API)
# ------------------------------------------------------------

def run_candidate_extraction(
    enhanced_img_path: Path,
    output_dir: Path,
    normalized_img_path: Path = None,
    show: bool = False,
) -> Path:
    """Extract major line candidates from the cleaned candidate image.

    Parameters
    ----------
    enhanced_img_path: Path to the cleaned candidate mask produced by the
        enhancement stage (e.g., ``10_cleaned_candidates.jpg``).
    output_dir: Directory where all intermediate artefacts and the final
        ``05_major_line_candidates.jpg`` will be stored. This is expected
        to be the per-image ``line_candidates/`` folder.
    normalized_img_path: Path to the 512x512 normalized palm image used for
        the overlay visualization. If None, falls back to the legacy
        ``data/processed/palm_512.jpg`` location.
    show: If True, displays OpenCV windows (default: False for headless batch).

    Returns
    -------
    Path to the major-line-candidates mask (``05_major_line_candidates.jpg``).
    """
    enhanced_img_path = enhanced_img_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve normalized_img_path fallback
    if normalized_img_path is None:
        ROOT_DIR = Path(__file__).resolve().parents[1]
        normalized_img_path = ROOT_DIR / "data" / "processed" / "palm_512.jpg"
    normalized_img_path = normalized_img_path.resolve()

    # ------------------------------------------------------------------
    # STEP 1 – LOAD AND VALIDATE INPUT
    # ------------------------------------------------------------------
    print("=" * 68)
    print("  PalmVerse - Phase 5, Milestone 5.2")
    print("  Major Palm Line Candidate Extraction & Filtering")
    print("=" * 68)
    print(f"\n--- Step 1: Load cleaned candidate image ---")
    print(f"  Input path: {enhanced_img_path}")
    if not enhanced_img_path.is_file():
        raise FileNotFoundError(
            f"Cleaned candidate image not found: {enhanced_img_path}\n"
            "Run palm_line_enhancement.py first to generate it."
        )
    cleaned_candidate = cv2.imread(str(enhanced_img_path), cv2.IMREAD_UNCHANGED)
    if cleaned_candidate is None:
        raise RuntimeError("Failed to read the cleaned candidate image.")
    # Ensure single-channel
    if len(cleaned_candidate.shape) == 3:
        cleaned_candidate = cv2.cvtColor(cleaned_candidate, cv2.COLOR_BGR2GRAY)
    height, width = cleaned_candidate.shape[:2]
    print(f"  Dimensions: {width} x {height}")
    print(f"  Non-zero pixels: {np.count_nonzero(cleaned_candidate)}")
    print(f"  Foreground %: {percentage_foreground(cleaned_candidate):.2f}%")

    # ------------------------------------------------------------------
    # STEP 2 – ENSURE STRICT BINARY IMAGE
    # ------------------------------------------------------------------
    print("\n--- Step 2: Binary conversion ---")
    _, binary_img = cv2.threshold(cleaned_candidate, 127, 255, cv2.THRESH_BINARY)
    save_image(output_dir / "01_binary_input.jpg", binary_img, "binary input")
    print(f"  Foreground % after binary: {percentage_foreground(binary_img):.2f}%")

    # ------------------------------------------------------------------
    # STEP 3 – LIGHT MORPHOLOGICAL CLEANING
    # ------------------------------------------------------------------
    print("\n--- Step 3: Light morphological cleaning ---")
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel_small)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_small)
    save_image(output_dir / "02_morphology_cleaned.jpg", closed, "morphology cleaned")
    print(f"  Foreground % before morph: {percentage_foreground(binary_img):.2f}%")
    print(f"  Foreground % after morph : {percentage_foreground(closed):.2f}%")

    # ------------------------------------------------------------------
    # STEP 4 – CONNECTED COMPONENT ANALYSIS
    # ------------------------------------------------------------------
    print("\n--- Step 4: Connected components ---")
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
    print(f"  Total components (including background): {num_labels - 1}")

    # ------------------------------------------------------------------
    # STEP 5 – FILTER BY AREA / LENGTH
    # ------------------------------------------------------------------
    print("\n--- Step 5: Size filtering ---")
    size_filtered = np.zeros_like(closed)
    retained_ids = []
    for comp_id in range(1, num_labels):  # skip background 0
        area = stats[comp_id, cv2.CC_STAT_AREA]
        w = stats[comp_id, cv2.CC_STAT_WIDTH]
        h = stats[comp_id, cv2.CC_STAT_HEIGHT]
        est_len = max(w, h)
        if area >= MIN_COMPONENT_AREA and est_len >= MIN_COMPONENT_LENGTH:
            size_filtered[labels == comp_id] = 255
            retained_ids.append(comp_id)
    print(f"  Components retained after size filter: {len(retained_ids)}")
    print(f"  Components rejected: {num_labels - 1 - len(retained_ids)}")
    save_image(output_dir / "03_size_filtered.jpg", size_filtered, "size filtered")

    # ------------------------------------------------------------------
    # STEP 6 – BORDER SUPPRESSION
    # ------------------------------------------------------------------
    print("\n--- Step 6: Border suppression ---")
    border_suppressed = size_filtered.copy()
    border_removed = 0
    for comp_id in retained_ids:
        left = stats[comp_id, cv2.CC_STAT_LEFT]
        top = stats[comp_id, cv2.CC_STAT_TOP]
        w = stats[comp_id, cv2.CC_STAT_WIDTH]
        h = stats[comp_id, cv2.CC_STAT_HEIGHT]
        touches_left = left <= BORDER_MARGIN
        touches_top = top <= BORDER_MARGIN
        touches_right = (left + w) >= (width - BORDER_MARGIN)
        touches_bottom = (top + h) >= (height - BORDER_MARGIN)
        if touches_left or touches_top or touches_right or touches_bottom:
            # Very conservative: also require relatively large area to be considered border structure
            if stats[comp_id, cv2.CC_STAT_AREA] > (MIN_COMPONENT_AREA * 5):
                border_suppressed[labels == comp_id] = 0
                border_removed += 1
    print(f"  Border-connected components removed: {border_removed}")
    save_image(output_dir / "04_border_suppressed.jpg", border_suppressed, "border suppressed")

    # ------------------------------------------------------------------
    # STEP 7 – CONTOUR EXTRACTION
    # ------------------------------------------------------------------
    print("\n--- Step 7: Contour extraction ---")
    contours, _ = cv2.findContours(
        border_suppressed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    print(f"  Total contours found: {len(contours)}")

    # ------------------------------------------------------------------
    # STEP 8 – FILTER MAJOR LINE-LIKE CONTOURS
    # ------------------------------------------------------------------
    print("\n--- Step 8: Major line candidate filtering ---")
    major_mask = np.zeros_like(border_suppressed)
    major_contours = []
    for idx, cnt in enumerate(contours):
        arc_len = cv2.arcLength(cnt, closed=True)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = max(w, h) / (min(w, h) + 1e-5)  # avoid div-zero
        area = cv2.contourArea(cnt)
        if arc_len >= MIN_CONTOUR_LENGTH and aspect >= MIN_ASPECT_RATIO:
            cv2.drawContours(major_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            major_contours.append({
                "id": idx,
                "area": float(area),
                "arc_length": float(arc_len),
                "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                "aspect_ratio": float(aspect),
                "centroid": {"x": float(centroids[idx][0]), "y": float(centroids[idx][1])},
            })
    print(f"  Major line candidates retained: {len(major_contours)}")
    save_image(output_dir / "05_major_line_candidates.jpg", major_mask, "major line candidates")

    # ------------------------------------------------------------------
    # STEP 9 – DRAW OVERLAY ON NORMALIZED PALM
    # ------------------------------------------------------------------
    print("\n--- Step 9: Overlay visualization ---")
    if not normalized_img_path.is_file():
        raise FileNotFoundError(f"Normalized palm image not found for overlay: {normalized_img_path}")
    normalized = cv2.imread(str(normalized_img_path))
    overlay = normalized.copy()
    for cand in major_contours:
        cx = int(cand["centroid"]["x"])
        cy = int(cand["centroid"]["y"])
        cv2.circle(overlay, (cx, cy), 4, (0, 255, 0), -1)
        # Locate contour index in original list matching this centroid
        best_idx = None
        best_dist = float('inf')
        for i, cnt in enumerate(contours):
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            cxi = int(M['m10'] / M['m00'])
            cyi = int(M['m01'] / M['m00'])
            d = (cxi - cx) ** 2 + (cyi - cy) ** 2
            if d < best_dist:
                best_dist = d
                best_idx = i
        if best_idx is not None:
            cv2.drawContours(overlay, [contours[best_idx]], -1, (0, 255, 0), 2)
        cv2.putText(
            overlay,
            str(cand["id"]),
            (cx - 10, cy + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    save_image(output_dir / "06_major_candidates_overlay.jpg", overlay, "candidate overlay")

    # ------------------------------------------------------------------
    # STEP 10 – SAVE JSON MEASUREMENTS
    # ------------------------------------------------------------------
    json_path = output_dir / "candidate_measurements.json"
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(major_contours, jf, indent=2)
    print(f"  JSON measurements saved to {json_path.name}")

    # ------------------------------------------------------------------
    # STEP 11 – BUILD COMPARISON GRID (2x3)
    # ------------------------------------------------------------------
    print("\n--- Step 11: Comparison grid ---")
    imgs = [
        add_label(resize_for_display(binary_img), "1. Binary Input"),
        add_label(resize_for_display(closed), "2. Morphology Cleaned"),
        add_label(resize_for_display(size_filtered), "3. Size Filtered"),
        add_label(resize_for_display(border_suppressed), "4. Border Suppressed"),
        add_label(resize_for_display(major_mask), "5. Major Candidates"),
        add_label(resize_for_display(overlay), "6. Overlay"),
    ]
    top = np.hstack(imgs[:3])
    bottom = np.hstack(imgs[3:])
    grid = np.vstack([top, bottom])
    save_image(output_dir / "line_candidate_filtering_comparison.jpg", grid, "comparison grid")

    if show:
        cv2.namedWindow("PalmVerse - Milestone 5.2: Comparison", cv2.WINDOW_NORMAL)
        cv2.imshow("PalmVerse - Milestone 5.2: Comparison", grid)
        print("Press any key to close the comparison window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("  MILESTONE 5.2 COMPLETE")
    print("=" * 68)
    print(f"\nInput foreground %........: {percentage_foreground(binary_img):.2f}%")
    print(f"Connected components......: {num_labels - 1}")
    print(f"Components after size filter: {len(retained_ids)}")
    print(f"Border components removed.....: {border_removed}")
    print(f"Total contours.................: {len(contours)}")
    print(f"Major line candidates retained.: {len(major_contours)}")
    print("All output files saved in:")
    print(f"  {output_dir}")
    print("=" * 68)

    return output_dir / "05_major_line_candidates.jpg"


# ------------------------------------------------------------
# CLI entry point – retains original behaviour
# ------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract major palm line candidates.")
    parser.add_argument("--input", type=str, help="Path to the cleaned candidate image.")
    parser.add_argument("--normalized", type=str, help="Path to the 512x512 normalized palm image (for overlay).")
    parser.add_argument("--output-dir", type=str, help="Directory to store outputs.")
    parser.add_argument("--show", action="store_true", help="Display OpenCV windows.")
    args = parser.parse_args()

    ROOT_DIR = Path(__file__).resolve().parents[1]
    input_path = (
        Path(args.input)
        if args.input
        else ROOT_DIR / "data" / "processed" / "line_enhancement" / "10_cleaned_candidates.jpg"
    )
    norm_path = (
        Path(args.normalized)
        if args.normalized
        else ROOT_DIR / "data" / "processed" / "palm_512.jpg"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else ROOT_DIR / "data" / "processed" / "line_candidates"
    )
    run_candidate_extraction(input_path, output_dir, normalized_img_path=norm_path, show=args.show)
