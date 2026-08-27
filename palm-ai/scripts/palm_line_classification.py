#!/usr/bin/env python3
"""palm_line_classification.py
Phase 5 – Milestone 5.3

This script groups line‑candidate components (produced in Milestone 5.2) into larger line paths and assigns a **preliminary** class (LifeLine, HeadLine, HeartLine, FateLine or Unknown) with a confidence score.

The implementation follows the heuristics described in the implementation plan.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np

# -----------------------------------------------------------------------------
# Heuristic constants – can be overridden from the CLI
# Phase 5.7 calibration applied — see PHASE_5_7_CLASSIFICATION_ANALYSIS.md
# -----------------------------------------------------------------------------
PROXIMITY_DIST = 40          # max Euclidean distance (pixels) between centroids
                              # (raised 15→40: allows nearby line fragments to merge)
ANGLE_TOLERANCE = 20          # max angle difference (degrees) between orientations
MIN_GROUP_LENGTH = 40        # discard groups shorter than this (pixels)
CURVATURE_THRESHOLD = 1.5    # arc_length/area ratio; all observed values 0.29–1.11
                              # (raised 0.25→1.5: restores 30% of confidence formula)
SHOW_WINDOWS = False         # set True for debugging visualisation

# Spatial region boxes for the four classic palm lines (in 512×512 normalized space).
# Phase 5.7: widened based on actual centroid positions observed across 6-image dataset.
# Previous values caused 35–40% of Unknown assignments due to centroids 5–80px outside boxes.
CLASS_REGION_BOUNDS = {
    "HeartLine": {"x": (100, 420), "y": (330, 480)},  # was (150,380),(350,460)
    "HeadLine":  {"x": ( 80, 400), "y": (180, 340)},  # was (120,350),(210,320)
    "LifeLine":  {"x": ( 40, 280), "y": (350, 512)},  # was ( 60,250),(380,500)
    "FateLine":  {"x": (360, 512), "y": (240, 400)},  # was (380,470),(260,380)
}

COLOR_MAP = {
    "LifeLine": (0, 0, 255),      # Red (BGR)
    "HeadLine": (255, 0, 0),      # Blue
    "HeartLine": (0, 255, 0),     # Green
    "FateLine": (255, 0, 255),    # Magenta
    "Unknown": (128, 128, 128),  # Gray
}

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def angle_diff(a1: float, a2: float) -> float:
    """Return the smallest difference between two angles in degrees."""
    diff = abs(a1 - a2) % 360
    return diff if diff <= 180 else 360 - diff

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1

# -----------------------------------------------------------------------------
# Core processing helpers
# -----------------------------------------------------------------------------

def load_measurements(meas_path: Path) -> List[Dict]:
    with open(meas_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_candidate_image(img_path: Path) -> np.ndarray:
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Candidate image not found: {img_path}")
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return binary

def extract_contours(binary: np.ndarray) -> List[np.ndarray]:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def contour_orientation(contour: np.ndarray) -> float:
    if len(contour) < 5:
        return 0.0
    ellipse = cv2.fitEllipse(contour)
    return ellipse[2]

def build_graph(centroids: List[Tuple[float, float]], angles: List[float]) -> List[List[int]]:
    n = len(centroids)
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if euclidean(centroids[i], centroids[j]) <= PROXIMITY_DIST and \
               angle_diff(angles[i], angles[j]) <= ANGLE_TOLERANCE:
                uf.union(i, j)
    groups: Dict[int, List[int]] = {}
    for idx in range(n):
        root = uf.find(idx)
        groups.setdefault(root, []).append(idx)
    return list(groups.values())

def group_properties(group_idxs: List[int], measurements: List[Dict], contours: List[np.ndarray]):
    total_length = 0.0
    points = []
    curvatures = []
    for idx in group_idxs:
        meas = measurements[idx]
        total_length += float(meas.get("arc_length", 0.0))
        curvatures.append(float(meas.get("arc_length", 0.0)) / (float(meas.get("area", 1.0)) + 1e-6))
        points.extend(contours[idx].reshape(-1, 2))
    points = np.array(points, dtype=np.float32)
    # PCA for orientation
    mean, eigenvectors = cv2.PCACompute(points, mean=np.array([]))
    principal_axis = eigenvectors[0]
    orientation = float((math.degrees(math.atan2(principal_axis[1], principal_axis[0]))) % 180)
    centroid_np = np.mean(points, axis=0)
    centroid = (float(centroid_np[0]), float(centroid_np[1]))
    avg_curvature = float(np.mean(curvatures))
    return {
        "length": float(total_length),
        "orientation": orientation,
        "centroid": centroid,
        "curvature": avg_curvature,
    }

def classify_group(props: Dict) -> Tuple[str, float]:
    """Assign a preliminary class and confidence score to a line group.

    Returns
    -------
    (assigned_class, confidence, confidence_reasons)
    confidence_reasons is a dict with the individual score components for
    explainability and diagnostics.
    """
    cx, cy = props["centroid"]
    assigned = "Unknown"
    for line, bounds in CLASS_REGION_BOUNDS.items():
        if bounds["x"][0] <= cx <= bounds["x"][1] and bounds["y"][0] <= cy <= bounds["y"][1]:
            assigned = line
            break

    # Phase 5.7: corrected orientation ranges.
    # HeartLine: lowered lower bound 20→0° (real HeartLine clusters at 5–25°)
    # HeadLine:  widened 10–30° → 0–50° (captures both horizontal and oblique HeadLine)
    # LifeLine:  widened 70–110° → 60–120° (accommodates curved LifeLine variability)
    # FateLine:  corrected 30–70° → 70–120° (near-vertical; all observed at 100–175°)
    ORIENT_RANGES = {
        "HeartLine": (0,  40),
        "HeadLine":  (0,  50),
        "LifeLine":  (60, 120),
        "FateLine":  (70, 120),
    }

    # Phase 5.7: normalizer lowered 1500→400px.
    # Longest observed line across 6-image dataset is 836px; most are 50–250px.
    # At 1500, length contributed near-zero despite 20% weight. At 400 it discriminates.
    length_score = min(props["length"] / 400.0, 1.0)

    # Phase 5.7: CURVATURE_THRESHOLD raised 0.25→1.5.
    # All observed curvature values (arc_length/area) range 0.29–1.11.
    # At 0.25, curvature_score was 0.0 for every group, zeroing out 30% of confidence.
    curvature_score = 1.0 - min(props["curvature"] / CURVATURE_THRESHOLD, 1.0)

    orient_score = 0.5  # baseline for Unknown or unmatched orientation
    orient_in_range = False
    orient_delta = None
    if assigned != "Unknown":
        low, high = ORIENT_RANGES[assigned]
        if low <= props["orientation"] <= high:
            orient_score = 1.0
            orient_in_range = True
        else:
            delta = min(abs(props["orientation"] - low), abs(props["orientation"] - high))
            orient_delta = round(delta, 2)
            orient_score = max(0.0, 1.0 - delta / 30.0)

    confidence = 0.2 * length_score + 0.3 * curvature_score + 0.5 * orient_score
    confidence = round(min(max(confidence, 0.0), 1.0), 3)

    confidence_reasons = {
        "region": assigned,
        "length": round(props["length"], 2),
        "length_score": round(length_score, 3),
        "curvature": round(props["curvature"], 4),
        "curvature_score": round(curvature_score, 3),
        "orientation": round(props["orientation"], 2),
        "orient_score": round(orient_score, 3),
        "orient_in_range": orient_in_range,
        "orient_delta_deg": orient_delta,
        "weights": "length×0.2 + curvature×0.3 + orient×0.5",
    }

    return assigned, confidence, confidence_reasons

def draw_overlay(base_img: np.ndarray, groups_info: List[Dict]):
    overlay = cv2.cvtColor(base_img.copy(), cv2.COLOR_GRAY2BGR)
    for grp in groups_info:
        color = COLOR_MAP.get(grp["class"], (255, 255, 255))
        for cnt in grp["contours"]:
            cv2.drawContours(overlay, [cnt], -1, color, 2)
        cx, cy = map(int, grp["centroid"].values())
        label = f"{grp['class']} ({grp['confidence']})"
        cv2.putText(overlay, label, (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return overlay

# -----------------------------------------------------------------------------
# Public API – callable from batch script
# -----------------------------------------------------------------------------

def run_classification(candidate_img_path: Path, meas_path: Path, normalized_img_path: Path,
                       output_dir: Path, show: bool = False) -> List[Dict]:
    """Run the full classification pipeline for a single image.

    Parameters
    ----------
    candidate_img_path: Path to the binary candidate mask (05_major_line_candidates.jpg).
    meas_path: Path to the JSON measurements produced by Milestone 5.2.
    normalized_img_path: Path to the 512×512 normalized palm image.
    output_dir: Directory where the JSON, overlay and comparison images will be saved.
    show: If True, OpenCV windows are displayed.

    Returns
    -------
    groups_info: List of dictionaries describing each classified group (including
                 class, confidence, centroid, length, orientation, curvature and
                 contour list).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    binary = load_candidate_image(candidate_img_path)
    measurements = load_measurements(meas_path)
    # Connected components + nearest‑centroid matching (same logic as original main)
    num_labels, labels, stats, comp_centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    contours: List[np.ndarray] = []
    centroids: List[Tuple[float, float]] = []
    angles: List[float] = []
    matched_component_ids = set()
    for meas in measurements:
        mx = float(meas["centroid"]["x"])
        my = float(meas["centroid"]["y"])
        best_id = None
        best_dist = float('inf')
        for comp_id in range(1, num_labels):
            if comp_id in matched_component_ids:
                continue
            cx, cy = comp_centroids[comp_id]
            d = euclidean((mx, my), (cx, cy))
            if d < best_dist:
                best_dist = d
                best_id = comp_id
        if best_id is None:
            continue
        matched_component_ids.add(best_id)
        comp_mask = (labels == best_id).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = cnts[0]
        contours.append(cnt)
        centroids.append((mx, my))
        if len(cnt) >= 5:
            angles.append(cv2.fitEllipse(cnt)[2])
        else:
            angles.append(0.0)

    groups_idx = build_graph(centroids, angles)
    groups_info: List[Dict] = []
    for gid, idxs in enumerate(groups_idx):
        props = group_properties(idxs, measurements, contours)
        line_class, confidence, confidence_reasons = classify_group(props)
        group_entry = {
            "id": gid,
            "class": line_class,
            "confidence": confidence,
            "confidence_reasons": confidence_reasons,
            "component_ids": [measurements[i]["id"] for i in idxs],
            "centroid": {"x": props["centroid"][0], "y": props["centroid"][1]},
            "length": props["length"],
            "orientation": props["orientation"],
            "curvature": props["curvature"],
            "contours": [contours[i] for i in idxs],
        }
        groups_info.append(group_entry)

    # Save JSON (without heavy contour data)
    json_path = output_dir / "semantic_line_classification.json"
    serialisable = []
    for g in groups_info:
        g_copy = g.copy()
        g_copy.pop("contours")
        serialisable.append(g_copy)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2)


    # Visual overlay
    norm_img = cv2.imread(str(normalized_img_path), cv2.IMREAD_GRAYSCALE)
    if norm_img is None:
        raise FileNotFoundError(f"Normalized palm image not found: {normalized_img_path}")
    overlay = draw_overlay(norm_img, groups_info)
    overlay_path = output_dir / "07_line_grouping_overlay.jpg"
    cv2.imwrite(str(overlay_path), overlay)

    # Comparison image (candidates vs overlay)
    cand_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    comparison = np.hstack([cand_color, overlay])
    comp_path = output_dir / "region_classification_comparison.jpg"
    cv2.imwrite(str(comp_path), comparison)

    if show or SHOW_WINDOWS:
        cv2.imshow("Candidates", cand_color)
        cv2.imshow("Classification Overlay", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return groups_info

# -----------------------------------------------------------------------------
# CLI entry point – retains original behaviour
# -----------------------------------------------------------------------------

def main(args):
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "processed"
    candidate_img_path = data_dir / "line_candidates" / "05_major_line_candidates.jpg"
    meas_path = data_dir / "line_candidates" / "candidate_measurements.json"
    normalized_img_path = data_dir / "palm_512.jpg"
    out_dir = data_dir / "line_classification"
    run_classification(candidate_img_path, meas_path, normalized_img_path, out_dir, show=args.show)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Group line candidates and assign preliminary palm‑line classes.")
    parser.add_argument("--show", action="store_true", help="Display OpenCV windows for debugging.")
    parser.add_argument("--proximity", type=float, default=PROXIMITY_DIST, help="Proximity distance threshold.")
    parser.add_argument("--angle", type=float, default=ANGLE_TOLERANCE, help="Angle tolerance (degrees).")
    args = parser.parse_args()
    PROXIMITY_DIST = args.proximity
    ANGLE_TOLERANCE = args.angle
    main(args)
