#!/usr/bin/env python3
"""palm_line_classification.py
Phase 5 – Milestone 5.3

This script groups line‑candidate components (produced in Milestone 5.2) into larger line paths and assigns a **preliminary** class
(LifeLine, HeadLine, HeartLine, FateLine or Unknown) with a confidence score.

The implementation follows the heuristics described in the implementation plan.
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np

# -----------------------------------------------------------------------------
# Heuristic constants – can be overridden from the CLI
# -----------------------------------------------------------------------------
PROXIMITY_DIST = 15          # max Euclidean distance (pixels) between centroids
ANGLE_TOLERANCE = 20          # max angle difference (degrees) between orientations
MIN_GROUP_LENGTH = 40        # discard groups shorter than this (pixels)
CURVATURE_THRESHOLD = 0.25   # higher values mean more curved
SHOW_WINDOWS = False         # set True for debugging visualisation

# Approximate region boxes for the four classic palm lines (in 512×512 normalized space)
CLASS_REGION_BOUNDS = {
    "HeartLine": {"x": (150, 380), "y": (350, 460)},
    "HeadLine":  {"x": (120, 350), "y": (210, 320)},
    "LifeLine":  {"x": ( 60, 250), "y": (380, 500)},
    "FateLine":  {"x": (380, 470), "y": (260, 380)},
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
# Main processing
# -----------------------------------------------------------------------------

def load_measurements(meas_path: Path) -> List[Dict]:
    with open(meas_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

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

def build_graph(centroids: List[Tuple[float, float]], angles: List[float]):
    n = len(centroids)
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            if euclidean(centroids[i], centroids[j]) <= PROXIMITY_DIST and \
               angle_diff(angles[i], angles[j]) <= ANGLE_TOLERANCE:
                uf.union(i, j)
    groups = {}
    for idx in range(n):
        root = uf.find(idx)
        groups.setdefault(root, []).append(idx)
    return list(groups.values())

def group_properties(group_idxs: List[int], measurements: List[Dict], contours: List[np.ndarray]):
    # Compute aggregated properties for a group of component indices
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
    cx, cy = props["centroid"]
    assigned = "Unknown"
    for line, bounds in CLASS_REGION_BOUNDS.items():
        if bounds["x"][0] <= cx <= bounds["x"][1] and bounds["y"][0] <= cy <= bounds["y"][1]:
            assigned = line
            break
    ORIENT_RANGES = {
        "HeartLine": (20, 40),
        "HeadLine":  (10, 30),
        "LifeLine":  (70, 110),
        "FateLine":  (30, 70),
    }
    length_score = min(props["length"] / 1500.0, 1.0)
    curvature_score = 1.0 - min(props["curvature"] / CURVATURE_THRESHOLD, 1.0)
    orient_score = 0.5
    if assigned != "Unknown":
        low, high = ORIENT_RANGES[assigned]
        if low <= props["orientation"] <= high:
            orient_score = 1.0
        else:
            delta = min(abs(props["orientation"] - low), abs(props["orientation"] - high))
            orient_score = max(0.0, 1.0 - delta / 30.0)
    confidence = 0.2 * length_score + 0.3 * curvature_score + 0.5 * orient_score
    confidence = round(min(max(confidence, 0.0), 1.0), 3)
    return assigned, confidence

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

def main(args):
    # Use the palm-ai directory as the project root (scripts -> palm-ai)
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "processed"
    # Load the binary mask of major line candidates from line_candidates folder
    candidate_img_path = data_dir / "line_candidates" / "05_major_line_candidates.jpg"
    meas_path = data_dir / "line_candidates" / "candidate_measurements.json"
    normalized_img_path = data_dir / "palm_512.jpg"

    binary = load_candidate_image(candidate_img_path)
    measurements = load_measurements(meas_path)
    # ------------------------------------------------------------
    # Component extraction using nearest component centroid mapping
    # ------------------------------------------------------------
    # Obtain connected components from the binary candidate mask
    num_labels, labels, stats, comp_centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    # comp_centroids[0] is background centroid (ignored)
    # Prepare containers
    contours = []
    centroids = []
    angles = []
    unmatched_measurements = []
    matched_component_ids = set()
    # For each measurement, find the nearest component centroid (excluding already matched)
    for meas in measurements:
        mx = float(meas["centroid"]["x"])
        my = float(meas["centroid"]["y"])
        best_id = None
        best_dist = float('inf')
        for comp_id in range(1, num_labels):  # skip background 0
            if comp_id in matched_component_ids:
                continue
            cx, cy = comp_centroids[comp_id]
            d = euclidean((mx, my), (cx, cy))
            if d < best_dist:
                best_dist = d
                best_id = comp_id
        if best_id is None:
            unmatched_measurements.append(meas["id"])
            continue
        matched_component_ids.add(best_id)
        # Extract contour for the matched component
        comp_mask = (labels == best_id).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            unmatched_measurements.append(meas["id"])
            continue
        cnt = cnts[0]
        contours.append(cnt)
        centroids.append((mx, my))  # use measurement centroid for downstream processing
        # Orientation via ellipse fitting (fallback to 0 if insufficient points)
        if len(cnt) >= 5:
            angles.append(cv2.fitEllipse(cnt)[2])
        else:
            angles.append(0.0)
    # Validation summary
    total_measurements = len(measurements)
    total_components = num_labels - 1  # exclude background
    matched_candidates = len(contours)
    unmatched_components = total_components - matched_candidates
    print("--- Alignment Validation Summary ---")
    print(f"Total measurement records      : {total_measurements}")
    print(f"Total connected components     : {total_components}")
    print(f"Successfully matched candidates: {matched_candidates}")
    print(f"Unmatched measurements (ids)   : {unmatched_measurements}")
    print(f"Unmatched components count    : {unmatched_components}")
    # Continue with grouped processing using the prepared lists
    groups_idx = build_graph(centroids, angles)
    groups_info = []
    for gid, idxs in enumerate(groups_idx, start=1):
        if not idxs:
            continue
        props = group_properties(idxs, measurements, contours)
        if props["length"] < MIN_GROUP_LENGTH:
            continue
        line_class, confidence = classify_group(props)
        group_entry = {
            "id": gid,
            "class": line_class,
            "confidence": confidence,
            "component_ids": [measurements[i]["id"] for i in idxs],
            "centroid": {"x": props["centroid"][0], "y": props["centroid"][1]},
            "length": props["length"],
            "orientation": props["orientation"],
            "curvature": props["curvature"],
            "contours": [contours[i] for i in idxs],
        }
        groups_info.append(group_entry)
    output_dir = data_dir / "line_classification"
    os.makedirs(output_dir, exist_ok=True)
    json_path = output_dir / "semantic_line_classification.json"
    serialisable = []
    for g in groups_info:
        g_copy = g.copy()
        g_copy.pop("contours")
        serialisable.append(g_copy)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2)
    print(f"[INFO] Classification JSON saved to {json_path}")
    norm_img = cv2.imread(str(normalized_img_path), cv2.IMREAD_GRAYSCALE)
    if norm_img is None:
        raise FileNotFoundError(f"Normalized palm image not found: {normalized_img_path}")
    overlay = draw_overlay(norm_img, groups_info)
    overlay_path = output_dir / "07_line_grouping_overlay.jpg"
    cv2.imwrite(str(overlay_path), overlay)
    print(f"[INFO] Overlay image saved to {overlay_path}")
    cand_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    comparison = np.hstack([cand_color, overlay])
    comp_path = output_dir / "region_classification_comparison.jpg"
    cv2.imwrite(str(comp_path), comparison)
    print(f"[INFO] Comparison image saved to {comp_path}")
    if SHOW_WINDOWS or args.show:
        cv2.imshow("Candidates", cand_color)
        cv2.imshow("Classification Overlay", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Group line candidates and assign preliminary palm‑line classes.")
    parser.add_argument("--show", action="store_true", help="Display OpenCV windows for debugging.")
    parser.add_argument("--proximity", type=float, default=PROXIMITY_DIST, help="Proximity distance threshold.")
    parser.add_argument("--angle", type=float, default=ANGLE_TOLERANCE, help="Angle tolerance (degrees).")
    args = parser.parse_args()
    PROXIMITY_DIST = args.proximity
    ANGLE_TOLERANCE = args.angle
    main(args)
