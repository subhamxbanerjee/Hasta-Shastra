#!/usr/bin/env python3
"""palm_multi_image_validation.py
Phase 5 – Milestone 5.4 / 5.5 / 5.6

Batch validation of the palm-line pipeline over all raw images.

Each input image is processed through a completely isolated pipeline:

  data/processed/multi_image_validation/<image_stem>/
      normalization/          <- palm_normalization outputs + palm_512.jpg
      line_enhancement/       <- palm_line_enhancement outputs
      line_candidates/        <- palm_line_candidates outputs + candidate_measurements.json
      line_classification/    <- palm_line_classification outputs + semantic_line_classification.json

No intermediate file is shared between images. Raw images are read-only.

An aggregated JSON summary (evaluation_summary.json) and CSV summary
(evaluation_summary.csv) are written to the root of the output directory.
A human-readable robustness_report.txt is also generated.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import List, Dict

# ---------------------------------------------------------------------------
# Import reusable pipeline functions (same package directory)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from palm_normalization import run_normalization
from palm_line_enhancement import run_enhancement
from palm_line_candidates import run_candidate_extraction
from palm_line_classification import run_classification


# ---------------------------------------------------------------------------
# Helper: per-image processing
# ---------------------------------------------------------------------------

def process_image(
    img_path: Path,
    out_root: Path,
    conf_thresh: float,
    show: bool,
) -> Dict:
    """Run the full isolated pipeline for a single image.

    Returns a record dict with per-image metrics and status.
    """
    stem = img_path.stem
    image_id = stem  # stable, file-name-based identifier

    # Per-image isolated output directories
    base_dir = out_root / stem
    norm_dir   = base_dir / "normalization"
    enh_dir    = base_dir / "line_enhancement"
    cand_dir   = base_dir / "line_candidates"
    class_dir  = base_dir / "line_classification"

    for d in [norm_dir, enh_dir, cand_dir, class_dir]:
        d.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    print(f"\n{'='*68}")
    print(f"  Processing: {img_path.name}  (image_id={image_id})")
    print(f"{'='*68}")

    # ------------------------------------------------------------------
    # Stage 1 – Normalization
    # ------------------------------------------------------------------
    print(f"[STAGE 1] Normalization -> {norm_dir}")
    try:
        palm_512_path = run_normalization(
            input_image=img_path,
            output_dir=norm_dir,
        )
    except Exception as exc:
        duration = round(time.time() - start_time, 2)
        print(f"  [ERROR] Normalization failed: {exc}")
        return {
            "image_id": image_id,
            "image": img_path.name,
            "status": "failed",
            "stage": "normalization",
            "error": str(exc),
            "duration_sec": duration,
        }

    # ------------------------------------------------------------------
    # Stage 2 – Line Enhancement
    # ------------------------------------------------------------------
    print(f"[STAGE 2] Line Enhancement -> {enh_dir}")
    try:
        cleaned_candidates_path = run_enhancement(
            normalized_img_path=palm_512_path,
            output_dir=enh_dir,
            show=show,
        )
    except Exception as exc:
        duration = round(time.time() - start_time, 2)
        print(f"  [ERROR] Enhancement failed: {exc}")
        return {
            "image_id": image_id,
            "image": img_path.name,
            "status": "failed",
            "stage": "line_enhancement",
            "error": str(exc),
            "duration_sec": duration,
        }

    # ------------------------------------------------------------------
    # Stage 3 – Candidate Extraction
    # ------------------------------------------------------------------
    print(f"[STAGE 3] Candidate Extraction -> {cand_dir}")
    try:
        candidate_img_path = run_candidate_extraction(
            enhanced_img_path=cleaned_candidates_path,
            output_dir=cand_dir,
            normalized_img_path=palm_512_path,
            show=show,
        )
    except Exception as exc:
        duration = round(time.time() - start_time, 2)
        print(f"  [ERROR] Candidate extraction failed: {exc}")
        return {
            "image_id": image_id,
            "image": img_path.name,
            "status": "failed",
            "stage": "line_candidates",
            "error": str(exc),
            "duration_sec": duration,
        }

    meas_path = cand_dir / "candidate_measurements.json"

    # ------------------------------------------------------------------
    # Stage 4 – Classification
    # ------------------------------------------------------------------
    print(f"[STAGE 4] Classification -> {class_dir}")
    try:
        groups = run_classification(
            candidate_img_path=candidate_img_path,
            meas_path=meas_path,
            normalized_img_path=palm_512_path,
            output_dir=class_dir,
            show=show,
        )
    except Exception as exc:
        duration = round(time.time() - start_time, 2)
        print(f"  [ERROR] Classification failed: {exc}")
        return {
            "image_id": image_id,
            "image": img_path.name,
            "status": "failed",
            "stage": "line_classification",
            "error": str(exc),
            "duration_sec": duration,
        }

    # ------------------------------------------------------------------
    # Build per-image summary record
    # ------------------------------------------------------------------
    class_counts = {"LifeLine": 0, "HeadLine": 0, "HeartLine": 0, "FateLine": 0, "Unknown": 0}
    for g in groups:
        cls = g.get("class", "Unknown")
        class_counts[cls] = class_counts.get(cls, 0) + 1

    confidences = [g["confidence"] for g in groups if "confidence" in g]
    min_conf = round(min(confidences), 3) if confidences else 0.0
    max_conf = round(max(confidences), 3) if confidences else 0.0
    avg_conf = round(sum(confidences) / len(confidences), 3) if confidences else 0.0
    low_conf_count = sum(1 for c in confidences if c < conf_thresh)
    low_conf_ratio = round(low_conf_count / len(confidences), 3) if confidences else 0.0
    low_conf_candidates = [g.get("id") for g in groups if g.get("confidence", 1.0) < conf_thresh]

    # Load candidate count from JSON
    try:
        with open(meas_path, "r", encoding="utf-8") as mf:
            measurements = json.load(mf)
        num_candidates = len(measurements)
    except Exception:
        num_candidates = None

    duration = round(time.time() - start_time, 2)

    record = {
        "image_id": image_id,
        "image": img_path.name,
        "status": "success",
        "num_candidates": num_candidates,
        "num_groups": len(groups),
        "class_counts": class_counts,
        "min_confidence": min_conf,
        "avg_confidence": avg_conf,
        "max_confidence": max_conf,
        "low_confidence_count": low_conf_count,
        "low_confidence_ratio": low_conf_ratio,
        "low_confidence_candidates": low_conf_candidates,
        "output_dir": str(base_dir),
        "duration_sec": duration,
    }

    print(f"  [OK] {img_path.name} -> candidates={num_candidates}, groups={len(groups)}, "
          f"avg_conf={avg_conf:.3f}, min_conf={min_conf:.3f}")
    return record


# ---------------------------------------------------------------------------
# Main batch routine
# ---------------------------------------------------------------------------

def main(
    raw_dir: Path,
    out_root: Path,
    conf_thresh: float = 0.5,
    show: bool = False,
) -> None:
    # Diagnostic: show which Python interpreter is being used
    print(f"[DEBUG] Running with interpreter: {sys.executable}")
    print(f"[DEBUG] Python version: {sys.version.splitlines()[0]}")

    raw_dir = raw_dir.resolve()
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Gather supported image files
    supported_ext = {'.jpg', '.jpeg', '.png'}
    image_paths = sorted(
        p for p in raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() in supported_ext
    )
    print(f"[INFO] Found {len(image_paths)} supported image(s) in {raw_dir}")
    if not image_paths:
        print("[WARN] No supported raw images found.")
        return

    summary_records: List[Dict] = []

    for img_path in image_paths:
        record = process_image(img_path, out_root, conf_thresh, show)
        summary_records.append(record)

    # -----------------------------------------------------------------------
    # Write aggregated JSON summary
    # -----------------------------------------------------------------------
    successful = [r for r in summary_records if r.get("status") == "success"]
    failed = [r for r in summary_records if r.get("status") != "success"]

    overall = {
        "total_images": len(summary_records),
        "successful_images": len(successful),
        "failed_images": len(failed),
        "average_candidate_count": (
            sum(r.get("num_candidates", 0) or 0 for r in successful) / len(successful)
        ) if successful else 0,
        "min_candidate_count": min(
            (r.get("num_candidates", 0) or 0 for r in successful), default=0
        ),
        "max_candidate_count": max(
            (r.get("num_candidates", 0) or 0 for r in successful), default=0
        ),
        "average_avg_confidence": (
            sum(r.get("avg_confidence", 0) for r in successful) / len(successful)
        ) if successful else 0,
        "average_min_confidence": (
            sum(r.get("min_confidence", 0) for r in successful) / len(successful)
        ) if successful else 0,
        "class_distribution": {
            cls: sum(r.get("class_counts", {}).get(cls, 0) for r in successful)
            for cls in ["LifeLine", "HeadLine", "HeartLine", "FateLine", "Unknown"]
        },
        "candidate_count_per_image": {
            r["image_id"]: r.get("num_candidates") for r in successful
        },
        "avg_confidence_per_image": {
            r["image_id"]: r.get("avg_confidence") for r in successful
        },
        "min_confidence_per_image": {
            r["image_id"]: r.get("min_confidence") for r in successful
        },
        "groups_per_image": {
            r["image_id"]: r.get("num_groups") for r in successful
        },
    }

    json_path = out_root / "evaluation_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": overall, "records": summary_records}, f, indent=2)
    print(f"\n[INFO] JSON summary written to {json_path}")

    # -----------------------------------------------------------------------
    # Write CSV summary (easy-to-review comparison output)
    # -----------------------------------------------------------------------
    csv_path = out_root / "evaluation_summary.csv"
    csv_fields = [
        "image_id", "image", "status",
        "num_candidates", "num_groups",
        "min_confidence", "avg_confidence", "max_confidence",
        "low_confidence_count", "low_confidence_ratio",
        "LifeLine", "HeadLine", "HeartLine", "FateLine", "Unknown",
        "duration_sec",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for rec in summary_records:
            row = dict(rec)
            # Flatten class_counts into top-level columns
            for cls in ["LifeLine", "HeadLine", "HeartLine", "FateLine", "Unknown"]:
                row[cls] = rec.get("class_counts", {}).get(cls, 0)
            writer.writerow(row)
    print(f"[INFO] CSV summary written to {csv_path}")

    # -----------------------------------------------------------------------
    # Robustness report (human-readable txt)
    # -----------------------------------------------------------------------
    report_path = out_root / "robustness_report.txt"
    with open(report_path, "w", encoding="utf-8") as rpt:
        rpt.write("PHASE 5.6 - ROBUSTNESS REPORT\n")
        rpt.write("=" * 60 + "\n\n")
        for rec in summary_records:
            rpt.write(f"Image   : {rec.get('image')} (id={rec.get('image_id')})\n")
            rpt.write(f"Status  : {rec.get('status', '').upper()}\n")
            if rec.get("status") == "success":
                rpt.write(f"Candidates : {rec.get('num_candidates')}\n")
                rpt.write(f"Groups     : {rec.get('num_groups')}\n")
                rpt.write(f"min_conf   : {rec.get('min_confidence')}\n")
                rpt.write(f"avg_conf   : {rec.get('avg_confidence')}\n")
                rpt.write(f"max_conf   : {rec.get('max_confidence')}\n")
                rpt.write(f"low_conf_count : {rec.get('low_confidence_count')} "
                          f"(ratio={rec.get('low_confidence_ratio')})\n")
                cls_counts = rec.get("class_counts", {})
                rpt.write(
                    "Classes    : LifeLine={}, HeadLine={}, HeartLine={}, "
                    "FateLine={}, Unknown={}\n".format(
                        cls_counts.get("LifeLine", 0),
                        cls_counts.get("HeadLine", 0),
                        cls_counts.get("HeartLine", 0),
                        cls_counts.get("FateLine", 0),
                        cls_counts.get("Unknown", 0),
                    )
                )
                rpt.write(f"Duration   : {rec.get('duration_sec')}s\n")
            else:
                rpt.write(f"Stage failed: {rec.get('stage')}\n")
                rpt.write(f"Error       : {rec.get('error')}\n")
            rpt.write("\n")

        rpt.write("-" * 60 + "\n")
        rpt.write("OVERALL FINDINGS\n")
        rpt.write(f"Images processed   : {len(summary_records)}\n")
        rpt.write(f"Successful         : {len(successful)}\n")
        rpt.write(f"Failed             : {len(failed)}\n")
        if successful:
            rpt.write(f"Avg candidate count: {overall['average_candidate_count']:.2f}\n")
            rpt.write(f"Avg avg_confidence : {overall['average_avg_confidence']:.3f}\n")
            rpt.write(f"Avg min_confidence : {overall['average_min_confidence']:.3f}\n")
            rpt.write("\nPer-image candidate counts:\n")
            for img_id, cnt in overall["candidate_count_per_image"].items():
                rpt.write(f"  {img_id}: {cnt}\n")
            rpt.write("\nPer-image avg confidence:\n")
            for img_id, conf in overall["avg_confidence_per_image"].items():
                rpt.write(f"  {img_id}: {conf}\n")
        rpt.write("\n")
    print(f"[INFO] Robustness report written to {report_path}")

    # -----------------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------------
    print("\n[INFO] ---- Validation Summary ----")
    print(f"[INFO] Total images processed : {len(summary_records)}")
    print(f"[INFO] Successful             : {len(successful)}")
    print(f"[INFO] Failed                 : {len(failed)}")
    if successful:
        print(f"\n[INFO] Per-image metrics:")
        print(f"  {'image_id':<20} {'candidates':>10} {'groups':>7} {'avg_conf':>10} {'min_conf':>10}")
        print(f"  {'-'*20} {'-'*10} {'-'*7} {'-'*10} {'-'*10}")
        for r in successful:
            print(
                f"  {r['image_id']:<20} {str(r.get('num_candidates', 'N/A')):>10} "
                f"{str(r.get('num_groups', 'N/A')):>7} "
                f"{r.get('avg_confidence', 0):>10.3f} "
                f"{r.get('min_confidence', 0):>10.3f}"
            )
    if failed:
        print(f"\n[WARN] Failed images:")
        for r in failed:
            print(f"  {r['image']}: stage={r.get('stage')}, error={r.get('error')}")

    # Check for metric variation (Phase 5.6 isolation verification)
    if len(successful) > 1:
        candidate_counts = [r.get("num_candidates", 0) or 0 for r in successful]
        if len(set(candidate_counts)) == 1:
            print("\n[WARN] All images produced identical candidate counts — "
                  "possible pipeline isolation issue. Check intermediate outputs.")
        else:
            print("\n[OK] Image-specific variation detected in candidate counts — "
                  "per-image isolation is working correctly.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run batch validation of the palm-line pipeline over all raw images."
    )
    parser.add_argument(
        "--raw-dir", "--raw_dir",
        dest="raw_dir",
        type=str,
        default="data/raw/images",
        help="Directory containing raw palm images (default: data/raw/images).",
    )
    parser.add_argument(
        "--out-dir", "--out_root",
        dest="out_dir",
        type=str,
        default="data/processed/multi_image_validation",
        help="Root directory for per-image results and summaries.",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.5,
        help="Confidence threshold for flagging low-confidence candidates.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Display OpenCV windows per image (headless by default).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    main(
        raw_dir=project_root / args.raw_dir,
        out_root=project_root / args.out_dir,
        conf_thresh=args.conf_thresh,
        show=args.show,
    )
