import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import cv2

# Ensure clean ASCII output for the logger
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

MAJOR_LINES = ["LifeLine", "HeadLine", "HeartLine", "FateLine"]

def calculate_weighted_average(fragments: List[Dict], key: str) -> Optional[float]:
    """Calculate the length-weighted average of a specific property (e.g., orientation)."""
    if not fragments:
        return None
    total_length = sum(f.get("length", 0.0) for f in fragments)
    if total_length == 0:
        return None
    weighted_sum = sum(f.get(key, 0.0) * f.get("length", 0.0) for f in fragments)
    return round(weighted_sum / total_length, 2)

def extract_line_features(line_name: str, classification_results: List[Dict], normalized_height: int) -> Dict:
    """Extract structured features for a single major palm line."""
    fragments = [g for g in classification_results if g.get("class") == line_name]
    
    if not fragments:
        return {
            "detected": False,
            "num_fragments": 0,
            "max_confidence": None,
            "total_length": 0.0,
            "normalized_length": 0.0,
            "weighted_orientation": None,
            "weighted_curvature": None
        }

    total_length = sum(f.get("length", 0.0) for f in fragments)
    max_conf = max((f.get("confidence", 0.0) for f in fragments), default=0.0)
    
    return {
        "detected": True,
        "num_fragments": len(fragments),
        "max_confidence": round(max_conf, 3),
        "total_length": round(total_length, 2),
        "normalized_length": round(total_length / normalized_height, 4) if normalized_height else 0.0,
        "weighted_orientation": calculate_weighted_average(fragments, "orientation"),
        "weighted_curvature": calculate_weighted_average(fragments, "curvature")
    }

def run_feature_extraction(image_id: str, classification_results: List[Dict], normalized_img_path: Path, output_dir: Path) -> Dict:
    """
    Transform raw CV classification results into a structured feature schema.
    
    Parameters
    ----------
    image_id : str
        The unique identifier for the image.
    classification_results : List[Dict]
        The output groups from run_classification().
    normalized_img_path : Path
        Path to the normalized palm image (used for metadata/dimensions).
    output_dir : Path
        The directory where palm_features.json will be saved.
        
    Returns
    -------
    Dict
        The structured feature representation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract image metadata
    norm_width, norm_height = 512, 512 # Fallback defaults
    norm_img = cv2.imread(str(normalized_img_path), cv2.IMREAD_GRAYSCALE)
    if norm_img is not None:
        norm_height, norm_width = norm_img.shape[:2]
    else:
        logger.warning(f"Could not load normalized image {normalized_img_path}. Using default 512x512.")
        
    # Extract line-specific features
    lines_features = {}
    detected_lines_count = 0
    total_major_groups = 0
    
    for line_name in MAJOR_LINES:
        line_feat = extract_line_features(line_name, classification_results, norm_height)
        lines_features[line_name] = line_feat
        if line_feat["detected"]:
            detected_lines_count += 1
            total_major_groups += line_feat["num_fragments"]
            
    # Calculate summary metrics (only over major lines, excluding Unknown)
    major_fragments = [g for g in classification_results if g.get("class") in MAJOR_LINES]
    if major_fragments:
        avg_conf = sum(g.get("confidence", 0.0) for g in major_fragments) / len(major_fragments)
        min_conf = min(g.get("confidence", 0.0) for g in major_fragments)
    else:
        avg_conf = 0.0
        min_conf = 0.0

    features = {
        "image_id": image_id,
        "image_metadata": {
            "normalized_width": norm_width,
            "normalized_height": norm_height
        },
        "lines": lines_features,
        "summary": {
            "detected_major_lines": detected_lines_count,
            "total_major_line_groups": total_major_groups,
            "average_confidence": round(avg_conf, 3),
            "minimum_confidence": round(min_conf, 3)
        }
    }
    
    # Write to JSON
    json_path = output_dir / "palm_features.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2)
        
    logger.info(f"  [OK] Extracted structured features -> {json_path.name}")
    
    return features

if __name__ == "__main__":
    pass
