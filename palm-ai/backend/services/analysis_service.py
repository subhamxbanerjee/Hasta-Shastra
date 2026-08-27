import uuid
import time
from pathlib import Path
from typing import Dict, Tuple

from backend.schemas import PalmFeatures, ErrorDetail

# Import CV pipeline functions
import sys
# Resolve the root directory of the project to allow importing scripts module
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.palm_normalization import run_normalization
from scripts.palm_line_enhancement import run_enhancement
from scripts.palm_line_candidates import run_candidate_extraction
from scripts.palm_line_classification import run_classification
from scripts.palm_feature_extraction import run_feature_extraction

# Directory for storing API request outputs
API_REQUESTS_DIR = PROJECT_ROOT / "data" / "processed" / "api_requests"

def process_palm_image(image_bytes: bytes, filename: str) -> Tuple[str, str, Dict]:
    """
    Orchestrates the CV pipeline for a single API request.
    Returns (request_id, status, result_or_error_dict)
    """
    request_id = f"api-{uuid.uuid4().hex[:12]}"
    req_dir = API_REQUESTS_DIR / request_id
    
    # Subdirectories for isolated processing
    input_dir = req_dir / "input"
    norm_dir = req_dir / "normalization"
    enh_dir = req_dir / "line_enhancement"
    cand_dir = req_dir / "line_candidates"
    class_dir = req_dir / "line_classification"
    feat_dir = req_dir / "features"
    
    for d in [input_dir, norm_dir, enh_dir, cand_dir, class_dir, feat_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    # Save the uploaded file
    safe_filename = filename.replace("/", "_").replace("\\", "_")
    input_path = input_dir / safe_filename
    
    try:
        with open(input_path, "wb") as f:
            f.write(image_bytes)
    except Exception as e:
        return request_id, "failed", {"code": "FILE_SAVE_ERROR", "message": f"Could not save uploaded file: {str(e)}"}
        
    # Execute pipeline sequentially
    try:
        # Stage 1: Normalization
        palm_512_path = run_normalization(
            input_image=input_path,
            output_dir=norm_dir,
        )
        
        # Stage 2: Line Enhancement
        cleaned_candidates_path = run_enhancement(
            normalized_img_path=palm_512_path,
            output_dir=enh_dir,
            show=False
        )
        
        # Stage 3: Candidate Extraction
        candidate_img_path = run_candidate_extraction(
            enhanced_img_path=cleaned_candidates_path,
            output_dir=cand_dir,
            normalized_img_path=palm_512_path,
            show=False
        )
        meas_path = cand_dir / "candidate_measurements.json"
        
        # Stage 4: Classification
        groups = run_classification(
            candidate_img_path=candidate_img_path,
            meas_path=meas_path,
            normalized_img_path=palm_512_path,
            output_dir=class_dir,
            show=False
        )
        
        # Stage 5: Feature Extraction
        features = run_feature_extraction(
            image_id=safe_filename,
            classification_results=groups,
            normalized_img_path=palm_512_path,
            output_dir=feat_dir
        )
        
        return request_id, "success", features
        
    except Exception as e:
        # We catch all exceptions from the underlying CV scripts
        # and wrap them in a clean error payload.
        # Python traceback is deliberately omitted from the API response.
        error_msg = str(e)
        if "No hand landmarks detected" in error_msg:
            code = "NO_HAND_DETECTED"
        else:
            code = "PIPELINE_ERROR"
            
        return request_id, "failed", {"code": code, "message": error_msg}
