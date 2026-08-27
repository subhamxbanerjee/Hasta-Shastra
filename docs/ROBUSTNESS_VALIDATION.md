# Phase 6.3 — End-to-End Robustness Validation

## Test Environment
- **Date**: August 27, 2026
- **Backend**: FastAPI running on Python 3.13, MediaPipe 0.10.21, OpenCV 4.11
- **Frontend**: Vite + React Next.js
- **Dataset**: `palm-ai/data/raw/images/` + custom edge case images (`no_hand.jpg`, `corrupt.jpg`, `README.md`)

## Test Matrix & Results

| Test Category | Image / File | Expected Status | Actual Status | Result / Error Code |
|--------------|--------------|-----------------|---------------|---------------------|
| Valid Palm 1 | `IMG_0005.jpg` | Success | Success | 3 major lines, 10 groups, 50.5% avg conf. |
| Valid Palm 2 | `IMG_0016.jpg` | Success | Success | 4 major lines, 10 groups, 59.5% avg conf. |
| Valid Palm 3 | `IMG_0020.jpg` | Success | Success | 3 major lines, 7 groups, 57.4% avg conf. |
| No Hand | `no_hand.jpg` | Failed | Failed | `NO_HAND_DETECTED` (graceful failure) |
| Invalid Type | `README.md` | Failed | Failed | `INVALID_FORMAT` (graceful HTTP 400) |
| Corrupt Image | `corrupt.jpg` | Failed | Failed | `CORRUPT_IMAGE` (graceful HTTP 400) |

## API Request Isolation Verification
- **Verified**: Each API request dynamically generates a unique identifier (e.g., `api-8ed2369939d3`).
- **Verified**: Independent output directories (`input`, `normalization`, `line_enhancement`, `line_candidates`, `line_classification`, `features`) are populated cleanly for every single request without cross-contamination.

## Regression Validation (6-Image Baseline)
- Total images processed: 8
- Successful: 6
- Failed: 2 (`no_hand.jpg`, `corrupt.jpg` — expected failures)
- Baseline Phase 5.8 candidate metrics are exactly identical. **No regressions.**

## Bugs Found & Fixed
1. **Raw Exceptions Exposed to Frontend**: `backend/services/analysis_service.py` was returning raw stack traces and exception strings (e.g., MediaPipe `RuntimeError`) directly in the API JSON. 
   - **Fix**: Implemented a string-mapping catch block that maps MediaPipe failures to `NO_HAND_DETECTED` and all others to `PIPELINE_ERROR`. Error messages sent to the frontend are now completely user-friendly ("No clear palm was detected...", "An unexpected error occurred..."), while raw errors are logged securely to the backend console.

## Known Limitations (Intentionally Not Changed)
1. CV Heuristics struggle slightly with extremely dim or very rotated hands (producing lower confidence or fewer groupings). These represent hardware/environmental limitations to be addressed by future deep learning models rather than rule-based adjustments.
2. Extremely large fragments split by heavy noise might occasionally be grouped as separate candidates, which is a known limitation of the current spatial grouping heuristics.

## Final Readiness Assessment
The system is fully robust, effectively isolated per API request, prevents pipeline crashes via FastAPI `HTTPException` validation, and successfully displays CV results on the frontend. Error propagation is now fully secure and user-friendly.

**Readiness**: Phase 6 Complete. Ready for Next Steps.
