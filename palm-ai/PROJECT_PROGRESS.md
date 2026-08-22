# PalmVerse — Project Progress

## Phase 1 — Python + Image Fundamentals

### ✅ Milestone 1.1 — Project Structure Created
**Date:** 2026-08-20

**What was done:**
- Created full palm-ai/ folder structure
- Created `src/config.py` — central path configuration
- Created `src/__init__.py` — makes src a Python package
- Created `scripts/test_image.py` — image loading and inspection
- Created `scripts/preprocess.py` — placeholder for Phase 1 tasks
- Created `requirements.txt`
- Created README files

**Concepts introduced:**
- Project structure and why modular layout matters
- Single Source of Truth (config.py pattern)
- Python packages (`__init__.py`)
- `os.path` for cross-platform path handling

---

### ✅ Milestone 1.2 — Load and Inspect an Image
**Date:** 2026-08-20

**What was done:**
- Placed `test_palm.jpg` in `data/raw/images/`
- Ran `scripts/test_image.py` successfully
- Confirmed image shape: `(3046, 3736, 3)` — height, width, channels
- Confirmed dtype: `uint8` (values 0–255)
- Understood BGR channel order vs RGB
- Verified `None` check after `cv2.imread()`

**Concepts mastered:**
- `cv2.imread()` returns a NumPy array or `None`
- `image.shape` → `(H, W, C)` ordering
- `dtype=uint8` means integer pixel values 0–255
- OpenCV's BGR channel order (historical quirk)
- Importance of defensive `None` check

---

### ✅ Milestone 1.3 — Grayscale + Contrast Enhancement
**Date:** 2026-08-20

**What was done:**
- Ran `scripts/preprocess.py` successfully
- Converted BGR image `(3046, 3736, 3)` → grayscale `(3046, 3736)`
- Applied global histogram equalization: mean shifted from 133.4 → 128.5
- Applied CLAHE (clipLimit=2.0, tile 8×8): mean shifted from 133.4 → 146.7
- Saved 3 individual processed images + 1 comparison grid to `data/processed/`

**Concepts mastered:**
- `cv2.cvtColor(img, COLOR_BGR2GRAY)` — weighted brightness collapse
- Grayscale removes channel dimension: shape goes from `(H, W, 3)` → `(H, W)`
- `cv2.equalizeHist()` — global contrast redistribution
- `cv2.createCLAHE()` — per-tile adaptive enhancement, better for uneven lighting
- `cv2.COLOR_GRAY2BGR` — promote grayscale to 3-ch for `np.hstack()` compatibility
- `cv2.resize()` with `INTER_AREA` — correct interpolation when shrinking
- `cv2.imwrite()` — saves NumPy arrays as image files

---

### ✅ Milestone 1.4 — Basic Image Quality Assessment
**Date:** 2026-08-20

**What was done:**
- Created and ran `scripts/check_quality.py` successfully
- Measured: width=3736px, height=3046px, mean brightness=133.4, blur score=22.60
- Blur check flagged WARNING (score 22.60 < threshold 50)
- Identified that blur threshold needs calibration: smooth palm skin has low
  Laplacian variance even in a sharp photo — threshold must be tuned per use case

**Concepts mastered:**
- Resolution = width × height in total pixels
- `np.mean(gray)` → mean brightness as an exposure proxy
- `cv2.Laplacian(img, cv2.CV_64F).var()` → sharpness metric
- `CV_64F` needed to preserve negative Laplacian values (uint8 would clip them)
- Quality thresholds are heuristics — must be calibrated with real-world data
- Validating input before inference prevents silent bad outputs

---

## ✅ PHASE 1 COMPLETE — 2026-08-20

**Phase 1 Summary:**
All four milestones completed. You can now:
- Load, inspect, and validate any image from Python
- Understand NumPy image arrays: shape, dtype, channel order
- Convert between colour spaces (BGR, grayscale)
- Apply and compare contrast enhancement methods (EQ, CLAHE)
- Assess image quality programmatically with explainable metrics

---

## Phase 2 — OpenCV Image Processing Experiments

### ✅ Milestone 2.1 — Thresholding Experiments
**Date:** 2026-08-20

**What was done:**
- Ran `scripts/threshold_experiment.py` successfully
- Applied 5 methods: binary (T=127), inverse binary, Otsu (T=138 auto), adaptive mean, adaptive Gaussian
- Saved 6 individual outputs + 2×3 comparison grid to `data/processed/thresholds/`
- Visually confirmed: thresholding reveals dark structures but cannot label them

**Key result:**
- Otsu auto-selected T=138 vs manual T=127 — Otsu found a slightly higher threshold
- Adaptive methods preserved more local detail under uneven illumination

**Concepts mastered:**
- `cv2.threshold()` → global binary, inverse, Otsu
- Otsu minimises intra-class variance to auto-find optimal T
- `cv2.adaptiveThreshold()` → local T per `blockSize×blockSize` neighbourhood
- Gaussian adaptive weighting smoother than mean on fine structures
- `THRESH_BINARY_INV` makes dark lines white — correct for downstream morphology
- CLAHE before thresholding normalises uneven illumination

**Critical insight:**
Thresholding is pixel-level intensity classification. It has no semantic understanding. It cannot tell which dark structure is a Life Line, Head Line, or shadow.

---

### ✅ Milestone 2.2 — Gaussian Blur + Morphological Operations
**Date:** 2026-08-21

**What was done:**
- Ran `scripts/morphology_experiment.py` successfully
- Resized image to 1024px wide before operations (performance lesson)
- Compared Gaussian blur at (5,5) vs (15,15): mean diff = 2.75px
- Applied adaptive Gaussian threshold (block=35, C=8): 105,361 white px (12.3%)
- Applied all four morphological operations with ELLIPSE (3,3) kernel:
  - Erosion:  69,477 px  (−35,884 removed)
  - Dilation: 146,338 px (+40,977 added)
  - Opening:  98,696 px  (−6,665 net — small noise removed, structures intact)
  - Closing:  106,666 px (+1,305 net — tiny gaps filled)

**Concepts mastered:**
- `cv2.GaussianBlur()` — weighted neighbourhood smoothing, sigma auto from ksize
- Resize BEFORE heavy operations — work at needed resolution, not camera resolution
- `cv2.getStructuringElement()` — defines morphology neighbourhood shape
- Erosion: all neighbours must be white → shrinks structures, removes speckles
- Dilation: any neighbour white → grows structures, fills holes
- Opening = erode→dilate → removes small isolated blobs
- Closing = dilate→erode → fills small gaps between nearby structures
- MORPH_ELLIPSE gentler on curves than MORPH_RECT
- Kernel size has no universal value — depends on resolution and structure size

**Critical insight:**
Morphological operations manipulate pixel shape and size. They have no semantic understanding of what a structure represents.

---

### ⏳ Milestone 2.3 — Canny Edge Detection
**Status:** In progress

**Goal:**
- Run `scripts/canny_experiment.py`
- Compare three blur levels (none, 5×5, 11×11) as Canny input
- Compare three threshold pairs (low/medium/high sensitivity)
- Understand gradient, NMS, double-threshold, hysteresis
- Save outputs to `data/processed/edges/`
- Build four labeled comparison grids (preprocessing, threshold, blur impact, full)

---

### 🔲 Milestone 2.3 — Canny Edge Detection
_Not started_

### 🔲 Milestone 2.4 — Contour Detection + Visualization
_Not started_

## Phase 3 — PalmVerse Web Application (Frontend MVP)
**Status:** ✅ Completed

- Scaffolded React + TypeScript + Vite project
- Created Tailwind design system (mystic theme, glow effects)
- Implemented responsive pages: Landing, Details, Capture, Analysis, Results
- Implemented `PalmReadingContext` for global state management
- Created deterministic mock analysis engine

---

## Phase 4 — Real Palm Detection

### ⏳ Milestone 4.1 — MediaPipe Hand Landmarks + Palm Crop
**Status:** In progress

**Goal:**
- Use MediaPipe to detect hand landmarks
- Convert normalized coordinates to pixel coordinates
- Extract a clean palm crop excluding fingers
- Normalize palm crop to 512x512 for future ML models

---

## Phase 5 — Dataset Creation + Annotation
_Not started_

## Phase 6 — Semantic Segmentation
_Not started_

## Phase 7 — Feature Extraction
_Not started_

## Phase 8 — Interpretation Engine
_Not started_

## Phase 9 — FastAPI Integration
_Not started_
