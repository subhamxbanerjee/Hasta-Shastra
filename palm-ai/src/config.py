"""
config.py — Central configuration for the palm-ai project.

WHY THIS FILE EXISTS:
  Instead of hardcoding paths in every script, we define them once here.
  Every script imports from this file. This means:
    - If you rename a folder, you only change it in ONE place.
    - Your scripts stay clean and readable.

CONCEPT: This is called the "Single Source of Truth" principle.
"""

import os

# ─────────────────────────────────────────────
# ROOT PATH
# ─────────────────────────────────────────────
# os.path.dirname(__file__)  → the folder where THIS file lives  (src/)
# os.path.join(..., "..")    → one level up                       (palm-ai/)
# os.path.abspath(...)       → converts to a clean absolute path

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ─────────────────────────────────────────────
# DATA PATHS
# ─────────────────────────────────────────────
DATA_DIR        = os.path.join(BASE_DIR, "data")
RAW_IMAGES_DIR  = os.path.join(DATA_DIR, "raw", "images")
PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")
ANNOTATIONS_DIR = os.path.join(DATA_DIR, "annotations")
MASKS_DIR       = os.path.join(DATA_DIR, "masks")

# ─────────────────────────────────────────────
# OUTPUT PATHS
# ─────────────────────────────────────────────
OUTPUTS_DIR      = os.path.join(BASE_DIR, "outputs")
MODELS_DIR       = os.path.join(BASE_DIR, "models")

# ── Experiment sub-folders (created by scripts as needed) ─────────────────────
THRESHOLDS_DIR   = os.path.join(PROCESSED_DIR, "thresholds")
MORPHOLOGY_DIR   = os.path.join(PROCESSED_DIR, "morphology")
EDGES_DIR        = os.path.join(PROCESSED_DIR, "edges")

# ─────────────────────────────────────────────
# IMAGE SETTINGS
# ─────────────────────────────────────────────
# We'll standardize all palm images to this size during preprocessing.
# 512x512 is a common choice for segmentation models — large enough for
# fine details, small enough to train without a powerful GPU.
IMAGE_SIZE = (512, 512)   # (width, height)

# ─────────────────────────────────────────────
# SUPPORTED IMAGE FORMATS
# ─────────────────────────────────────────────
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
