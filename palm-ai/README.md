# PalmVerse — palm-ai

AI/computer vision backend for the Hasta-Shastra palm reading application.

## Phase 1 — Python + Image Fundamentals

### How to run the image test

1. Place a palm photo (any JPG) inside:
   ```
   data/raw/images/test_palm.jpg
   ```

2. Activate your virtual environment:
   ```
   .venv\Scripts\activate
   ```

3. From the `palm-ai/` folder, run:
   ```
   python scripts/test_image.py
   ```

## Project Structure

```
palm-ai/
├── data/
│   ├── raw/images/       ← original palm images (read-only)
│   ├── processed/        ← preprocessed images
│   ├── annotations/      ← label files
│   └── masks/            ← segmentation mask PNGs
├── notebooks/            ← Jupyter exploration notebooks
├── scripts/              ← runnable Python scripts
│   ├── test_image.py
│   └── preprocess.py
├── src/                  ← reusable modules
│   ├── __init__.py
│   └── config.py
├── models/               ← saved model checkpoints
├── outputs/              ← generated debug images
├── requirements.txt
└── PROJECT_PROGRESS.md
```
