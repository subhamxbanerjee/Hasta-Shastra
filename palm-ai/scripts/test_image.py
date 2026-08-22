"""
test_image.py — Phase 1, Step 4: Load and Inspect a Palm Image

WHAT WE ARE BUILDING:
  A script that loads one palm image, validates it loaded correctly,
  prints what OpenCV tells us about it, and displays it on screen.

WHY WE ARE BUILDING IT:
  Before doing any AI work, we must prove we can:
    1. Read an image file from disk
    2. Understand how it is stored as a NumPy array
    3. Handle errors gracefully if the image is missing or corrupted

WHAT YOU ARE LEARNING:
  - OpenCV's cv2.imread() function
  - How images are NumPy arrays
  - What shape (H, W, C) means
  - The BGR vs RGB difference
  - cv2.imshow() for displaying images

HOW TO RUN:
  1. Place a palm image in:  data/raw/images/test_palm.jpg
  2. Open a terminal in the palm-ai/ folder
  3. Run:  python scripts/test_image.py
"""

import cv2          # OpenCV — the main computer vision library
import numpy as np  # NumPy — OpenCV returns images as NumPy arrays
import sys          # sys.exit() — lets us stop the script with an error message
import os

# ── Step 1: Import our central config ────────────────────────────────────────
# sys.path.insert makes Python look in the palm-ai/ folder first,
# so we can import from `src/` even when running from the scripts/ folder.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import RAW_IMAGES_DIR

# ── Step 2: Define which image to load ───────────────────────────────────────
IMAGE_FILENAME = "test_palm.jpg"
IMAGE_PATH = os.path.join(RAW_IMAGES_DIR, IMAGE_FILENAME)

print("=" * 60)
print("  PalmVerse — Phase 1: Image Loading Test")
print("=" * 60)
print(f"\n📂 Looking for image at:\n   {IMAGE_PATH}\n")

# ── Step 3: Check if the file exists BEFORE trying to load it ────────────────
# This is better than letting OpenCV fail silently.
if not os.path.exists(IMAGE_PATH):
    print("❌ ERROR: Image file not found!")
    print(f"\n👉 Please place a palm photo named '{IMAGE_FILENAME}' inside:")
    print(f"   {RAW_IMAGES_DIR}")
    print("\n   Any JPG or PNG photo of a palm will work for now.")
    sys.exit(1)   # Exit with code 1 = something went wrong

# ── Step 4: Load the image with OpenCV ───────────────────────────────────────
# cv2.imread() reads an image file from disk and returns it as a NumPy array.
#
# IMPORTANT: OpenCV loads images in BGR order (Blue, Green, Red)
# NOT the standard RGB order (Red, Green, Blue) that most other
# tools use. We'll deal with this conversion in a moment.
#
# If loading fails (corrupted file, wrong path), imread returns None.
image = cv2.imread(IMAGE_PATH)

# ── Step 5: Validate that loading worked ─────────────────────────────────────
if image is None:
    print("❌ ERROR: OpenCV failed to load the image.")
    print("   The file exists but could not be read.")
    print("   Possible causes:")
    print("   - The file is corrupted")
    print("   - It is not a valid image format")
    sys.exit(1)

print("✅ Image loaded successfully!\n")

# ── Step 6: Inspect the image array ──────────────────────────────────────────
# An image loaded by OpenCV is a NumPy array with shape: (H, W, C)
#   H = Height in pixels
#   W = Width in pixels
#   C = Channels (3 for BGR colour images, 1 for grayscale)
#
# Example: shape (1200, 900, 3) means:
#   1200 rows (height), 900 columns (width), 3 colour channels (BGR)

height, width, channels = image.shape   # Unpack the three dimensions

print("─" * 40)
print("  IMAGE PROPERTIES")
print("─" * 40)
print(f"  File         : {IMAGE_FILENAME}")
print(f"  Shape        : {image.shape}")
print(f"  Height       : {height} pixels")
print(f"  Width        : {width} pixels")
print(f"  Channels     : {channels}  (B=Blue, G=Green, R=Red)")
print(f"  Data type    : {image.dtype}  (uint8 = values 0–255)")
print(f"  Total pixels : {height * width:,}")
print(f"  Array size   : {image.nbytes:,} bytes in memory")

# ── Step 7: Peek at individual pixel values ───────────────────────────────────
# Let's look at the pixel in the top-left corner (row 0, column 0)
# image[row, col] returns an array of [B, G, R] values for that pixel
top_left_pixel = image[0, 0]   # row=0, col=0
center_pixel   = image[height // 2, width // 2]   # centre of the image

print("\n─" * 20)
print("  PIXEL SAMPLES (BGR order)")
print("─" * 20)
print(f"  Top-left pixel  [0, 0]    : BGR {top_left_pixel}")
print(f"  Centre pixel               : BGR {center_pixel}")

# ── Step 8: Convert BGR → RGB for display explanation ────────────────────────
# OpenCV stores channels as B-G-R but human eyes think R-G-B.
# cv2.cvtColor converts between colour spaces.
# We are NOT saving this — just converting to show you the difference.
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
center_rgb = image_rgb[height // 2, width // 2]
print(f"  Centre pixel (RGB)         : RGB {center_rgb}")
print(f"\n  Note: the values are the same array, just reordered.")
print(f"        BGR channel order: Blue={center_pixel[0]}, Green={center_pixel[1]}, Red={center_pixel[2]}")
print(f"        RGB channel order: Red={center_rgb[0]},  Green={center_rgb[1]}, Blue={center_rgb[2]}")

# ── Step 9: Display the image ─────────────────────────────────────────────────
# cv2.imshow(window_name, image) — opens a window showing the image.
# cv2.waitKey(0) — pauses the script and waits for any key press.
# cv2.destroyAllWindows() — closes all OpenCV windows cleanly.
#
# NOTE: OpenCV displays BGR images correctly by default —
# it knows its own format, so do NOT convert to RGB before imshow().

print("\n─" * 40)
print("  DISPLAY")
print("─" * 40)
print("  Opening image in a new window...")
print("  👉 Press any key while the window is selected to close it.")

cv2.imshow("PalmVerse — test_palm.jpg", image)
cv2.waitKey(0)            # 0 = wait indefinitely until a key is pressed
cv2.destroyAllWindows()   # always clean up windows after use

print("\n✅ Done! Window closed.")
print("\n📚 What you just learned:")
print("   1. cv2.imread() loads an image as a NumPy array")
print("   2. shape = (Height, Width, Channels)")
print("   3. dtype = uint8 means pixel values are integers 0–255")
print("   4. OpenCV uses BGR — not RGB — channel order")
print("   5. Always check for None after imread()")
print("=" * 60)
