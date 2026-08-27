import requests
import json
import os
from pathlib import Path
import numpy as np
import cv2

BASE_URL = "http://localhost:8000/api/analyze"
DATA_DIR = Path("C:/Users/subha/OneDrive/Desktop/Hasta-Shastra/palm-ai/data/raw/images")

def test_file(filepath: Path, expected_status: str, desc: str):
    print(f"\n--- Testing: {desc} ---")
    print(f"File: {filepath.name}")
    try:
        with open(filepath, 'rb') as f:
            files = {'file': (filepath.name, f, 'image/jpeg')}
            # Change mime type for non-images
            if filepath.suffix == '.md':
                files = {'file': (filepath.name, f, 'text/markdown')}
                
            response = requests.post(BASE_URL, files=files)
            
        print(f"HTTP Status Code: {response.status_code}")
        
        try:
            data = response.json()
            print(f"Response Body: {json.dumps(data, indent=2)}")
            
            if data.get("status") == expected_status:
                print(f"[PASS] Expected status '{expected_status}' matched.")
            else:
                print(f"[FAIL] Expected status '{expected_status}', got '{data.get('status')}'.")
                
        except json.JSONDecodeError:
            print("[FAIL] Response is not valid JSON.")
            print(response.text)
            
    except Exception as e:
        print(f"[FAIL] Exception occurred: {e}")

if __name__ == "__main__":
    # Create no-hand image
    no_hand_path = DATA_DIR / "no_hand.jpg"
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.putText(img, "NOT A HAND", (100, 256), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite(str(no_hand_path), img)
    
    # Create corrupt image
    corrupt_path = DATA_DIR / "corrupt.jpg"
    with open(corrupt_path, "wb") as f:
        f.write(b"This is not a real JPEG image.")

    # Run tests
    test_file(DATA_DIR / "IMG_0005.jpg", "success", "Valid Palm 1")
    test_file(DATA_DIR / "IMG_0016.jpg", "success", "Valid Palm 2")
    test_file(DATA_DIR / "IMG_0020.jpg", "success", "Valid Palm 3")
    
    test_file(no_hand_path, "failed", "No Hand Image")
    
    readme_path = Path("C:/Users/subha/OneDrive/Desktop/Hasta-Shastra/README.md")
    test_file(readme_path, "failed", "Invalid File Type")
    
    test_file(corrupt_path, "failed", "Corrupted Image")
