import requests
import json
import sys

url = "http://localhost:8000/api/analyze"
file_path = "data/raw/images/IMG_0022.jpg"

try:
    with open(file_path, "rb") as f:
        files = {"file": ("IMG_0022.jpg", f, "image/jpeg")}
        print(f"Sending POST request to {url} with {file_path}...")
        response = requests.post(url, files=files)
        
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
