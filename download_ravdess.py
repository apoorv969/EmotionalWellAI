import os
import zipfile
import urllib.request

URL = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
ZIP_PATH = "ravdess.zip"
DATASET_DIR = "datasets/RAVDESS"

os.makedirs(DATASET_DIR, exist_ok=True)

print("📥 Downloading RAVDESS dataset...")
urllib.request.urlretrieve(URL, ZIP_PATH)

with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(DATASET_DIR)

os.remove(ZIP_PATH)

print("✅ RAVDESS dataset downloaded and extracted")