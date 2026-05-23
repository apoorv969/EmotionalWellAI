from kaggle.api.kaggle_api_extended import KaggleApi
import os

dataset_dir = "C:/Users/alokm/datasets/FER2013"
os.makedirs(dataset_dir, exist_ok=True)

api = KaggleApi()
api.authenticate()  # reads kaggle.json automatically

# Correct dataset slug
api.dataset_download_files("msambare/fer2013", path=dataset_dir, unzip=True, quiet=False)

print("FER2013 downloaded to:", dataset_dir)