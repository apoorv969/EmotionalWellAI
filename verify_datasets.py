import os

datasets = {
    "FER2013": "datasets/FER2013",
    "RAF_DB": "datasets/RAF_DB",
    "RAVDESS": "datasets/RAVDESS",
    "Indic Text": "datasets/multilingual_text"
}

print("🔍 Verifying all datasets...\n")

for name, path in datasets.items():
    if os.path.exists(path):
        print(f"✅ {name} dataset found")
    else:
        print(f"❌ {name} dataset missing")

print("\n✔ Dataset verification completed")