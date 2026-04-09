"""
download_dataset.py
===================
Auto-download the Crop Recommendation CSV from a public mirror.

If this fails, download manually from:
  https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

Run:
    py download_dataset.py
"""

import sys
import os
import urllib.request

FILENAME = "Crop_recommendation.csv"

# Public raw GitHub mirror of the dataset
URLS = [
    # Direct raw file from a known public fork
    "https://raw.githubusercontent.com/dsrscientist/dataset1/master/Crop_recommendation.csv",
    # Backup: another common mirror
    "https://raw.githubusercontent.com/Gladiator07/Harvestify/master/Data-processed/crop_recommendation.csv",
]

def download(url, dest):
    print(f"->   Trying: {url}")
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp, open(dest, "wb") as f:
            size = 0
            while chunk := resp.read(8192):
                f.write(chunk)
                size += len(chunk)
        print(f"OK  Downloaded {dest} ({size/1024:.1f} KB)")
        return True
    except Exception as e:
        print(f"XX  Failed: {e}")
        return False

if os.path.exists(FILENAME):
    print(f"OK  {FILENAME} already exists. Nothing to do.")
    sys.exit(0)

success = False
for url in URLS:
    if download(url, FILENAME):
        success = True
        break

if not success:
    print("\n!!   Automatic download failed.")
    print("    Please download manually from:")
    print("    https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset")
    print(f"    and place '{FILENAME}' in this folder.\n")
    sys.exit(1)

# Quick sanity check
with open(FILENAME) as f:
    lines = f.readlines()
print(f"##  Rows: {len(lines)-1}  |  Header: {lines[0].strip()}")
