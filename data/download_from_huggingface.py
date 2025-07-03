# download_from_huggingface.py

import os
import csv
from datasets import load_dataset
from PIL import Image

# Save the first 40 images: 20 cats, 20 dogs
SAVE_DIR = "data/processed"
CSV_PATH = "data/sample.csv"
os.makedirs(SAVE_DIR, exist_ok=True)

print("📦 Loading Hugging Face cats_vs_dogs dataset...")
dataset = load_dataset("microsoft/cats_vs_dogs", split="train")

# Save first 20 cat images and first 20 dog images
cat_count, dog_count = 0, 0
records = []

for item in dataset:
    label = item["labels"]  # 0 = cat, 1 = dog
    img = item["image"]

    if label == 0 and cat_count < 20:
        filename = f"cat_{cat_count+1:03}.jpg"
        cat_count += 1
    elif label == 1 and dog_count < 20:
        filename = f"dog_{dog_count+1:03}.jpg"
        dog_count += 1
    else:
        continue

    filepath = os.path.join(SAVE_DIR, filename)
    img.save(filepath)
    records.append({"file_name": filename, "label": "cat" if label == 0 else "dog"})

    if cat_count == 20 and dog_count == 20:
        break

# Write label CSV file
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["file_name", "label"])
    writer.writeheader()
    writer.writerows(records)

print(f"✅ Download complete! {len(records)} images saved to: {SAVE_DIR}")
print(f"📄 Label CSV file created at: {CSV_PATH}")
