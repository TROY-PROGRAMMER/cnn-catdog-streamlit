# cnn_pet_classifier/utils/helper.py

import csv

def load_label_map_from_csv(csv_path):
    label_map = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["file_name"]
            label_str = row["label"].strip().lower()
            label = 0 if label_str == "cat" else 1
            label_map[filename] = label
    return label_map
