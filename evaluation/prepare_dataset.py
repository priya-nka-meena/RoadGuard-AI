
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_DIR / "data" / "potholes_raw" / "annotated-images"
SPLITS_FILE = PROJECT_DIR / "data" / "potholes_raw" / "splits.json"

IMAGE_DIR = PROJECT_DIR / "evaluation" / "images"
LABEL_DIR = PROJECT_DIR / "evaluation" / "labels"

# Create output folders
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True)

# Read the original train/test split
with open(SPLITS_FILE, "r", encoding="utf-8") as file:
    splits = json.load(file)

test_annotations = splits["test"]

converted = 0
missing = 0
total_boxes = 0

for xml_name in test_annotations:
    xml_path = SOURCE_DIR / xml_name

    # Read XML annotation
    root = ET.parse(xml_path).getroot()

    image_name = root.findtext("filename")
    size = root.find("size")

    if not image_name or size is None:
        print(f"Skipping invalid annotation: {xml_name}")
        continue

    image_path = SOURCE_DIR / image_name

    if not image_path.exists():
        print(f"Missing image: {image_name}")
        missing += 1
        continue

    image_width = int(size.findtext("width"))
    image_height = int(size.findtext("height"))

    if image_width <= 0 or image_height <= 0:
        print(f"Invalid dimensions: {image_name}")
        continue

    yolo_lines = []

    # Convert every pothole bounding box
    for obj in root.findall("object"):
        class_name = obj.findtext("name")

        if class_name != "pothole":
            continue

        box = obj.find("bndbox")
        if box is None:
            continue

        xmin = float(box.findtext("xmin"))
        ymin = float(box.findtext("ymin"))
        xmax = float(box.findtext("xmax"))
        ymax = float(box.findtext("ymax"))

        if xmax <= xmin or ymax <= ymin:
            continue

        # Convert to normalized YOLO coordinates
        x_center = ((xmin + xmax) / 2) / image_width
        y_center = ((ymin + ymax) / 2) / image_height
        box_width = (xmax - xmin) / image_width
        box_height = (ymax - ymin) / image_height

        yolo_lines.append(
            f"0 {x_center:.6f} {y_center:.6f} "
            f"{box_width:.6f} {box_height:.6f}"
        )

    # Copy original image
    shutil.copy2(image_path, IMAGE_DIR / image_name)

    # Save YOLO annotation
    label_name = Path(image_name).stem + ".txt"
    label_path = LABEL_DIR / label_name
    label_path.write_text(
        "\n".join(yolo_lines),
        encoding="utf-8"
    )

    converted += 1
    total_boxes += len(yolo_lines)

print("\n========== DATASET PREPARATION ==========")
print(f"Test annotations in split: {len(test_annotations)}")
print(f"Images and labels converted: {converted}")
print(f"Missing images: {missing}")
print(f"Total pothole bounding boxes: {total_boxes}")
print(f"Images saved in: {IMAGE_DIR}")
print(f"Labels saved in: {LABEL_DIR}")
print("=========================================")
