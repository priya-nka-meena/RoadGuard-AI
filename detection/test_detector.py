from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from detection.detector import PotholeDetector

image_path = PROJECT_ROOT / "data" / "Road.jpg"

detector = PotholeDetector()
detections = detector.detect(str(image_path))

print(f"Image: {image_path}")
print(f"Detections found: {len(detections)}")

for detection in detections:
    print(detection)
