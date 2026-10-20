
"""Combine pothole detection, severity estimation, and image GPS."""

from pathlib import Path

from PIL import Image

from assessment.severity import estimate_severity
from common.location import extract_gps
from detection.detector import PotholeDetector


def analyze_image(image_path, model_path=None, confidence=0.25):
    """Analyze an image and return enriched pothole detections."""
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as image:
        image_width, image_height = image.size

    gps = extract_gps(image_path)

    if model_path is None:
        detector = PotholeDetector(confidence=confidence)
    else:
        detector = PotholeDetector(
            model_path=model_path,
            confidence=confidence,
        )

    detections = detector.detect(str(image_path))
    results = []

    for detection in detections:
        bbox = detection["bbox"]

        severity = estimate_severity(
            bbox,
            image_width,
            image_height,
        )

        results.append({
            **detection,
            "severity": severity,
            "latitude": gps["latitude"] if gps else None,
            "longitude": gps["longitude"] if gps else None,
        })

    return {
        "image": str(image_path),
        "image_width": image_width,
        "image_height": image_height,
        "location": gps,
        "detections": results,
    }
