
from unittest.mock import patch

from PIL import Image

from assessment.analyzer import analyze_image


def test_analyzer_combines_detection_severity_and_gps(tmp_path):
    image_path = tmp_path / "road.jpg"
    Image.new("RGB", (100, 100)).save(image_path)

    fake_detection = {
        "label": "pothole",
        "confidence": 0.92,
        "bbox": [0, 0, 10, 10],
    }

    fake_gps = {
        "latitude": 28.6,
        "longitude": 77.2,
    }

    with (
        patch("assessment.analyzer.PotholeDetector") as mock_detector,
        patch("assessment.analyzer.extract_gps", return_value=fake_gps),
    ):
        mock_detector.return_value.detect.return_value = [fake_detection]

        result = analyze_image(image_path)

    assert result["image_width"] == 100
    assert result["image_height"] == 100
    assert result["location"] == fake_gps
    assert len(result["detections"]) == 1

    detection = result["detections"][0]
    assert detection["label"] == "pothole"
    assert detection["confidence"] == 0.92
    assert detection["severity"] == "LOW"
    assert detection["latitude"] == 28.6
    assert detection["longitude"] == 77.2


def test_analyzer_handles_missing_gps(tmp_path):
    image_path = tmp_path / "road.jpg"
    Image.new("RGB", (100, 100)).save(image_path)

    fake_detection = {
        "label": "pothole",
        "confidence": 0.85,
        "bbox": [0, 0, 50, 50],
    }

    with (
        patch("assessment.analyzer.PotholeDetector") as mock_detector,
        patch("assessment.analyzer.extract_gps", return_value=None),
    ):
        mock_detector.return_value.detect.return_value = [fake_detection]

        result = analyze_image(image_path)

    detection = result["detections"][0]
    assert result["location"] is None
    assert detection["severity"] == "HIGH"
    assert detection["latitude"] is None
    assert detection["longitude"] is None
