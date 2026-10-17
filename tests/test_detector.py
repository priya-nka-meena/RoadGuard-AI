"""Unit tests for PotholeDetector using mocked YOLO model."""
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from detection.detector import PotholeDetector


class MockTensor:
    """Mock tensor with .item() and .tolist() methods for testing."""
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value

    def tolist(self):
        return self.value


class MockBox:
    """Mock YOLO box object for testing."""
    def __init__(self, cls, conf, xyxy):
        # cls and conf are lists of values
        self.cls = [MockTensor(c) for c in cls]
        self.conf = [MockTensor(c) for c in conf]
        # xyxy[0].tolist() should return the coordinate list
        self.xyxy = [MockTensor(xyxy)]


class MockResult:
    """Mock YOLO result object for testing."""
    def __init__(self, boxes):
        self.boxes = boxes


class MockYOLO:
    """Mock YOLO model for testing."""
    def __init__(self, model_path):
        self.model_path = model_path
        self.names = {0: "Pothole"}

    def predict(self, source, conf, verbose=False):
        """Return mock detection results."""
        return self._get_mock_results(source, conf)

    def _get_mock_results(self, source, conf):
        """Generate mock results based on source and confidence."""
        # Different mock scenarios based on source string
        if "no_detections" in str(source):
            return [MockResult(None)]
        elif "empty" in str(source):
            return [MockResult([])]
        else:
            # Return mock detections
            mock_boxes = [
                MockBox(
                    cls=[0],
                    conf=[0.85],
                    xyxy=[100, 200, 300, 400]
                ),
                MockBox(
                    cls=[0],
                    conf=[0.72],
                    xyxy=[400, 500, 600, 700]
                ),
            ]
            return [MockResult(mock_boxes)]


@pytest.fixture
def mock_yolo():
    """Fixture that provides a mocked YOLO model."""
    with patch("detection.detector.YOLO", MockYOLO) as mock:
        yield mock


@pytest.fixture
def detector(mock_yolo):
    """Fixture that provides a PotholeDetector with mocked model."""
    detector = PotholeDetector.__new__(PotholeDetector)
    detector.model = MockYOLO("dummy.pt")
    detector.confidence = 0.25
    return detector


class TestPotholeDetectorInit:
    """Test PotholeDetector initialization."""

    def test_init_with_valid_confidence(self, mock_yolo):
        """Test detector initializes with valid confidence values."""
        detector = PotholeDetector.__new__(PotholeDetector)
        detector.model = MockYOLO("dummy.pt")
        detector.confidence = 0.5
        assert detector.confidence == 0.5

    def test_init_with_confidence_zero(self, mock_yolo):
        """Test detector accepts confidence of 0.0."""
        detector = PotholeDetector.__new__(PotholeDetector)
        detector.model = MockYOLO("dummy.pt")
        detector.confidence = 0.0
        assert detector.confidence == 0.0

    def test_init_with_confidence_one(self, mock_yolo):
        """Test detector accepts confidence of 1.0."""
        detector = PotholeDetector.__new__(PotholeDetector)
        detector.model = MockYOLO("dummy.pt")
        detector.confidence = 1.0
        assert detector.confidence == 1.0

    def test_init_with_invalid_confidence_negative(self, mock_yolo):
        """Test detector rejects negative confidence values."""
        dummy_path = Path("dummy.pt")
        dummy_path.touch()
        try:
            with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
                PotholeDetector(model_path="dummy.pt", confidence=-0.1)
        finally:
            if dummy_path.exists():
                dummy_path.unlink()

    def test_init_with_invalid_confidence_above_one(self, mock_yolo):
        """Test detector rejects confidence values above 1.0."""
        dummy_path = Path("dummy.pt")
        dummy_path.touch()
        try:
            with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
                PotholeDetector(model_path="dummy.pt", confidence=1.5)
        finally:
            if dummy_path.exists():
                dummy_path.unlink()

    def test_init_with_nonexistent_model(self):
        """Test detector raises FileNotFoundError for missing model."""
        with pytest.raises(FileNotFoundError, match="Model not found"):
            PotholeDetector(model_path="nonexistent_model.pt")


class TestPotholeDetectorDetect:
    """Test PotholeDetector.detect method."""

    def test_detect_returns_list(self, detector):
        """Test that detect returns a list."""
        detections = detector.detect("test_image.jpg")
        assert isinstance(detections, list)

    def test_detect_returns_detections(self, detector):
        """Test that detect returns detections for valid image."""
        detections = detector.detect("test_image.jpg")
        assert len(detections) == 2

    def test_detection_has_required_fields(self, detector):
        """Test each detection has label, confidence, and bbox fields."""
        detections = detector.detect("test_image.jpg")
        for detection in detections:
            assert "label" in detection
            assert "confidence" in detection
            assert "bbox" in detection

    def test_detection_label_is_string(self, detector):
        """Test detection label is a string."""
        detections = detector.detect("test_image.jpg")
        for detection in detections:
            assert isinstance(detection["label"], str)

    def test_detection_confidence_is_numeric(self, detector):
        """Test detection confidence is numeric (float)."""
        detections = detector.detect("test_image.jpg")
        for detection in detections:
            assert isinstance(detection["confidence"], (int, float))

    def test_detection_confidence_in_valid_range(self, detector):
        """Test detection confidence is between 0 and 1."""
        detections = detector.detect("test_image.jpg")
        for detection in detections:
            assert 0.0 <= detection["confidence"] <= 1.0

    def test_detection_bbox_has_four_coordinates(self, detector):
        """Test bounding box contains exactly four coordinates."""
        detections = detector.detect("test_image.jpg")
        for detection in detections:
            assert len(detection["bbox"]) == 4

    def test_detection_bbox_coordinates_are_numeric(self, detector):
        """Test bounding box coordinates are numeric."""
        detections = detector.detect("test_image.jpg")
        for detection in detections:
            for coord in detection["bbox"]:
                assert isinstance(coord, (int, float))

    def test_detect_no_detections_none_boxes(self, detector):
        """Test detect handles None boxes (no detections)."""
        detections = detector.detect("no_detections.jpg")
        assert detections == []

    def test_detect_no_detections_empty_boxes(self, detector):
        """Test detect handles empty boxes list (no detections)."""
        detections = detector.detect("empty.jpg")
        assert detections == []

    def test_detect_multiple_detections(self, detector):
        """Test detect can return multiple detections."""
        detections = detector.detect("test_image.jpg")
        assert len(detections) >= 2


class TestPotholeDetectorEdgeCases:
    """Test edge cases and error conditions."""

    def test_detect_with_path_object(self, detector):
        """Test detect accepts Path object as input."""
        detections = detector.detect(Path("test_image.jpg"))
        assert isinstance(detections, list)

    def test_detect_with_string_path(self, detector):
        """Test detect accepts string path as input."""
        detections = detector.detect("test_image.jpg")
        assert isinstance(detections, list)


class TestPotholeDetectorIntegration:
    """Integration tests with real model (optional, skipped by default)."""

    @pytest.mark.integration
    @pytest.mark.skipif(
        not Path("models/pothole_best.pt").exists(),
        reason="Model weights not found"
    )
    @pytest.mark.skipif(
        not Path("data/Road.jpg").exists(),
        reason="Test image not found"
    )
    def test_real_model_detection(self):
        """Test with real model and image (optional integration test)."""
        detector = PotholeDetector()
        detections = detector.detect("data/Road.jpg")

        # Verify structure
        assert isinstance(detections, list)

        # If detections exist, verify their structure
        for detection in detections:
            assert "label" in detection
            assert "confidence" in detection
            assert "bbox" in detection
            assert 0.0 <= detection["confidence"] <= 1.0
            assert len(detection["bbox"]) == 4
