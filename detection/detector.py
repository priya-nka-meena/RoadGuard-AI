from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "pothole_best.pt"


class PotholeDetector:
    """Reusable YOLO pothole detector for images and video frames."""

    def __init__(self, model_path=MODEL_PATH, confidence=0.25):
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1.")

        self.model = YOLO(str(model_path))
        self.confidence = confidence

    def detect(self, image):
        """Return detected labels, confidence scores, and bounding boxes."""
        results = self.model.predict(
            source=image,
            conf=self.confidence,
            verbose=False,
        )

        detections = []

        # Handle case where no detections are found
        if results[0].boxes is None:
            return detections

        for box in results[0].boxes:
            class_id = int(box.cls[0].item())

            detections.append({
                "label": self.model.names[class_id],
                "confidence": float(box.conf[0].item()),
                "bbox": [int(v) for v in box.xyxy[0].tolist()],
            })

        return detections
