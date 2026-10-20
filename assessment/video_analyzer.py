
"""Analyze video frames for potholes, severity, and available GPS data."""

from pathlib import Path
import cv2

from assessment.severity import estimate_severity
from detection.detector import PotholeDetector


def analyze_video(video_path, model_path=None, confidence=0.25, frame_stride=1):
    """Analyze video frames and return structured detection results."""
    video_path = Path(video_path)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1.")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise RuntimeError("The video has an invalid frame rate.")

    if model_path is None:
        detector = PotholeDetector(confidence=confidence)
    else:
        detector = PotholeDetector(
            model_path=model_path,
            confidence=confidence,
        )

    frames = []
    frame_index = 0

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            current_index = frame_index
            frame_index += 1

            if current_index % frame_stride != 0:
                continue

            height, width = frame.shape[:2]
            detections = detector.detect(frame)

            enriched_detections = []

            for detection in detections:
                enriched_detections.append({
                    **detection,
                    "severity": estimate_severity(
                        detection["bbox"], width, height
                    ),
                    "latitude": None,
                    "longitude": None,
                })

            frames.append({
                "frame_index": current_index,
                "timestamp_seconds": round(current_index / fps, 3),
                "detections": enriched_detections,
            })

    finally:
        cap.release()

    return {
        "video": str(video_path),
        "fps": fps,
        "frames_processed": len(frames),
        "total_detections": sum(
            len(frame["detections"]) for frame in frames
        ),
        "frames": frames,
    }
