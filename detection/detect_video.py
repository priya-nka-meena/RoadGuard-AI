from pathlib import Path
import cv2
from ultralytics import YOLO

MODEL_PATH = Path("models/pothole_best.pt")
VIDEO_PATH = Path("data/road_video.mp4")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH}\n"
        "Place a road video at this path before running the detector."
    )

model = YOLO(str(MODEL_PATH))

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0 or width <= 0 or height <= 0:
    cap.release()
    raise RuntimeError("The video has invalid dimensions or frame rate.")

output_path = OUTPUT_DIR / "road_video_detected.mp4"
writer = cv2.VideoWriter(
    str(output_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError(f"Could not create output video: {output_path}")

frame_count = 0
detection_count = 0

try:
    while True:
        success, frame = cap.read()

        if not success:
            break

        results = model.predict(frame, conf=0.25, verbose=False)
        annotated_frame = results[0].plot()

        boxes = results[0].boxes
        if boxes is not None:
            detection_count += len(boxes)

        writer.write(annotated_frame)
        frame_count += 1

        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")

finally:
    cap.release()
    writer.release()

print("\nVideo processing complete!")
print(f"Frames processed: {frame_count}")
print(f"Total detections across frames: {detection_count}")
print(f"Annotated video saved to: {output_path.resolve()}")
