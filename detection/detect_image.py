from pathlib import Path
from ultralytics import YOLO

# Project paths
MODEL_PATH = Path("models/pothole_best.pt")
IMAGE_PATH = Path("data/road.jpg")

# Load the pretrained road-damage detector
model = YOLO(str(MODEL_PATH))

# Run object detection on the road image
results = model.predict(
    source=str(IMAGE_PATH),
    conf=0.10,
    save=True,
    project="outputs",
    name="image_detection",
    exist_ok=True
)

# Print detection details
for result in results:
    print("\nDetection results:")
    if result.boxes is None or len(result.boxes) == 0:
        print("No road hazards detected.")
        continue

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        label = model.names[class_id]

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        print(
            f"Class: {label} | "
            f"Confidence: {confidence:.2%} | "
            f"Box: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})"
        )

print("\nCheck outputs/image_detection for the annotated image.")

