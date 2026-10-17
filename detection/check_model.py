from ultralytics import YOLO

# Load the downloaded pothole-detection model
model = YOLO("models/best.pt")

print("Model loaded successfully!")
print("Detection classes:", model.names)
