
import cv2
from detection.detector import PotholeDetector


def main():
    # Initialize the YOLO pothole detector
    detector = PotholeDetector(confidence=0.50)

    # Open the default laptop webcam
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Live pothole detection started.")
    print("Press Q to quit.")

    try:
        while True:
            # Read one frame from the webcam
            success, frame = camera.read()

            if not success:
                print("Error: Could not read webcam frame.")
                break

            # Detect potholes in the current frame
            detections = detector.detect(frame)

            # Draw bounding boxes and confidence scores
            for detection in detections:
                x1, y1, x2, y2 = detection["bbox"]
                label = detection["label"]
                confidence = detection["confidence"]

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                text = f"{label}: {confidence:.2f}"

                cv2.putText(
                    frame,
                    text,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            # Display the live camera feed
            cv2.imshow("RoadGuard AI - Live Pothole Detection", frame)

            # Press Q to exit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Live detection stopped.")


if __name__ == "__main__":
    main()
