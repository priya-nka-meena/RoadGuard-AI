# RoadGuard AI

**Edge AI-Based Road Hazard Detection, Severity Assessment and Location Mapping System**

RoadGuard AI is a software-based road monitoring project that uses computer vision to detect potholes in road images and videos. The project aims to support visual severity assessment and location mapping through a modular pipeline.

## Project Overview

The project is divided into three modules:

- **Module 1 — AI Detection:** Detect potholes using a pretrained YOLO model.
- **Module 2 — Severity Assessment and Location Mapping:** Estimate hazard severity and associate GPS coordinates when available.
- **Module 3 — Dashboard and Visualization:** Display detection results, severity levels, map locations, and hazard history.

## Technology Stack

- Python 3.10
- Ultralytics YOLO
- PyTorch
- OpenCV
- pytest
- Streamlit (planned for the dashboard)

## Project Structure

```text
RoadGuard-AI/
├── assessment/       # Severity assessment and location logic
├── common/           # Shared utilities and data structures
├── dashboard/        # Dashboard and visualization
├── data/             # Local input images and videos
├── detection/        # YOLO-based pothole detection
│   ├── detector.py
│   ├── detect_image.py
│   ├── detect_video.py
│   └── check_model.py
├── docs/             # Project documentation
├── models/           # Local trained model weights
├── outputs/          # Generated results
├── tests/            # Automated tests
├── requirements.txt
└── README.md
```

## Setup

Clone the repository and open the project directory:

```powershell
git clone https://github.com/priya-nka-meena/RoadGuard-AI.git
cd RoadGuard-AI
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Ensure the model file `models/pothole_best.pt` and the required input files are available locally. Model weights and sample data may not be included in the Git repository.

## Module 1: Pothole Detection

The detection module uses YOLO to identify potholes and return their predicted class, confidence score, and bounding-box coordinates.

### Reusable detector

```python
from detection.detector import PotholeDetector

detector = PotholeDetector()
detections = detector.detect("data/Road.jpg")

for detection in detections:
    print(detection)
```

Each detection contains:

- `label`: Predicted hazard class.
- `confidence`: Model confidence between 0 and 1.
- `bbox`: Bounding-box pixel coordinates in `[x1, y1, x2, y2]` format.

The reusable detector uses a default confidence threshold of `0.25`.

### Image detection

Place the input image at `data/road.jpg` and run:

```powershell
python detection/detect_image.py
```

The script prints the detected class, confidence, and bounding-box coordinates and saves an annotated image. The standalone image script currently uses a confidence threshold of `0.10`.

### Video detection

Place the input video at `data/road_video.mp4` and run:

```powershell
python detection/detect_video.py
```

The script processes the video frame by frame and generates:

`outputs/road_video_detected.mp4`

It also reports the number of processed frames and the total detections across those frames. This count does not represent unique potholes.

## Testing

Run the automated tests:

```powershell
python -m pytest tests/ -v
```

**Current status:** 20 tests passed, including unit tests and a real-model integration test.

## Initial Detection Result

On the sample road image, the model produced:

- **Detected class:** Pothole
- **Confidence score:** 84.45%
- **Bounding box:** `(437, 521, 1184, 939)`
- **Image dimensions