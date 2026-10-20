# RoadGuard AI

**Edge AI-Based Road Hazard Detection, Severity Assessment and Location Mapping System**

RoadGuard AI is a computer vision-based road monitoring system that detects potholes in road images and videos, estimates their visual severity, and extracts GPS coordinates when available.

## Project Modules

- **Module 1 — AI Detection:** Detects potholes using a trained YOLO model.
- **Module 2 — Severity Assessment and Location Mapping:** Estimates severity, extracts available image GPS metadata, and processes video frames.
- **Module 3 — Dashboard and Visualization:** Displays detection results, severity levels, map locations, and hazard history.

## Technology Stack

- Python
- Ultralytics YOLO
- PyTorch
- OpenCV
- Pillow
- pytest
- JSON
- Streamlit (planned for the dashboard)

## Project Structure

```text
RoadGuard-AI/
├── assessment/
│   ├── analyzer.py
│   ├── severity.py
│   ├── video_analyzer.py
│   └── save_video_results.py
├── common/
│   └── location.py
├── dashboard/
├── data/
├── detection/
│   ├── detector.py
│   ├── detect_image.py
│   ├── detect_video.py
│   └── check_model.py
├── docs/
├── models/
├── outputs/
├── tests/
├── requirements.txt
└── README.md
```

## Setup

Clone the repository and navigate to the project directory.

```powershell
git clone https://github.com/priya-nka-meena/RoadGuard-AI.git
cd RoadGuard-AI
```

Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
py -m pip install -r requirements.txt
```

Ensure that `models/pothole_best.pt` and the required input files are available locally.

## Module 1 — Pothole Detection

The detection module uses YOLO to identify potholes and return their predicted class, confidence score, and bounding-box coordinates.

### Reusable detector

```python
from detection.detector import PotholeDetector

detector = PotholeDetector()
detections = detector.detect("data/road.jpg")

for detection in detections:
    print(detection)
```

Each detection contains:

- `label`: Predicted class.
- `confidence`: Confidence score between 0 and 1.
- `bbox`: Bounding-box coordinates in `[x1, y1, x2, y2]` format.

The reusable detector uses a default confidence threshold of `0.25`.

### Image detection

Place the input image at `data/road.jpg` and run:

```powershell
py detection/detect_image.py
```

The script detects potholes and saves an annotated image.

### Video detection

Place the input video at `data/road_video.mp4` and run:

```powershell
py detection/detect_video.py
```

The script processes the video frame by frame and saves the annotated video to:

`outputs/road_video_detected.mp4`

It also reports the number of frames processed and total detections across frames.

## Module 2 — Severity Assessment and Location Mapping

### Severity assessment

File: `assessment/severity.py`

The function `estimate_severity(bbox, image_width, image_height)` estimates severity using the bounding-box area relative to the image area.

| Bounding-box area ratio | Severity |
|---|---|
| Up to 2% | LOW |
| Above 2%, up to 10% | MEDIUM |
| Above 10% | HIGH |

This is a visual size-based estimate and does not measure actual pothole depth.

### Image analysis

File: `assessment/analyzer.py`

The function `analyze_image(image_path, model_path=None, confidence=0.25)` combines pothole detection, severity assessment, and GPS extraction.

Example:

```python
from assessment.analyzer import analyze_image

results = analyze_image("data/road.jpg")

for detection in results["detections"]:
    print(detection)
```

The results include image dimensions, available GPS information, and detections enriched with severity and location fields.

### GPS extraction

File: `common/location.py`

The function `extract_gps(image_path)` extracts latitude and longitude from image EXIF GPS metadata when available. If the metadata is missing or unusable, it returns `None`.

Image GPS represents the camera's recorded location and does not guarantee the exact position of an individual pothole.

### Video analysis

File: `assessment/video_analyzer.py`

The function `analyze_video(video_path, model_path=None, confidence=0.25, frame_stride=1)`:

- Processes video frames using OpenCV.
- Detects potholes using the reusable YOLO detector.
- Estimates severity for each detection.
- Records frame indices and timestamps.
- Returns structured results for further processing.

The `frame_stride` parameter controls frame skipping. For example, `frame_stride=5` analyzes approximately every fifth frame.

Video GPS association has not yet been implemented. Coordinates remain `None` until a genuine GPS source synchronized with the video is available.

### JSON export

File: `assessment/save_video_results.py`

Run:

```powershell
py -m assessment.save_video_results
```

The script analyzes `data/road_video.mp4` using a frame stride of five and saves the results to:

`outputs/road_video_results.json`

The JSON includes video information, processed frame count, timestamps, detections, confidence scores, bounding boxes, severity, and available location fields.

Detections across multiple frames may represent the same physical pothole.

## Testing

Run the automated tests:

```powershell
py -m pytest tests/ -v
```

**Latest reported status: 42 tests passed.**
