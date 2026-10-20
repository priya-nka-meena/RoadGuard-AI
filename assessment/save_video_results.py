
"""Run video analysis and save the results as JSON."""

import json
from pathlib import Path

from assessment.video_analyzer import analyze_video


VIDEO_PATH = Path("data/road_video.mp4")
OUTPUT_PATH = Path("outputs/road_video_results.json")


def main():
    results = analyze_video(VIDEO_PATH, frame_stride=5)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print("Video analysis complete!")
    print(f"Frames processed: {results['frames_processed']}")
    print(f"Total detections: {results['total_detections']}")
    print(f"JSON saved to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
