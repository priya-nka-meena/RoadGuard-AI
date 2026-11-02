"""Live camera helpers for the dashboard.

Uses Member 1's PotholeDetector (unchanged) on frames from a webcam, a network
camera URL, or a video file. A webcam has no GPS, so no location is attached.
"""

import os

import cv2
import streamlit as st

from assessment.severity import estimate_severity
from detection.detector import PotholeDetector

SEV_RANK = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}


@st.cache_resource(show_spinner="Loading detection model...")
def get_detector(confidence):
    return PotholeDetector(confidence=confidence)


def parse_source(text):
    """'0' -> webcam 0. Anything else is treated as a stream URL or file path."""
    text = (text or "").strip()
    return int(text) if text.isdigit() else text


def open_camera(source):
    """Open the camera. Returns a VideoCapture, or None if it cannot be opened."""
    if isinstance(source, int) and os.name == "nt":
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)  # more reliable on Windows
        if cap.isOpened():
            return cap
        cap.release()
    cap = cv2.VideoCapture(source)
    if cap.isOpened():
        return cap
    cap.release()
    return None


def add_severity(detections, width, height):
    """Add Member 2's visual severity to each detection. No GPS for live frames."""
    out = []
    for d in sorted(detections, key=lambda x: x["confidence"], reverse=True):
        try:
            severity = estimate_severity(d["bbox"], width, height)
        except ValueError:
            continue  # skip a malformed box instead of crashing the live view
        out.append({**d, "severity": severity, "latitude": None, "longitude": None,
                    "id": len(out) + 1})
    return out


def frame_score(detections):
    """Rank a frame: worst severity first, then highest confidence."""
    if not detections:
        return (0, 0.0)
    return (max(SEV_RANK[d["severity"]] for d in detections),
            max(d["confidence"] for d in detections))