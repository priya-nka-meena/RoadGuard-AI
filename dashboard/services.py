"""Glue code between the dashboard and the detection/assessment modules.

Nothing here invents data: every detection comes from analyze_image() or
analyze_video(). GPS is only used when it really exists.
"""

import io
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from assessment.analyzer import analyze_image
from assessment.video_analyzer import analyze_video

NO_GPS = "Location unavailable"
SEVERITIES = ["LOW", "MEDIUM", "HIGH"]
SEVERITY_RGB = {"LOW": (46, 160, 67), "MEDIUM": (240, 150, 20), "HIGH": (220, 38, 38)}


def _write_temp(file_bytes, filename):
    """Save upload to a temp file, keeping the extension (needed for EXIF GPS)."""
    suffix = Path(filename).suffix or ".bin"
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(file_bytes)
    return path


def _safe_remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


# ---------------------------------------------------------------- drawing
def draw_boxes(image_bgr, detections):
    """Draw severity-coloured boxes on a BGR image. Returns an RGB array."""
    img = image_bgr.copy()
    h, w = img.shape[:2]
    thickness = max(2, round(max(h, w) / 400))
    scale = max(0.5, max(h, w) / 1200)
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        r, g, b = SEVERITY_RGB.get(d["severity"], (37, 99, 235))
        color = (b, g, r)  # OpenCV uses BGR
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        text = f'{d["label"]} {d["confidence"] * 100:.0f}% {d["severity"]}'
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        top = max(0, y1 - th - base - 4)
        cv2.rectangle(img, (x1, top), (x1 + tw + 6, top + th + base + 4), color, -1)
        cv2.putText(img, text, (x1 + 3, top + th + 1), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def decode_image(file_bytes):
    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not read this file as an image.")
    return img


# ----------------------------------------------------------------- images
def run_image_analysis(file_bytes, filename):
    path = _write_temp(file_bytes, filename)
    try:
        result = analyze_image(path)
    finally:
        _safe_remove(path)
    result["image"] = filename
    annotated = draw_boxes(decode_image(file_bytes), result["detections"])
    return result, annotated


# ----------------------------------------------------------------- videos
def attach_gps_from_csv(result, csv_bytes, tolerance_seconds=2.0):
    """Match a real GPS log (timestamp_seconds, latitude, longitude) to video
    frames by nearest timestamp. Returns the number of detections that got GPS."""
    df = pd.read_csv(io.BytesIO(csv_bytes))
    df.columns = [c.strip().lower() for c in df.columns]
    needed = {"timestamp_seconds", "latitude", "longitude"}
    if not needed.issubset(df.columns):
        raise ValueError("GPS CSV needs columns: timestamp_seconds, latitude, longitude")
    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=list(needed))
    df = df[df["latitude"].between(-90, 90) & df["longitude"].between(-180, 180)]
    if df.empty:
        raise ValueError("GPS CSV has no valid rows.")
    times = df["timestamp_seconds"].to_numpy()
    matched = 0
    for frame in result["frames"]:
        i = int(np.abs(times - frame["timestamp_seconds"]).argmin())
        if abs(times[i] - frame["timestamp_seconds"]) > tolerance_seconds:
            continue
        for d in frame["detections"]:
            d["latitude"] = float(df.iloc[i]["latitude"])
            d["longitude"] = float(df.iloc[i]["longitude"])
            matched += 1
    return matched


def _best_frames(video_path, result, n=3):
    """Annotated preview frames for the n most confident frames."""
    scored = [
        (max(d["confidence"] for d in f["detections"]), f)
        for f in result["frames"] if f["detections"]
    ]
    scored.sort(key=lambda t: t[0], reverse=True)
    previews = []
    cap = cv2.VideoCapture(str(video_path))
    try:
        for _, f in scored[:n]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f["frame_index"])
            ok, frame = cap.read()
            if ok:
                previews.append((f["timestamp_seconds"], draw_boxes(frame, f["detections"])))
    finally:
        cap.release()
    return previews


def run_video_analysis(file_bytes, filename, frame_stride=5, gps_csv_bytes=None):
    path = _write_temp(file_bytes, filename)
    try:
        result = analyze_video(path, frame_stride=frame_stride)
        gps_matched = attach_gps_from_csv(result, gps_csv_bytes) if gps_csv_bytes else 0
        previews = _best_frames(path, result)
    finally:
        _safe_remove(path)
    result["video"] = filename
    return result, previews, gps_matched


# ------------------------------------------------------------ table helpers
def detections_to_rows(detections, frame_index=None, timestamp_seconds=None):
    """Plain rows for saving to the database (confidence kept as 0-1)."""
    return [
        {
            "label": d["label"], "confidence": d["confidence"], "severity": d["severity"],
            "latitude": d.get("latitude"), "longitude": d.get("longitude"),
            "frame_index": frame_index, "timestamp_seconds": timestamp_seconds,
        }
        for d in detections
    ]


def video_rows(result):
    rows = []
    for f in result["frames"]:
        rows.extend(detections_to_rows(f["detections"], f["frame_index"], f["timestamp_seconds"]))
    return rows


def display_table(rows):
    """Human-friendly DataFrame for showing in the UI."""
    data = []
    for r in rows:
        has_gps = r.get("latitude") is not None and r.get("longitude") is not None
        row = {
            "Label": r["label"],
            "Confidence %": round(r["confidence"] * 100, 1),
            "Severity": r["severity"],
            "Latitude": round(r["latitude"], 6) if has_gps else NO_GPS,
            "Longitude": round(r["longitude"], 6) if has_gps else NO_GPS,
        }
        if r.get("timestamp_seconds") is not None:
            row = {"Time (s)": r["timestamp_seconds"], "Frame": r["frame_index"], **row}
        data.append(row)
    return pd.DataFrame(data)


def severity_counts(rows):
    counts = {s: 0 for s in SEVERITIES}
    for r in rows:
        if r["severity"] in counts:
            counts[r["severity"]] += 1
    return counts