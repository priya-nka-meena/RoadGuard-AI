"""SQLite storage for dashboard detection history."""

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parents[1] / "outputs" / "roadguard.db"


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    with _connect() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_name TEXT NOT NULL,
                label TEXT NOT NULL,
                confidence REAL NOT NULL,
                severity TEXT NOT NULL,
                latitude REAL,
                longitude REAL,
                frame_index INTEGER,
                timestamp_seconds REAL
            )"""
        )


def save_detections(source_type, source_name, rows):
    if not rows:
        return 0
    now = datetime.now().isoformat(timespec="seconds")
    with _connect() as conn:
        conn.executemany(
            """INSERT INTO detections (created_at, source_type, source_name, label,
               confidence, severity, latitude, longitude, frame_index, timestamp_seconds)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            [
                (now, source_type, source_name, r["label"], float(r["confidence"]),
                 r["severity"], r.get("latitude"), r.get("longitude"),
                 r.get("frame_index"), r.get("timestamp_seconds"))
                for r in rows
            ],
        )
    return len(rows)


def load_history():
    with _connect() as conn:
        return pd.read_sql_query("SELECT * FROM detections ORDER BY id DESC", conn)


def clear_history():
    with _connect() as conn:
        conn.execute("DELETE FROM detections")