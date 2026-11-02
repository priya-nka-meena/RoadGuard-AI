"""SQLite storage for dashboard detection history."""

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parents[1] / "outputs" / "roadguard.db"
STATUSES = ["Reported", "Verified", "In Progress", "Fixed"]


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
                timestamp_seconds REAL,
                status TEXT NOT NULL DEFAULT 'Reported'
            )"""
        )
        # Older databases (from the first dashboard version) have no status column.
        cols = [r[1] for r in conn.execute("PRAGMA table_info(detections)")]
        if "status" not in cols:
            conn.execute(
                "ALTER TABLE detections ADD COLUMN status TEXT NOT NULL DEFAULT 'Reported'"
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


def update_statuses(changes):
    """changes: list of (detection_id, new_status)."""
    valid = [(s, int(i)) for i, s in changes if s in STATUSES]
    if not valid:
        return 0
    with _connect() as conn:
        conn.executemany("UPDATE detections SET status = ? WHERE id = ?", valid)
    return len(valid)


def clear_history():
    with _connect() as conn:
        conn.execute("DELETE FROM detections")


def delete_source(source_name):
    with _connect() as conn:
        conn.execute("DELETE FROM detections WHERE source_name = ?", (source_name,))


def storage_info():
    """(path, size in KB, row count) of the local history database."""
    with _connect() as conn:
        rows = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
    size_kb = DB_PATH.stat().st_size / 1024 if DB_PATH.exists() else 0
    return DB_PATH, size_kb, rows