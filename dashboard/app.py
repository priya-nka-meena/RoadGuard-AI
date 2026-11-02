"""RoadGuard AI dashboard. Run from the repo root:
    python -m streamlit run dashboard/app.py
"""

import io
import json
import sys
import time
import wave
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard import db, live, services  # noqa: E402

st.set_page_config(page_title="RoadGuard AI", page_icon="🛣️", layout="wide")
db.init_db()

SEV_HEX = {"LOW": "#2ea043", "MEDIUM": "#f09614", "HIGH": "#dc2626"}
SEV_BG = {"LOW": "#dcfce7", "MEDIUM": "#ffedd5", "HIGH": "#fee2e2"}
SEV_ORDER = ["LOW", "MEDIUM", "HIGH"]
SEV_SCALE = alt.Scale(domain=SEV_ORDER, range=[SEV_HEX[s] for s in SEV_ORDER])

st.markdown(
    """
<style>
.block-container {padding-top: 1.6rem; max-width: 1250px;}
section[data-testid="stSidebar"] {background: #0f172a;}
section[data-testid="stSidebar"] * {color: #e2e8f0 !important;}
section[data-testid="stSidebar"] hr {border-color: #334155;}
.rg-brand {font-size: 1.35rem; font-weight: 800; letter-spacing: .3px; margin-bottom: .2rem;}
.rg-sub {font-size: .78rem; opacity: .7; margin-bottom: 1rem;}
.rg-hero {background: linear-gradient(120deg,#0b1220 0%,#1e3a8a 55%,#2563eb 100%);
  padding: 1.5rem 1.9rem; border-radius: 18px; margin-bottom: 1.3rem;
  box-shadow: 0 10px 28px rgba(37,99,235,.25);}
.rg-hero h1 {margin: 0; font-size: 2.1rem; color: #fff !important; font-weight: 800;}
.rg-hero p {margin: .35rem 0 .9rem 0; color: #cbd5e1 !important;}
.rg-chip {display: inline-block; background: rgba(255,255,255,.14); color: #fff;
  padding: .25rem .75rem; border-radius: 999px; font-size: .82rem; margin-right: .5rem;
  border: 1px solid rgba(255,255,255,.18);}
.rg-title {font-size: 1.45rem; font-weight: 700; margin: .2rem 0 0 0;}
.rg-desc {color: #64748b; margin-bottom: 1rem;}
div[data-testid="stMetric"] {background: #fff; border-radius: 14px; padding: .6rem .9rem;
  box-shadow: 0 2px 10px rgba(15,23,42,.07); border: 1px solid #e2e8f0;}
div[data-testid="stMetricValue"] {font-weight: 800;}
.stButton > button, .stDownloadButton > button {border-radius: 10px; font-weight: 600;}
button[data-baseweb="tab"] {font-weight: 600;}
.rg-banner {border-radius: 14px; padding: .9rem 1.2rem; margin: .8rem 0; font-weight: 600;
  border-left: 7px solid;}
.rg-banner small {display: block; font-weight: 400; opacity: .85; margin-top: .15rem;}
.rg-gps {border-radius: 14px; padding: .8rem 1.2rem; margin: .6rem 0; background: #fff;
  border: 1px solid #e2e8f0;}
</style>
""",
    unsafe_allow_html=True,
)

hero_slot = st.empty()


def render_hero():
    latest = db.load_history()
    n_total = len(latest)
    n_high = int((latest["severity"] == "HIGH").sum()) if n_total else 0
    n_gps = int(latest["latitude"].notna().sum()) if n_total else 0
    hero_slot.markdown(
        '<div class="rg-hero"><h1>🛣️ RoadGuard AI</h1>'
        "<p>Pothole detection, visual severity assessment and location mapping</p>"
        f'<span class="rg-chip">{n_total} detections</span>'
        f'<span class="rg-chip">{n_high} HIGH</span>'
        f'<span class="rg-chip">{n_gps} with GPS</span></div>', unsafe_allow_html=True)


render_hero()

PAGES = {
    "📊 Overview": ("Overview", "Summary of everything detected and saved so far."),
    "🔍 Detect": ("Detect", "Upload road images or a video, or use the live camera."),
    "🗂️ History": ("History", "Every saved detection. Filter, update repair status, export or delete."),
    "🗺️ Map": ("Map", "Detections with real GPS. Hover or click a dot for details."),
    "🚨 Alerts": ("Alerts", "HIGH-severity hazards that are not fixed yet."),
}

with st.sidebar:
    st.markdown('<div class="rg-brand">🛣️ RoadGuard AI</div>'
                '<div class="rg-sub">Road hazard monitoring</div>', unsafe_allow_html=True)
    choice = st.radio("Navigation", list(PAGES), label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Detection settings**")
    conf = st.slider("Confidence threshold", 0.10, 0.90, 0.25, 0.05,
                     help="Minimum score a box needs to be shown. Higher = fewer false "
                          "alarms but may miss faint potholes.")
    sound_on = st.toggle("🔊 Sound alert on HIGH", value=False)
    st.markdown("---")
    st.caption(
        "Severity is a visual estimate from bounding-box size, not a measurement of pothole "
        "depth. GPS appears only when real GPS data exists."
    )


def page_header(key):
    title, desc = PAGES[key]
    st.markdown(f'<div class="rg-title">{title}</div><div class="rg-desc">{desc}</div>',
                unsafe_allow_html=True)


def hex_to_rgb(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


def beep_wav():
    rate, dur, freq = 22050, 0.7, 880
    t = np.linspace(0, dur, int(rate * dur), endpoint=False)
    data = (np.sin(2 * np.pi * freq * t) * 0.5 * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(data.tobytes())
    return buf.getvalue()


def severity_cards(counts, total_label="Total detections"):
    c = st.columns(4)
    c[0].metric(total_label, sum(counts.values()))
    c[1].metric("🟢 LOW", counts["LOW"])
    c[2].metric("🟠 MEDIUM", counts["MEDIUM"])
    c[3].metric("🔴 HIGH", counts["HIGH"])


def priority_banner(counts):
    level = services.priority_level(counts)
    text = {
        "HIGH": ("🚨", "#b91c1c", "Repair priority: HIGH",
                 f"{counts['HIGH']} large pothole(s): box covers more than 10% of the image."),
        "MEDIUM": ("⚠️", "#c2410c", "Repair priority: MEDIUM",
                   "Medium-sized potholes found (box 2% to 10% of the image)."),
        "LOW": ("✅", "#15803d", "Repair priority: LOW", "Only small potholes found."),
        "NONE": ("ℹ️", "#0369a1", "No potholes detected", "Nothing above the confidence threshold."),
    }[level]
    bg = SEV_BG.get(level, "#e0f2fe")
    st.markdown(
        f'<div class="rg-banner" style="background:{bg};color:{text[1]};border-color:{text[1]}">'
        f"{text[0]} {text[2]}<small>{text[3]} (visual estimate)</small></div>",
        unsafe_allow_html=True,
    )
    return level


def play_alert(play_sound):
    if play_sound and sound_on:
        st.audio(beep_wav(), format="audio/wav", autoplay=True)


def styled_table(df):
    """Colour the Severity column and show confidence as a bar."""
    def colour(v):
        return f"background-color:{SEV_BG.get(v, '')};color:{SEV_HEX.get(v, '')};font-weight:700"
    styler = df.style.map(colour, subset=["Severity"]) if "Severity" in df else df
    cfg = {}
    if "Confidence %" in df:
        cfg["Confidence %"] = st.column_config.ProgressColumn(
            "Confidence", min_value=0, max_value=100, format="%.1f%%")
    st.dataframe(styler, column_config=cfg, width="stretch", hide_index=True)


def gps_card(loc):
    if loc:
        st.markdown(
            f'<div class="rg-gps">📍 <b>Image GPS (camera location):</b> '
            f'{loc["latitude"]:.6f}, {loc["longitude"]:.6f}</div>', unsafe_allow_html=True)
        st.map(pd.DataFrame([loc]), latitude="latitude", longitude="longitude",
               size=20, zoom=15, height=240)
    else:
        st.markdown(f'<div class="rg-gps">📍 <b>{services.NO_GPS}</b> '
                    "(no GPS metadata found)</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------ Detect
def show_image_result(item):
    result = item["result"]
    dets = result["detections"]
    rows = services.detections_to_rows(dets)
    counts = services.severity_counts(rows)
    left, right = st.columns(2)
    left.image(item["original"], caption="Original", channels="BGR", width="stretch")
    right.image(item["annotated"], caption="Detections (numbers match the table)", width="stretch")
    severity_cards(counts)
    priority_banner(counts)
    if dets:
        styled_table(services.display_table(rows))
    gps_card(result["location"])
    d1, d2, _ = st.columns([1, 1, 3])
    d1.download_button("⬇️ Annotated image", services.png_bytes(item["annotated"]),
                       f"annotated_{Path(item['name']).stem}.png", "image/png",
                       key=f"dl_img_{item['name']}")
    d2.download_button("⬇️ Results JSON", json.dumps(result, indent=2, default=str),
                       f"result_{Path(item['name']).stem}.json", "application/json",
                       key=f"dl_json_{item['name']}")


def show_batch(batch, play_sound):
    items = batch["items"]
    ok = [i for i in items if i.get("result")]
    failed = [i for i in items if i.get("error")]
    all_rows = [r for i in ok for r in services.detections_to_rows(i["result"]["detections"])]
    counts = services.severity_counts(all_rows)
    if len(items) > 1:
        st.subheader(f"Batch summary: {len(items)} images")
        severity_cards(counts, "Total potholes")
        summary = pd.DataFrame([
            {
                "Image": i["name"],
                "Potholes": len(i["result"]["detections"]) if i.get("result") else 0,
                "Worst severity": (services.priority_level(services.severity_counts(
                    services.detections_to_rows(i["result"]["detections"]))) if i.get("result") else "-"),
                "GPS": ("Yes" if i["result"]["location"] else "No") if i.get("result") else "-",
                "Status": "OK" if i.get("result") else f"Failed: {i['error']}",
            }
            for i in items
        ])
        st.dataframe(summary, width="stretch", hide_index=True)
        if all_rows:
            st.download_button(
                "⬇️ Batch results CSV",
                pd.DataFrame([{"image": i["name"], **r} for i in ok
                              for r in services.detections_to_rows(i["result"]["detections"])])
                .to_csv(index=False).encode(), "roadguard_batch.csv", "text/csv")
    for i in failed:
        st.error(f"{i['name']}: {i['error']}")
    if counts["HIGH"] > 0 and play_sound:
        play_alert(True)
    if ok:
        names = [i["name"] for i in ok]
        pick = names[0] if len(names) == 1 else st.selectbox("View image", names)
        st.markdown("---")
        show_image_result(next(i for i in ok if i["name"] == pick))


def image_tab():
    files = st.file_uploader("Upload one or more road images", type=["jpg", "jpeg", "png"],
                             accept_multiple_files=True, key="img_up")
    play = False
    if st.button("▶ Run detection", type="primary", key="img_run", disabled=not files):
        items, bar = [], st.progress(0.0, text="Starting...")
        for n, f in enumerate(files, start=1):
            bar.progress((n - 1) / len(files), text=f"Analysing {f.name} ({n}/{len(files)})")
            try:
                data = f.getvalue()
                result, annotated = services.run_image_analysis(data, f.name, conf)
                db.save_detections("image", f.name, services.detections_to_rows(result["detections"]))
                items.append({"name": f.name, "result": result, "annotated": annotated,
                              "original": services.decode_image(data)})
            except Exception as e:  # show the real error instead of hiding it
                items.append({"name": f.name, "error": str(e)})
        bar.empty()
        st.session_state["batch"] = {"items": items}
        play = True
    batch = st.session_state.get("batch")
    if batch:
        show_batch(batch, play)


def show_video_result(res, play_sound):
    result = res["result"]
    rows = services.video_rows(result)
    counts = services.severity_counts(rows)
    m = st.columns(4)
    m[0].metric("Frames analysed", result["frames_processed"])
    m[1].metric("Video FPS", round(result["fps"], 1))
    m[2].metric("≈ Unique potholes", services.estimate_unique_potholes(result),
                help="Rough estimate: overlapping boxes in nearby frames are grouped as one "
                     "pothole. It is not real tracking, so treat it as approximate.")
    m[3].metric("Detections with GPS", res["gps_matched"])
    severity_cards(counts, "Sightings (all frames)")
    st.caption("Sightings count the same pothole once per frame it appears in.")
    level = priority_banner(counts)
    if level == "HIGH":
        play_alert(play_sound)
    timeline = services.video_timeline(result)
    if len(timeline):
        st.subheader("Detections over time")
        st.altair_chart(
            alt.Chart(timeline).mark_area(opacity=.5, line=True, color="#2563eb").encode(
                x=alt.X("Time (s):Q"), y=alt.Y("Detections:Q"),
                tooltip=["Time (s)", "Detections"]).properties(height=200),
            width="stretch")
    if res["previews"]:
        st.subheader("Most confident frames")
        cols = st.columns(len(res["previews"]))
        for col, (ts, img) in zip(cols, res["previews"]):
            col.image(img, caption=f"t = {ts}s", width="stretch")
    if rows:
        styled_table(services.display_table(rows))
    if res["gps_matched"] == 0:
        gps_card(None)
    st.download_button("⬇️ Results JSON", json.dumps(result, indent=2, default=str),
                       "roadguard_video_result.json", "application/json")


def video_tab():
    st.caption("Tip: test with a short clip (under 30 seconds) first. Processing runs on your CPU.")
    file = st.file_uploader("Upload a road video", type=["mp4", "avi", "mov", "mkv"], key="vid_up")
    stride = st.slider("Analyse every Nth frame", 1, 30, 5, help="Higher = faster, may miss potholes.")
    gps_csv = st.file_uploader(
        "Optional: real GPS log (CSV with columns timestamp_seconds, latitude, longitude)",
        type=["csv"], key="gps_up")
    play = False
    if st.button("▶ Run video analysis", type="primary", key="vid_run", disabled=file is None):
        try:
            with st.spinner("Analysing video... this can take a while."):
                result, previews, matched = services.run_video_analysis(
                    file.getvalue(), file.name, stride,
                    gps_csv.getvalue() if gps_csv else None, conf)
            for f in result["frames"]:
                db.save_detections(
                    "video", file.name,
                    services.detections_to_rows(f["detections"], f["frame_index"], f["timestamp_seconds"]))
            st.session_state["vid_result"] = {"result": result, "previews": previews, "gps_matched": matched}
            play = True
        except Exception as e:
            st.error(f"Video analysis failed: {e}")
    res = st.session_state.get("vid_result")
    if res:
        show_video_result(res, play)


# -------------------------------------------------------------------- Live
def run_live_loop(source, every_n):
    """Read frames, detect, draw and show them until Stop is pressed."""
    frame_box, alert_box, stats_box, sound_box = st.empty(), st.empty(), st.empty(), st.empty()
    cap = live.open_camera(source)
    if cap is None:
        st.session_state["live_running"] = False
        st.error("Could not open the camera. Close other programs using it (for example "
                 "detect_live.py), check the source, and try again.")
        return
    detector = live.get_detector(conf)
    st.session_state["live_best"] = None
    st.session_state["live_stats"] = None
    st.session_state["live_saved"] = False
    n = frames_with = 0
    detections, best_score, last_beep, start = [], (0, 0.0), 0.0, time.time()
    try:
        while st.session_state.get("live_running"):
            ok, frame = cap.read()
            if not ok:
                break  # camera disconnected or video file ended
            n += 1
            h, w = frame.shape[:2]
            if n % every_n == 1 or every_n == 1:
                detections = live.add_severity(detector.detect(frame), w, h)
            frames_with += 1 if detections else 0
            rgb = services.draw_boxes(frame, detections)
            frame_box.image(rgb, channels="RGB", width="stretch")
            st.session_state["live_last"] = {"rgb": rgb, "detections": detections}
            st.session_state["live_stats"] = {"frames": n, "frames_with": frames_with,
                                              "seconds": time.time() - start}
            score = live.frame_score(detections)
            if detections and score > best_score:
                best_score = score
                st.session_state["live_best"] = {"rgb": rgb, "detections": detections,
                                                 "time": time.strftime("%Y%m%d_%H%M%S")}
            has_high = any(d["severity"] == "HIGH" for d in detections)
            if has_high:
                alert_box.error("🚨 HIGH-severity pothole in view")
                if sound_on and time.time() - last_beep > 5:
                    last_beep = time.time()
                    sound_box.audio(beep_wav(), format="audio/wav", autoplay=True)
            else:
                alert_box.empty()
            if n % 8 == 0:
                with stats_box.container():
                    m = st.columns(4)
                    m[0].metric("Frames shown", n)
                    m[1].metric("Frames with potholes", frames_with)
                    m[2].metric("Potholes in view", len(detections))
                    m[3].metric("Speed (fps)", f"{n / max(time.time() - start, 0.001):.1f}")
    finally:
        cap.release()
    st.session_state["live_running"] = False  # reached only if the camera ended by itself
    st.rerun()


def live_review():
    """After stopping: show the strongest detection of the session and offer to save it."""
    item = st.session_state.get("live_best") or st.session_state.get("live_last")
    if not item:
        return
    dets = item["detections"]
    stats = st.session_state.get("live_stats")
    rows = services.detections_to_rows(dets)
    counts = services.severity_counts(rows)
    st.markdown("---")
    st.subheader("Strongest detection of this session" if dets else "Last frame (no potholes found)")
    img_col, info_col = st.columns([3, 2])
    img_col.image(item["rgb"], width="stretch")
    with info_col:
        if stats and stats["frames"]:
            a, b = st.columns(2)
            a.metric("Session length", f"{stats['seconds']:.0f} s")
            b.metric("Frames shown", stats["frames"])
            c, d = st.columns(2)
            c.metric("Frames with potholes", stats["frames_with"])
            d.metric("Share of frames", f"{100 * stats['frames_with'] / stats['frames']:.0f}%")
        priority_banner(counts)
    severity_cards(counts, "Potholes in this frame")
    if dets:
        styled_table(services.display_table(rows))
    st.markdown(
        '<div class="rg-gps">📍 <b>Location unavailable</b> '
        "(a webcam has no GPS, so no location is recorded for live frames)</div>",
        unsafe_allow_html=True)
    saved = st.session_state.get("live_saved", False)
    label = "✅ Saved to history" if saved else "💾 Save snapshot to history"
    if st.button(label, type="primary", disabled=(not dets) or saved, key="live_save"):
        name = f"live_{item.get('time', time.strftime('%Y%m%d_%H%M%S'))}"
        db.save_detections("live", name, rows)
        st.session_state["live_saved"] = True
        st.rerun()
    if saved:
        st.success("Saved to History (type: live). It has no location, so it will not appear on the Map.")


def live_tab():
    st.caption("Runs Member 1's detector on a live camera. A webcam has no GPS, so location "
               "stays unavailable. Only one program can use the webcam at a time.")
    c1, c2 = st.columns([2, 1])
    source = c1.text_input("Camera source", "0", key="live_source",
                           help="0 = laptop webcam. A phone used as a network camera gives a stream "
                                "URL you can paste here. A video file path also works for testing.")
    every_n = c2.slider("Detect every Nth frame", 1, 5, 2, key="live_every",
                        help="Higher = smoother video on slow laptops, but detection updates less often.")
    running = st.session_state.get("live_running", False)
    b1, b2, _ = st.columns([1, 1, 4])
    b1.button("▶ Start camera", type="primary", disabled=running,
              on_click=lambda: st.session_state.update(live_running=True))
    b2.button("⏹ Stop", disabled=not running,
              on_click=lambda: st.session_state.update(live_running=False))
    if running:
        run_live_loop(live.parse_source(source), every_n)
    else:
        live_review()


# -------------------------------------------------------------------- pages
def history_view(df, keep_id=False):
    out = df.copy()
    out["confidence"] = (out["confidence"] * 100).round(1)
    for col in ("latitude", "longitude"):
        out[col] = out[col].apply(lambda v: services.NO_GPS if pd.isna(v) else f"{v:.6f}")
    out["frame_index"] = out["frame_index"].apply(lambda v: "-" if pd.isna(v) else str(int(v)))
    out["timestamp_seconds"] = out["timestamp_seconds"].apply(
        lambda v: "-" if pd.isna(v) else f"{v:.1f}")
    out = out.rename(columns={
        "created_at": "Time", "source_type": "Type", "source_name": "Source", "label": "Label",
        "confidence": "Confidence %", "severity": "Severity", "status": "Status",
        "latitude": "Latitude", "longitude": "Longitude", "frame_index": "Frame",
        "timestamp_seconds": "Video time (s)",
    })
    cols = ["Time", "Type", "Source", "Label", "Confidence %", "Severity", "Status",
            "Latitude", "Longitude", "Frame", "Video time (s)"]
    return out[(["id"] if keep_id else []) + cols]


def page_overview(df):
    if df.empty:
        st.info("No detections yet. Open **Detect** and analyse an image or video.")
        with st.expander("How RoadGuard AI works", expanded=True):
            st.markdown(
                "1. **Upload** road images or a video.\n"
                "2. **YOLO** finds potholes (label, confidence, bounding box).\n"
                "3. **Severity** is estimated from the box size (LOW / MEDIUM / HIGH).\n"
                "4. **GPS** is read from the image when it exists. It is never invented.\n"
                "5. Results are **saved**, tracked through repair stages, shown on charts and "
                "the map, and HIGH ones raise an alert.")
        return
    counts = services.severity_counts(df.to_dict("records"))
    open_high = int(((df["severity"] == "HIGH") & (df["status"] != "Fixed")).sum())
    k = st.columns(4)
    k[0].metric("Total detections", len(df))
    k[1].metric("🔴 HIGH, not fixed", open_high)
    k[2].metric("📍 With GPS", int(df["latitude"].notna().sum()))
    k[3].metric("✅ Fixed", int((df["status"] == "Fixed").sum()))
    c1, c2, c3 = st.columns([1, 1.3, 1.1])
    with c1:
        st.subheader("By severity")
        donut = pd.DataFrame({"Severity": SEV_ORDER, "Count": [counts[s] for s in SEV_ORDER]})
        st.altair_chart(alt.Chart(donut).mark_arc(innerRadius=55).encode(
            theta="Count:Q",
            color=alt.Color("Severity:N", legend=alt.Legend(orient="bottom"), scale=SEV_SCALE),
            tooltip=["Severity", "Count"]).properties(height=260), width="stretch")
    with c2:
        st.subheader("Activity by day")
        day = df.assign(Day=df["created_at"].str[:10])
        st.altair_chart(alt.Chart(day).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
            x=alt.X("Day:N", title=None), y=alt.Y("count():Q", title="Detections"),
            color=alt.Color("severity:N", title="Severity", legend=alt.Legend(orient="bottom"),
                            scale=SEV_SCALE),
            tooltip=["Day", "severity", "count()"]).properties(height=260), width="stretch")
    with c3:
        st.subheader("Repair status")
        stat = (df["status"].value_counts().reindex(db.STATUSES, fill_value=0)
                .rename_axis("Status").reset_index(name="Count"))
        st.altair_chart(alt.Chart(stat).mark_bar(cornerRadiusEnd=5, color="#2563eb").encode(
            y=alt.Y("Status:N", sort=db.STATUSES, title=None), x=alt.X("Count:Q"),
            tooltip=["Status", "Count"]).properties(height=260), width="stretch")
    st.subheader("Recent detections")
    styled_table(history_view(df.head(8)))


def page_history(df):
    if df.empty:
        st.info("History is empty.")
    else:
        f1, f2, f3 = st.columns(3)
        sev = f1.multiselect("Severity", SEV_ORDER, default=SEV_ORDER)
        typ = f2.multiselect("Source type", ["image", "video", "live"],
                             default=["image", "video", "live"])
        sts = f3.multiselect("Repair status", db.STATUSES, default=db.STATUSES)
        view = df[df["severity"].isin(sev) & df["source_type"].isin(typ) & df["status"].isin(sts)]
        st.caption(f"{len(view)} of {len(df)} records. Change the **Status** column, then save.")
        table = history_view(view, keep_id=True)
        key = "hist_" + "_".join(sev) + "_".join(typ) + "_".join(sts)
        edited = st.data_editor(
            table, key=key, hide_index=True, width="stretch",
            disabled=[c for c in table.columns if c != "Status"],
            column_config={
                "id": None,
                "Status": st.column_config.SelectboxColumn(
                    "Status", options=db.STATUSES, required=True),
                "Confidence %": st.column_config.ProgressColumn(
                    "Confidence", min_value=0, max_value=100, format="%.1f%%"),
            })
        changes = [(int(r_id), new) for r_id, old, new in
                   zip(table["id"], table["Status"], edited["Status"]) if old != new]
        s1, s2, _ = st.columns([1, 1, 3])
        if s1.button("💾 Save status changes", type="primary", disabled=not changes):
            db.update_statuses(changes)
            st.rerun()
        s2.download_button("⬇️ Download CSV", history_view(view).to_csv(index=False).encode(),
                           "roadguard_history.csv", "text/csv")
    path, size_kb, rows = db.storage_info()
    with st.expander("💾 Storage and delete options"):
        st.write(f"**Where:** `{path}`  \n**Size:** {size_kb:.1f} KB  \n**Records:** {rows}")
        st.caption("Only detection numbers are stored (time, file name, label, confidence, "
                   "severity, status, GPS, video frame). Your images and videos are never saved.")
        if not df.empty:
            src = st.selectbox("Delete records of one source", sorted(df["source_name"].unique()))
            if st.button("Delete this source"):
                db.delete_source(src)
                st.rerun()
            if st.checkbox("I understand this deletes ALL saved history"):
                if st.button("Clear all history", type="primary"):
                    db.clear_history()
                    st.rerun()


def page_map(df):
    gps = df.dropna(subset=["latitude", "longitude"]) if not df.empty else df
    if gps.empty:
        st.info("📍 " + services.NO_GPS + ". No saved detection has real GPS yet. "
                "Upload a phone photo taken with location on, or a video with a GPS CSV.")
        return
    f1, f2 = st.columns(2)
    sev = f1.multiselect("Severity", SEV_ORDER, default=SEV_ORDER, key="map_sev")
    sts = f2.multiselect("Repair status", db.STATUSES, default=db.STATUSES, key="map_sts")
    pts = gps[gps["severity"].isin(sev) & gps["status"].isin(sts)].copy()
    if pts.empty:
        st.info("No detections match these filters.")
        return
    pts["rgb"] = pts["severity"].map(lambda s: hex_to_rgb(SEV_HEX[s]))
    pts["conf"] = (pts["confidence"] * 100).round(0).astype(int)
    st.caption(f"{len(pts)} detection(s) with real GPS. Image GPS is the camera location, "
               "not the exact pothole position. Click a dot to see its details below.")
    layer = pdk.Layer(
        "ScatterplotLayer", id="detections", data=pts,
        get_position="[longitude, latitude]", get_fill_color="rgb",
        get_radius=20, radius_min_pixels=8, radius_max_pixels=20,
        stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=2, pickable=True)
    try:
        view = pdk.data_utils.compute_view(pts[["longitude", "latitude"]])
        view.zoom = min(view.zoom, 16)
    except Exception:
        view = pdk.ViewState(latitude=float(pts["latitude"].mean()),
                             longitude=float(pts["longitude"].mean()), zoom=13)
    deck = pdk.Deck(
        layers=[layer], initial_view_state=view, map_provider="carto", map_style="light",
        tooltip={"html": "<b>{severity}</b> · {conf}%<br/>{source_name}<br/>Status: {status}<br/>"
                         "{latitude}, {longitude}",
                 "style": {"backgroundColor": "#0f172a", "color": "white"}})
    event = st.pydeck_chart(deck, on_select="rerun", selection_mode="single-object",
                            key="detections_map")
    selected = []
    try:
        selected = event.selection.objects.get("detections", [])
    except Exception:
        pass
    if selected:
        s = selected[0]
        st.markdown(
            f'<div class="rg-gps">📌 <b>Selected detection</b><br/>'
            f'Severity: <b style="color:{SEV_HEX.get(s.get("severity"), "#000")}">{s.get("severity")}</b>'
            f' · Confidence: {s.get("conf")}% · Status: {s.get("status")}<br/>'
            f'Source: {s.get("source_name")} · Time: {s.get("created_at")}<br/>'
            f'Location: {float(s.get("latitude")):.6f}, {float(s.get("longitude")):.6f}</div>',
            unsafe_allow_html=True)
    styled_table(history_view(pts.drop(columns=["rgb", "conf"])))


def page_alerts(df):
    high = df[(df["severity"] == "HIGH") & (df["status"] != "Fixed")] if not df.empty else df
    if high.empty:
        st.success("No unresolved HIGH-severity hazards.")
        return
    st.markdown(
        f'<div class="rg-banner" style="background:{SEV_BG["HIGH"]};color:#b91c1c;border-color:#b91c1c">'
        f"🚨 {len(high)} HIGH-severity detection(s) not fixed yet"
        "<small>Prioritise these for inspection and repair. Update their status on the History page."
        "</small></div>", unsafe_allow_html=True)
    styled_table(history_view(high))
    if sound_on and st.button("🔊 Test sound alert"):
        st.audio(beep_wav(), format="audio/wav", autoplay=True)


if not choice.endswith("Detect"):
    st.session_state["live_running"] = False  # never keep the camera on in the background
history = db.load_history()
page_header(choice)
if choice.endswith("Overview"):
    page_overview(history)
elif choice.endswith("Detect"):
    tab_i, tab_v, tab_l = st.tabs(["🖼️ Images", "🎞️ Video", "📷 Live camera"])
    with tab_i:
        image_tab()
    with tab_v:
        video_tab()
    with tab_l:
        live_tab()
elif choice.endswith("History"):
    page_history(history)
elif choice.endswith("Map"):
    page_map(history)
else:
    page_alerts(history)

# Refresh the header numbers so they include anything saved during this run.
render_hero()