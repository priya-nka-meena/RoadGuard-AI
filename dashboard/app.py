"""RoadGuard AI dashboard. Run from the repo root:
    python -m streamlit run dashboard/app.py
"""

import io
import json
import sys
import wave
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard import db, services  # noqa: E402

st.set_page_config(page_title="RoadGuard AI", page_icon="🛣️", layout="wide")
db.init_db()

SEV_HEX = {"LOW": "#2ea043", "MEDIUM": "#f09614", "HIGH": "#dc2626"}
SEV_BG = {"LOW": "#dcfce7", "MEDIUM": "#ffedd5", "HIGH": "#fee2e2"}

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

PAGES = {
    "📊 Overview": ("Overview", "Summary of everything detected and saved so far."),
    "🔍 Detect": ("Detect", "Upload a road image or video and run the AI detector."),
    "🗂️ History": ("History", "Every saved detection. Filter, export or delete records."),
    "🗺️ Map": ("Map", "Detections that have real GPS coordinates."),
    "🚨 Alerts": ("Alerts", "HIGH-severity hazards that need attention first."),
}

with st.sidebar:
    st.markdown('<div class="rg-brand">🛣️ RoadGuard AI</div>'
                '<div class="rg-sub">Road hazard monitoring</div>', unsafe_allow_html=True)
    choice = st.radio("Navigation", list(PAGES), label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Detection settings**")
    conf = st.slider("Confidence threshold", 0.10, 0.90, 0.25, 0.05,
                     help="Higher = fewer false alarms but may miss faint potholes.")
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


def priority_banner(counts, play_sound):
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
    if level == "HIGH" and play_sound and sound_on:
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
    else:
        st.markdown(f'<div class="rg-gps">📍 <b>{services.NO_GPS}</b> '
                    "(this image has no GPS metadata)</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------ Detect
def show_image_result(res, play_sound):
    result = res["result"]
    dets = result["detections"]
    rows = services.detections_to_rows(dets)
    counts = services.severity_counts(rows)
    left, right = st.columns(2)
    left.image(res["original"], caption="Original", channels="BGR", width="stretch")
    right.image(res["annotated"], caption="Detections (numbers match the table)", width="stretch")
    severity_cards(counts)
    priority_banner(counts, play_sound)
    if dets:
        styled_table(services.display_table(rows))
    gps_card(result["location"])
    d1, d2, _ = st.columns([1, 1, 3])
    d1.download_button("⬇️ Annotated image", services.png_bytes(res["annotated"]),
                       "roadguard_annotated.png", "image/png")
    d2.download_button("⬇️ Results JSON", json.dumps(result, indent=2, default=str),
                       "roadguard_result.json", "application/json")


def image_tab():
    file = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png"], key="img_up")
    play = False
    if st.button("▶ Run detection", type="primary", key="img_run", disabled=file is None):
        try:
            with st.spinner("Detecting potholes..."):
                data = file.getvalue()
                result, annotated = services.run_image_analysis(data, file.name, conf)
            db.save_detections("image", file.name, services.detections_to_rows(result["detections"]))
            st.session_state["img_result"] = {
                "result": result, "annotated": annotated, "original": services.decode_image(data),
            }
            play = True
        except Exception as e:  # show the real error instead of hiding it
            st.error(f"Detection failed: {e}")
    res = st.session_state.get("img_result")
    if res:
        show_image_result(res, play)


def show_video_result(res, play_sound):
    result = res["result"]
    rows = services.video_rows(result)
    counts = services.severity_counts(rows)
    m = st.columns(3)
    m[0].metric("Frames analysed", result["frames_processed"])
    m[1].metric("Video FPS", round(result["fps"], 1))
    m[2].metric("Detections with GPS", res["gps_matched"])
    severity_cards(counts, "Sightings (all frames)")
    st.caption("The same pothole can appear in many frames, so these are sightings, "
               "not unique potholes.")
    priority_banner(counts, play_sound)
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


# -------------------------------------------------------------------- pages
def history_view(df):
    out = df.copy()
    out["confidence"] = (out["confidence"] * 100).round(1)
    out["latitude"] = out["latitude"].apply(lambda v: services.NO_GPS if pd.isna(v) else f"{v:.6f}")
    out["longitude"] = out["longitude"].apply(lambda v: services.NO_GPS if pd.isna(v) else f"{v:.6f}")
    return out.rename(columns={
        "created_at": "Time", "source_type": "Type", "source_name": "Source", "label": "Label",
        "confidence": "Confidence %", "severity": "Severity", "latitude": "Latitude",
        "longitude": "Longitude", "frame_index": "Frame", "timestamp_seconds": "Video time (s)",
    }).drop(columns=["id"])


def page_overview(df):
    if df.empty:
        st.info("No detections yet. Open **Detect** and analyse an image or video.")
        with st.expander("How RoadGuard AI works", expanded=True):
            st.markdown(
                "1. **Upload** a road image or video.\n"
                "2. **YOLO** finds potholes (label, confidence, bounding box).\n"
                "3. **Severity** is estimated from the box size (LOW / MEDIUM / HIGH).\n"
                "4. **GPS** is read from the image when it exists. It is never invented.\n"
                "5. Results are **saved**, shown on charts and the map, and HIGH ones raise an alert.")
        return
    counts = services.severity_counts(df.to_dict("records"))
    total = len(df)
    k = st.columns(4)
    k[0].metric("Total detections", total)
    k[1].metric("🔴 HIGH priority", counts["HIGH"])
    k[2].metric("📍 With GPS", int(df["latitude"].notna().sum()))
    k[3].metric("Avg confidence", f"{df['confidence'].mean() * 100:.0f}%")
    c1, c2 = st.columns([1, 1.4])
    with c1:
        st.subheader("By severity")
        donut = pd.DataFrame({"Severity": services.SEVERITIES,
                              "Count": [counts[s] for s in services.SEVERITIES]})
        chart = alt.Chart(donut).mark_arc(innerRadius=62).encode(
            theta="Count:Q",
            color=alt.Color("Severity:N", legend=alt.Legend(orient="bottom"),
                            scale=alt.Scale(domain=services.SEVERITIES,
                                            range=[SEV_HEX[s] for s in services.SEVERITIES])),
            tooltip=["Severity", "Count"]).properties(height=280)
        st.altair_chart(chart, width="stretch")
    with c2:
        st.subheader("Activity by day")
        day = df.assign(Day=df["created_at"].str[:10])
        chart2 = alt.Chart(day).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
            x=alt.X("Day:N", title=None), y=alt.Y("count():Q", title="Detections"),
            color=alt.Color("severity:N", title="Severity", legend=alt.Legend(orient="bottom"),
                            scale=alt.Scale(domain=services.SEVERITIES,
                                            range=[SEV_HEX[s] for s in services.SEVERITIES])),
            tooltip=["Day", "severity", "count()"]).properties(height=280)
        st.altair_chart(chart2, width="stretch")
    st.subheader("Recent detections")
    styled_table(history_view(df.head(8)))


def page_history(df):
    if df.empty:
        st.info("History is empty.")
    else:
        f1, f2 = st.columns(2)
        sev = f1.multiselect("Severity", services.SEVERITIES, default=services.SEVERITIES)
        typ = f2.multiselect("Source type", ["image", "video"], default=["image", "video"])
        view = df[df["severity"].isin(sev) & df["source_type"].isin(typ)]
        st.caption(f"{len(view)} of {len(df)} records")
        styled_table(history_view(view))
        st.download_button("⬇️ Download CSV", history_view(view).to_csv(index=False).encode(),
                           "roadguard_history.csv", "text/csv")
    path, size_kb, rows = db.storage_info()
    with st.expander("💾 Storage and delete options"):
        st.write(f"**Where:** `{path}`  \n**Size:** {size_kb:.1f} KB  \n**Records:** {rows}")
        st.caption("Only detection numbers are stored (time, file name, label, confidence, "
                   "severity, GPS, video frame). Your images and videos are never saved.")
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
    st.caption(f"{len(gps)} detection(s) with real GPS. Image GPS is the camera location, "
               "not the exact pothole position.")
    pts = gps.assign(color=gps["severity"].map(SEV_HEX))
    st.map(pts, latitude="latitude", longitude="longitude", color="color", size=25, zoom=13)
    styled_table(history_view(gps))


def page_alerts(df):
    high = df[df["severity"] == "HIGH"] if not df.empty else df
    if high.empty:
        st.success("No HIGH-severity hazards recorded.")
        return
    st.markdown(
        f'<div class="rg-banner" style="background:{SEV_BG["HIGH"]};color:#b91c1c;border-color:#b91c1c">'
        f"🚨 {len(high)} HIGH-severity detection(s) recorded"
        "<small>Prioritise these for inspection and repair.</small></div>", unsafe_allow_html=True)
    styled_table(history_view(high))
    if sound_on and st.button("🔊 Test sound alert"):
        st.audio(beep_wav(), format="audio/wav", autoplay=True)


history = db.load_history()
if choice.endswith("Overview"):
    page_header(choice)
    page_overview(history)
elif choice.endswith("Detect"):
    page_header(choice)
    tab_i, tab_v = st.tabs(["🖼️ Image", "🎞️ Video"])
    with tab_i:
        image_tab()
    with tab_v:
        video_tab()
elif choice.endswith("History"):
    page_header(choice)
    page_history(history)
elif choice.endswith("Map"):
    page_header(choice)
    page_map(history)
else:
    page_header(choice)
    page_alerts(history)

# Hero is filled last so its numbers include anything saved during this run.
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