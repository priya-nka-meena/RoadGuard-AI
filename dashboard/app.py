"""RoadGuard AI dashboard. Run from the repo root:
    python -m streamlit run dashboard/app.py
"""

import io
import sys
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard import db, services  # noqa: E402

st.set_page_config(page_title="RoadGuard AI", page_icon="🛣️", layout="wide")
db.init_db()

CSS = """
<style>
.rg-hero {background: linear-gradient(135deg,#0f172a 0%,#1e3a8a 60%,#2563eb 100%);
  padding: 1.4rem 1.8rem; border-radius: 14px; margin-bottom: 1.2rem; color: white;}
.rg-hero h1 {margin: 0; font-size: 2rem; color: white;}
.rg-hero p {margin: .3rem 0 0 0; opacity: .85;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    '<div class="rg-hero"><h1>🛣️ RoadGuard AI</h1>'
    "<p>Road hazard detection, visual severity assessment and location mapping</p></div>",
    unsafe_allow_html=True,
)

page = st.sidebar.radio("Navigation", ["Overview", "Detect", "History", "Map", "Alerts"])
st.sidebar.markdown("---")
sound_on = st.sidebar.toggle("🔊 Sound alert on HIGH severity", value=False)
st.sidebar.caption(
    "Severity is a visual estimate from bounding-box size, not a measurement of pothole depth. "
    "GPS is shown only when real GPS data exists."
)


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
    c[0].metric(total_label, sum(counts.values()), border=True)
    c[1].metric("🟢 LOW", counts["LOW"], border=True)
    c[2].metric("🟠 MEDIUM", counts["MEDIUM"], border=True)
    c[3].metric("🔴 HIGH", counts["HIGH"], border=True)


def high_alert(counts, play_sound):
    if counts["HIGH"] > 0:
        st.error(f"🚨 HIGH-severity hazard detected: {counts['HIGH']} found. Prioritise for repair.")
        if play_sound and sound_on:
            st.audio(beep_wav(), format="audio/wav", autoplay=True)


# ------------------------------------------------------------------ Detect
def show_image_result(res, play_sound):
    dets = res["result"]["detections"]
    rows = services.detections_to_rows(dets)
    counts = services.severity_counts(rows)
    left, right = st.columns(2)
    left.image(res["original"], caption="Original", channels="BGR", width="stretch")
    right.image(res["annotated"], caption="Detections", width="stretch")
    severity_cards(counts)
    high_alert(counts, play_sound)
    if not dets:
        st.info("No potholes detected in this image.")
        return
    st.dataframe(services.display_table(rows), width="stretch", hide_index=True)
    loc = res["result"]["location"]
    if loc:
        st.success(f"📍 Image GPS (camera location): {loc['latitude']:.6f}, {loc['longitude']:.6f}")
    else:
        st.warning("📍 " + services.NO_GPS + " (this image has no GPS metadata).")


def image_tab():
    file = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png"], key="img_up")
    play = False
    if st.button("Run detection", type="primary", key="img_run", disabled=file is None):
        try:
            with st.spinner("Detecting potholes..."):
                data = file.getvalue()
                result, annotated = services.run_image_analysis(data, file.name)
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
    m[0].metric("Frames analysed", result["frames_processed"], border=True)
    m[1].metric("Video FPS", round(result["fps"], 1), border=True)
    m[2].metric("GPS matched", res["gps_matched"], border=True)
    severity_cards(counts, "Detections (all frames)")
    st.caption(
        "The same pothole can appear in many frames, so these counts are sightings, "
        "not unique potholes."
    )
    high_alert(counts, play_sound)
    if res["previews"]:
        st.subheader("Most confident frames")
        cols = st.columns(len(res["previews"]))
        for col, (ts, img) in zip(cols, res["previews"]):
            col.image(img, caption=f"t = {ts}s", width="stretch")
    if rows:
        st.dataframe(services.display_table(rows), width="stretch", hide_index=True)
    else:
        st.info("No potholes detected in the analysed frames.")
    if res["gps_matched"] == 0:
        st.warning("📍 " + services.NO_GPS + " (no GPS log was supplied or matched).")


def video_tab():
    st.caption("Tip: test with a short clip (under 30 seconds) first. Processing runs on your CPU.")
    file = st.file_uploader("Upload a road video", type=["mp4", "avi", "mov", "mkv"], key="vid_up")
    stride = st.slider("Analyse every Nth frame", 1, 30, 5, help="Higher = faster, may miss potholes.")
    gps_csv = st.file_uploader(
        "Optional: real GPS log (CSV with columns timestamp_seconds, latitude, longitude)",
        type=["csv"], key="gps_up",
    )
    play = False
    if st.button("Run video analysis", type="primary", key="vid_run", disabled=file is None):
        try:
            with st.spinner("Analysing video... this can take a while."):
                result, previews, matched = services.run_video_analysis(
                    file.getvalue(), file.name, stride, gps_csv.getvalue() if gps_csv else None
                )
            for f in result["frames"]:
                db.save_detections(
                    "video", file.name,
                    services.detections_to_rows(f["detections"], f["frame_index"], f["timestamp_seconds"]),
                )
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
    out["latitude"] = out["latitude"].apply(lambda v: services.NO_GPS if pd.isna(v) else round(v, 6))
    out["longitude"] = out["longitude"].apply(lambda v: services.NO_GPS if pd.isna(v) else round(v, 6))
    return out.rename(columns={
        "created_at": "Time", "source_type": "Type", "source_name": "Source", "label": "Label",
        "confidence": "Confidence %", "severity": "Severity", "latitude": "Latitude",
        "longitude": "Longitude", "frame_index": "Frame", "timestamp_seconds": "Video time (s)",
    }).drop(columns=["id"])


def page_overview():
    df = db.load_history()
    if df.empty:
        st.info("No detections yet. Go to the **Detect** page and analyse an image or video.")
        return
    counts = services.severity_counts(df.to_dict("records"))
    severity_cards(counts)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Detections by severity")
        st.bar_chart(pd.DataFrame({"Severity": list(counts), "Count": list(counts.values())}),
                     x="Severity", y="Count", color="#2563eb")
    with c2:
        st.subheader("Activity by day")
        daily = df.assign(Day=df["created_at"].str[:10]).groupby("Day").size().reset_index(name="Detections")
        st.bar_chart(daily, x="Day", y="Detections", color="#2563eb")
    st.subheader("Recent detections")
    st.dataframe(history_view(df.head(10)), width="stretch", hide_index=True)


def page_history():
    df = db.load_history()
    if df.empty:
        st.info("History is empty.")
        return
    f1, f2 = st.columns(2)
    sev = f1.multiselect("Severity", services.SEVERITIES, default=services.SEVERITIES)
    typ = f2.multiselect("Source type", ["image", "video"], default=["image", "video"])
    view = df[df["severity"].isin(sev) & df["source_type"].isin(typ)]
    st.caption(f"{len(view)} of {len(df)} records")
    st.dataframe(history_view(view), width="stretch", hide_index=True)
    st.download_button("Download CSV", history_view(view).to_csv(index=False).encode(),
                       "roadguard_history.csv", "text/csv")
    with st.expander("Danger zone"):
        if st.checkbox("I understand this deletes all saved history"):
            if st.button("Clear history"):
                db.clear_history()
                st.rerun()


def page_map():
    df = db.load_history()
    gps = df.dropna(subset=["latitude", "longitude"]) if not df.empty else df
    if gps.empty:
        st.info(
            "📍 " + services.NO_GPS + ". No saved detection has real GPS yet. "
            "Upload a phone photo with location enabled, or a video with a GPS CSV."
        )
        return
    st.caption(f"{len(gps)} detection(s) with real GPS. Image GPS is the camera location, "
               "not the exact pothole position.")
    st.map(gps[["latitude", "longitude"]], zoom=12)
    st.dataframe(history_view(gps), width="stretch", hide_index=True)


def page_alerts():
    df = db.load_history()
    high = df[df["severity"] == "HIGH"] if not df.empty else df
    if high.empty:
        st.success("No HIGH-severity hazards recorded.")
        return
    st.error(f"🚨 {len(high)} HIGH-severity detection(s) recorded.")
    st.dataframe(history_view(high), width="stretch", hide_index=True)
    if sound_on and st.button("Test sound alert"):
        st.audio(beep_wav(), format="audio/wav", autoplay=True)


if page == "Overview":
    page_overview()
elif page == "Detect":
    tab_i, tab_v = st.tabs(["🖼️ Image", "🎞️ Video"])
    with tab_i:
        image_tab()
    with tab_v:
        video_tab()
elif page == "History":
    page_history()
elif page == "Map":
    page_map()
else:
    page_alerts()