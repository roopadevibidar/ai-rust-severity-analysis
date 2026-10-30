"""
Streamlit web app: upload metal-surface images, detect rust and grade its severity.

USAGE:
    streamlit run app.py
Then open the URL it prints (default http://localhost:8501).
"""

from datetime import datetime
import io
import json
import zipfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

import severity as sv

MODELS = {
    "v3 · YOLO11s (best, default)": sv.HERE / "models/v3_yolo11s.pt",
    "v2 · YOLO11n (fast)": sv.HERE / "models/v2_yolo11n.pt",
    "v1 · YOLOv8n": sv.HERE / "models/v1_yolov8n.pt",
}
LEVEL_EMOJI = {"NONE": "🟢", "MILD": "🟡", "MODERATE": "🟠", "SEVERE": "🔴"}
LEVEL_ADVICE = {
    "NONE": "No significant corrosion found. Continue routine inspection.",
    "MILD": "Surface rust. Clean and re-coat at the next scheduled maintenance.",
    "MODERATE": "Spreading corrosion. Plan treatment soon and check for pitting.",
    "SEVERE": "Extensive corrosion. Inspect for section loss and prioritize repair.",
}

st.set_page_config(page_title="Rust Severity Analysis", page_icon="🔩", layout="wide")


@st.cache_resource
def load_model(path):
    return YOLO(str(path))


def to_rgb(image_bgr):
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def jpg_bytes(image_bgr):
    return cv2.imencode(".jpg", image_bgr)[1].tobytes()


# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Model", list(MODELS))
    conf = st.slider("Detection confidence", 0.05, 0.95, 0.40, 0.05,
                     help="Boxes below this confidence are ignored.")
    st.subheader("Severity thresholds")
    st.caption("Corroded area, % of the image, at which each level starts.")
    mild_from = st.number_input("Mild from (%)", 0.0, 100.0, sv.MILD_FROM, 0.5)
    moderate_from = st.number_input("Moderate from (%)", 0.0, 100.0, sv.MODERATE_FROM, 1.0)
    severe_from = st.number_input("Severe from (%)", 0.0, 100.0, sv.SEVERE_FROM, 1.0)
    if not mild_from <= moderate_from <= severe_from:
        st.error("Thresholds must increase: mild ≤ moderate ≤ severe.")
        st.stop()
    show_mask = st.checkbox("Show rust mask", value=True)

st.title("🔩 AI-based Rust Detection & Severity Analysis")
st.write(
    "Upload images of metal surfaces. A YOLO model locates rust, rust-colored pixels inside "
    "each detection form the rust mask, and the **corroded area** (% of the image) sets the severity."
)

files = st.file_uploader("Images", type=["jpg", "jpeg", "png", "bmp", "webp"], accept_multiple_files=True)
if not files:
    st.info("Upload one or more images to start.")
    st.stop()

model = load_model(MODELS[model_name])
summary_rows, report, annotated_files = [], [], {}

for f in files:
    image = cv2.imdecode(np.frombuffer(f.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        st.error(f"{f.name}: could not read image.")
        continue
    with st.spinner(f"Analyzing {f.name}…"):
        a = sv.analyze(image, model, conf, mild_from, moderate_from, severe_from)
    annotated = sv.draw(image, a, mask_alpha=0.45 if show_mask else 0.0)
    annotated_files[f"{f.name.rsplit('.', 1)[0]}_severity.jpg"] = jpg_bytes(annotated)

    st.divider()
    st.subheader(f"{LEVEL_EMOJI[a.level]} {f.name} — {a.level}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Severity", a.level)
    m2.metric("Corroded area", f"{a.corroded_pct:.1f}%")
    m3.metric("Detected box area", f"{a.box_coverage_pct:.1f}%")
    m4.metric("Rust regions", len(a.regions))
    st.caption(LEVEL_ADVICE[a.level])
    if not a.used_color_mask:
        st.warning("Image is nearly grayscale, so rust can't be separated by color: "
                   "the whole box area is counted as corroded.")

    c1, c2 = st.columns(2)
    c1.image(to_rgb(image), caption="Original", use_container_width=True)
    c2.image(to_rgb(annotated), caption="Detection + rust mask", use_container_width=True)

    if a.regions:
        regions = pd.DataFrame([{
            "Region": i,
            "Confidence": round(r.confidence, 2),
            "Box area (% image)": round(r.box_area_pct, 2),
            "Rust inside box (%)": round(r.rust_fill_pct, 1),
            "Region severity": r.level,
            "Box (x1, y1, x2, y2)": str(r.box),
        } for i, r in enumerate(a.regions, 1)])
        st.dataframe(regions, hide_index=True, use_container_width=True)

    summary_rows.append({
        "Image": f.name,
        "Severity": a.level,
        "Corroded area (%)": round(a.corroded_pct, 2),
        "Box area (%)": round(a.box_coverage_pct, 2),
        "Rust regions": len(a.regions),
    })
    report.append({
        "image": f.name,
        "severity": a.level,
        "corroded_pct": round(a.corroded_pct, 3),
        "box_coverage_pct": round(a.box_coverage_pct, 3),
        "color_segmentation": a.used_color_mask,
        "regions": [{
            "box": r.box,
            "confidence": round(r.confidence, 3),
            "box_area_pct": round(r.box_area_pct, 3),
            "rust_fill_pct": round(r.rust_fill_pct, 2),
            "severity": r.level,
        } for r in a.regions],
    })

# ── summary + downloads ────────────────────────────────────────────────────────
if summary_rows:
    st.divider()
    st.header("Summary")
    summary = pd.DataFrame(summary_rows)
    st.dataframe(summary, hide_index=True, use_container_width=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {
        "generated": stamp,
        "model": model_name,
        "confidence": conf,
        "thresholds_pct": {"mild": mild_from, "moderate": moderate_from, "severe": severe_from},
        "images": report,
    }
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as z:
        for name, data in annotated_files.items():
            z.writestr(name, data)
        z.writestr("summary.csv", summary.to_csv(index=False))
        z.writestr("report.json", json.dumps(meta, indent=2))

    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇ Summary (CSV)", summary.to_csv(index=False), f"rust_summary_{stamp}.csv", "text/csv")
    d2.download_button("⬇ Full report (JSON)", json.dumps(meta, indent=2), f"rust_report_{stamp}.json",
                       "application/json")
    d3.download_button("⬇ Everything + annotated images (ZIP)", zip_buf.getvalue(),
                       f"rust_analysis_{stamp}.zip", "application/zip")
