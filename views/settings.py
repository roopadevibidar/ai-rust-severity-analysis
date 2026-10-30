"""Plant settings, demo data and database reset."""

import cv2
import numpy as np
import streamlit as st

from core import db, demo
from core.inspection import MODELS

st.title("⚙️ Settings")
s = db.get_settings()

with st.form("settings"):
    st.subheader("Plant")
    c1, c2 = st.columns(2)
    plant = c1.text_input("Plant / site name", s["plant_name"])
    company = c2.text_input("Company", s["company"])

    st.subheader("Detection")
    c1, c2 = st.columns(2)
    model = c1.selectbox("Model", list(MODELS), index=list(MODELS).index(s["model"]),
                         format_func=lambda k: MODELS[k][0])
    conf = c2.slider("Detection confidence", 0.05, 0.95, float(s["confidence"]), 0.05)
    hide_nested = st.checkbox("Hide boxes inside bigger boxes", bool(s["hide_nested"]))

    st.subheader("Severity thresholds")
    st.caption("Corroded area, % of the photo, at which each level starts. Applies to new inspections; "
               "saved inspections keep the thresholds they were graded with.")
    c1, c2, c3 = st.columns(3)
    mild = c1.number_input("MILD from (%)", 0.0, 100.0, float(s["mild_from"]), 0.5)
    moderate = c2.number_input("MODERATE from (%)", 0.0, 100.0, float(s["moderate_from"]), 1.0)
    severe = c3.number_input("SEVERE from (%)", 0.0, 100.0, float(s["severe_from"]), 1.0)
    if st.form_submit_button("Save settings", type="primary"):
        if not mild <= moderate <= severe:
            st.error("Thresholds must increase: MILD ≤ MODERATE ≤ SEVERE.")
        else:
            db.save_settings({"plant_name": plant.strip() or "Plant", "company": company.strip(), "model": model,
                              "confidence": conf, "hide_nested": hide_nested, "mild_from": mild,
                              "moderate_from": moderate, "severe_from": severe})
            st.success("Settings saved.")
            st.rerun()

st.divider()
st.subheader("Demo data")
st.write(f"Creates {len(demo.ASSETS) + 1} example assets with about two years of backdated inspections and "
         "work orders, built from rust photos you upload (at least 3; more photos give more variety). "
         "Records are marked *Demo inspector*.")
photos = st.file_uploader("Rust photos for the demo", type=["jpg", "jpeg", "png", "bmp", "webp"],
                          accept_multiple_files=True, key="demo_photos")
if st.button("Generate demo plant", disabled=len(photos or []) < 3):
    imgs = [(f.name, cv2.imdecode(np.frombuffer(f.getvalue(), np.uint8), cv2.IMREAD_COLOR)) for f in photos]
    imgs = [(n, i) for n, i in imgs if i is not None]
    with st.spinner(f"Analyzing {len(imgs)} photos and building history…"):
        count = demo.seed(imgs, db.get_settings())
    st.success(f"Created {count} demo inspections. Open the **Dashboard**.")

st.divider()
st.subheader("Danger zone")
confirm = st.checkbox("I understand this permanently deletes every asset, inspection, photo and work order.")
if st.button("🗑 Delete all records", disabled=not confirm):
    db.reset()
    st.session_state.clear()
    st.success("All records deleted.")
