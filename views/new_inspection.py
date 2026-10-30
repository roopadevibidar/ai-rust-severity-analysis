"""Record an inspection: pick the asset, upload photos, review the AI result, save."""

from datetime import date, timedelta

import cv2
import numpy as np
import streamlit as st

from core import db, inspection as ins, maintenance as mt, ui
import severity as sv

settings = db.get_settings()
assets = db.asset_overview()
st.title("📷 New inspection")

# Outcome of the inspection saved on the previous run
saved = st.session_state.get("last_saved")
if saved:
    asset_name = db.get_asset(saved["asset_id"])["name"]
    st.success(f"Saved **INS-{saved['inspection_id']:05d}** for {saved['asset_id']} – {asset_name}: "
               f"**{saved['severity']}**. Next inspection due {saved['next_due']:%d %b %Y}.")
    if saved["work_order_action"]:
        wo = db.row("SELECT * FROM work_orders WHERE id = ?", (saved["work_order"],))
        verb = {"created": "Created", "escalated": "Escalated", "existing": "Already open:"}[saved["work_order_action"]]
        st.info(f"{verb} work order **WO-{wo['id']:05d}** · {mt.PRIORITY_LABEL[wo['priority']]} · "
                f"due {ui.d(wo['due_date']):%d %b %Y}")
    c1, c2, c3 = st.columns(3)
    with c1:
        ui.pdf_button(saved["inspection_id"], "saved")
    if c2.button("🏭 View asset"):
        ui.open_asset(saved["asset_id"])
    if c3.button("➕ Start another inspection"):
        del st.session_state["last_saved"]
        st.rerun()
    st.stop()

if not assets:
    st.warning("Register the equipment first on the **Assets** page.")
    if st.button("🏭 Go to Assets"):
        st.switch_page("views/assets.py")
    st.stop()

# ── inspection details ─────────────────────────────────────────────────────────
ids = [a["asset_id"] for a in assets]
default = ids.index(st.session_state["asset_id"]) if st.session_state.get("asset_id") in ids else 0
c1, c2, c3 = st.columns([2, 1, 1])
asset_id = c1.selectbox("Asset", ids, index=default,
                        format_func=lambda i: next(f"{a['asset_id']} – {a['name']} ({a['location'] or '-'})"
                                                   for a in assets if a["asset_id"] == i))
inspector = c2.text_input("Inspector", value=st.session_state.get("inspector", ""), placeholder="Your name")
inspected_on = c3.date_input("Inspection date", value=date.today(), max_value=date.today(), format="DD/MM/YYYY")
notes = st.text_area("Notes (optional)", placeholder="Access, weather, coating condition, anything unusual…",
                     height=80)
asset = db.get_asset(asset_id)
last = next(a for a in assets if a["asset_id"] == asset_id)
st.caption(f"Criticality **{asset['criticality']}** · normal interval {asset['interval_days']} days · "
           f"last inspection: {ui.sev(last['severity'])}"
           + (f" on {ui.d(last['last_inspected']):%d %b %Y}" if last["last_inspected"] else ""))

round_no = st.session_state.setdefault("upload_round", 0)
files = st.file_uploader("Photos of the asset", type=["jpg", "jpeg", "png", "bmp", "webp"],
                         accept_multiple_files=True, key=f"photos_{round_no}")
if not files:
    st.info("Upload one or more photos. The inspection takes the worst result across the photos.")
    st.stop()

# ── analysis (cached per photo and settings) ───────────────────────────────────
cache = st.session_state.setdefault("analysis_cache", {})
settings_key = tuple(sorted(settings.items()))
results = []
for f in files:
    key = (f.file_id, settings_key)
    if key not in cache:
        image = cv2.imdecode(np.frombuffer(f.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            st.error(f"{f.name}: could not read image.")
            continue
        with st.spinner(f"Analyzing {f.name}…"):
            cache[key] = (image, ins.analyze(image, settings))
    image, a = cache[key]
    results.append((f.name, image, a))
if not results:
    st.stop()

severity = "NONE"
for _, _, a in results:
    severity = mt.worse(severity, a.level)
corroded = max(a.corroded_pct for _, _, a in results)

st.divider()
st.subheader(f"Result: {ui.sev(severity)}")
m = st.columns(3)
m[0].metric("Corroded area (worst photo)", f"{corroded:.1f}%")
m[1].metric("Rust regions", sum(len(a.regions) for _, _, a in results))
m[2].metric("Photos", len(results))
st.caption(mt.ACTION[severity])

for name, image, a in results:
    with st.expander(f"{ui.sev(a.level)}  {name}: {a.corroded_pct:.1f}% corroded, {len(a.regions)} region(s)",
                     expanded=len(results) == 1):
        c1, c2 = st.columns(2)
        c1.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
        c2.image(cv2.cvtColor(sv.draw(image, a), cv2.COLOR_BGR2RGB), caption="Detection + rust mask",
                 use_container_width=True)
        if not a.used_color_mask:
            st.warning("Nearly grayscale photo: the whole box area is counted as corroded.")

# ── what saving will do ────────────────────────────────────────────────────────
st.subheader("On save")
next_due = mt.next_inspection(inspected_on, severity, asset["interval_days"])
plan = [f"Next inspection scheduled for **{next_due:%d %b %Y}**."]
terms = mt.work_order_terms(severity, asset["criticality"])
existing = db.open_work_order(asset_id)
if terms:
    priority, days = terms
    due = inspected_on + timedelta(days=days)
    if existing is None:
        plan.append(f"A work order **{mt.PRIORITY_LABEL[priority]}** will be created, due **{due:%d %b %Y}** "
                    f"({severity} on a {asset['criticality'].lower()}-criticality asset).")
    elif priority < existing["priority"] or str(due) < existing["due_date"]:
        plan.append(f"Open work order WO-{existing['id']:05d} will be **escalated** to "
                    f"{mt.PRIORITY_LABEL[min(priority, existing['priority'])]}.")
    else:
        plan.append(f"Work order WO-{existing['id']:05d} is already open for this asset; no new one is needed.")
else:
    plan.append("No work order needed.")
for line in plan:
    st.markdown(f"- {line}")

if st.button("💾 Save inspection", type="primary", disabled=not inspector.strip()):
    st.session_state["inspector"] = inspector.strip()
    with st.spinner("Saving…"):
        outcome = ins.save_inspection(asset, inspected_on, inspector.strip(), notes.strip(), results, settings)
    st.session_state["last_saved"] = {**outcome, "asset_id": asset_id}
    st.session_state["upload_round"] = round_no + 1
    st.session_state["analysis_cache"] = {}
    st.rerun()
if not inspector.strip():
    st.caption("Enter the inspector's name to save.")
