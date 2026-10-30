"""Equipment register and per-asset history: trend, inspections, photos, work orders."""

import pandas as pd
import streamlit as st

from core import db, maintenance as mt, ui

settings = db.get_settings()
st.title("🏭 Assets")
assets = db.asset_overview(include_inactive=True)


def asset_form(existing=None):
    e = existing or {}
    with st.form(f"asset_form_{e.get('asset_id', 'new')}", clear_on_submit=existing is None):
        c1, c2 = st.columns(2)
        asset_id = c1.text_input("Asset ID *", value=e.get("asset_id", ""), disabled=existing is not None,
                                 placeholder="e.g. PIPE-101")
        name = c2.text_input("Name *", value=e.get("name", ""), placeholder="e.g. Cooling water line A")
        c1, c2 = st.columns(2)
        asset_type = c1.text_input("Type", value=e.get("asset_type") or "", placeholder="Piping, tank, structure…")
        location = c2.text_input("Location", value=e.get("location") or "", placeholder="Area / building / rack")
        c1, c2 = st.columns(2)
        criticality = c1.selectbox("Criticality", db.CRITICALITY,
                                   index=db.CRITICALITY.index(e.get("criticality", "Medium")),
                                   help="How bad a failure would be. Higher criticality → more urgent work orders.")
        interval = c2.number_input("Normal inspection interval (days)", 7, 1825, int(e.get("interval_days", 180)),
                                   help="Shortened automatically after MODERATE or SEVERE findings.")
        notes = st.text_area("Notes", value=e.get("notes") or "", height=70)
        active = st.checkbox("Active (shown on the dashboard)", value=bool(e.get("active", 1)))
        if st.form_submit_button("Save asset", type="primary"):
            if not asset_id.strip() or not name.strip():
                st.error("Asset ID and name are required.")
                return
            if existing is None and db.get_asset(asset_id.strip()):
                st.error(f"Asset {asset_id.strip()} already exists.")
                return
            db.upsert_asset(asset_id.strip(), name.strip(), asset_type.strip(), location.strip(), criticality,
                            interval, notes.strip(), active)
            st.session_state["asset_id"] = asset_id.strip()
            st.rerun()


tab_list, tab_add = st.tabs(["Register", "➕ Add asset"])
with tab_add:
    asset_form()

with tab_list:
    if not assets:
        st.info("No assets yet. Add one in the **Add asset** tab.")
        st.stop()
    st.dataframe(pd.DataFrame([{
        "Asset": a["asset_id"], "Name": a["name"], "Type": a["asset_type"], "Location": a["location"],
        "Criticality": a["criticality"], "Severity": ui.sev(a["severity"]),
        "Corroded (%)": a["corroded_pct"], "Last inspected": ui.d(a["last_inspected"]),
        "Next inspection": ui.due_label(a["next_due"]), "Inspections": a["inspections"],
        "Open WOs": a["open_work_orders"], "Active": bool(a["active"]),
    } for a in assets]), hide_index=True, use_container_width=True, column_config={
        "Corroded (%)": st.column_config.NumberColumn(format="%.1f"),
        "Last inspected": st.column_config.DateColumn(format="DD MMM YYYY"),
    })

    ids = [a["asset_id"] for a in assets]
    current = st.session_state.get("asset_id")
    asset_id = st.selectbox("Asset details", ids, index=ids.index(current) if current in ids else 0,
                            format_func=lambda i: next(f"{a['asset_id']} – {a['name']}" for a in assets
                                                       if a["asset_id"] == i))
    st.session_state["asset_id"] = asset_id
    a = next(x for x in assets if x["asset_id"] == asset_id)
    history = db.asset_inspections(asset_id)
    t = mt.trend(history, settings["severe_from"])

    st.divider()
    st.header(f"{a['asset_id']} – {a['name']}")
    st.caption(" · ".join(x for x in (a["asset_type"], a["location"], f"criticality {a['criticality']}",
                                     f"interval {a['interval_days']} days") if x))
    m = st.columns(5)
    m[0].metric("Current severity", ui.sev(a["severity"]))
    m[1].metric("Corroded area", f"{a['corroded_pct']:.1f}%" if a["corroded_pct"] is not None else "–")
    m[2].metric("Growth (pp/month)", f"{t['rate_per_month']:+.2f}" if t["rate_per_month"] is not None else "–",
                help="Change in corroded area, percentage points per month (straight-line fit over all inspections)")
    m[3].metric("Projected SEVERE", f"{t['severe_on']:%b %Y}" if t["severe_on"] else "–",
                help="When the corroded area would reach the SEVERE threshold at the current growth rate")
    m[4].metric("Next inspection", ui.due_label(a["next_due"]))

    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("📷 Inspect this asset"):
        st.switch_page("views/new_inspection.py")
    with b2.popover("✏️ Edit asset"):
        asset_form(db.get_asset(asset_id))

    if not history:
        st.info("Not inspected yet.")
        st.stop()

    st.subheader("Corrosion trend")
    if len(history) >= 2:
        st.altair_chart(ui.trend_chart(history, settings), use_container_width=True)
    else:
        st.caption("The trend appears after the second inspection.")

    st.subheader("Inspection history")
    st.dataframe(pd.DataFrame([{
        "Inspection": f"INS-{i['id']:05d}", "Date": ui.d(i["inspected_on"]), "Inspector": i["inspector"],
        "Severity": ui.sev(i["severity"]), "Corroded (%)": round(i["corroded_pct"], 1),
        "Regions": i["regions"], "Model": i["model"], "Notes": i["notes"],
    } for i in reversed(history)]), hide_index=True, use_container_width=True,
        column_config={"Date": st.column_config.DateColumn(format="DD MMM YYYY")})

    insp_id = st.selectbox("Show inspection", [i["id"] for i in reversed(history)],
                           format_func=lambda i: next(f"INS-{x['id']:05d} · {ui.d(x['inspected_on']):%d %b %Y} · "
                                                      f"{x['severity']}" for x in history if x["id"] == i))
    for img in db.inspection_images(insp_id):
        c1, c2 = st.columns(2)
        c1.image(str(db.ROOT / img["original_path"]), caption=f"{img['filename']} (original)",
                 use_container_width=True)
        c2.image(str(db.ROOT / img["annotated_path"]),
                 caption=f"{img['severity']} · {img['corroded_pct']:.1f}% corroded", use_container_width=True)
    ui.pdf_button(insp_id, f"asset_{insp_id}")

    st.subheader("Work orders")
    wos = db.rows("SELECT * FROM work_orders WHERE asset_id = ? ORDER BY created_at DESC, id DESC", (asset_id,))
    if wos:
        st.dataframe(pd.DataFrame([{
            "WO": f"WO-{w['id']:05d}", "Title": w["title"], "Priority": mt.PRIORITY_LABEL[w["priority"]],
            "Status": w["status"], "Due": ui.d(w["due_date"]), "Closed": ui.d(w["closed_at"]),
            "Assigned to": w["assigned_to"],
        } for w in wos]), hide_index=True, use_container_width=True, column_config={
            "Due": st.column_config.DateColumn(format="DD MMM YYYY"),
            "Closed": st.column_config.DateColumn(format="DD MMM YYYY")})
    else:
        st.caption("No work orders for this asset.")
