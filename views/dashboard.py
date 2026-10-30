"""Plant overview: what is corroding, what is overdue, what needs action first."""

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from core import db, maintenance as mt, ui

settings = db.get_settings()
assets = db.asset_overview()
today = date.today()

st.title(f"📊 {settings['plant_name']}: corrosion overview")
st.caption(f"{today:%A %d %B %Y}")

if not assets:
    st.info("No assets registered yet. Add your equipment on the **Assets** page, or generate a demo "
            "plant from a few rust photos under **Settings → Demo data**.")
    c1, c2 = st.columns(2)
    if c1.button("🏭 Add assets"):
        st.switch_page("views/assets.py")
    if c2.button("⚙️ Demo data"):
        st.switch_page("views/settings.py")
    st.stop()

wos = db.work_orders(list(db.OPEN_STATUSES))
overdue_insp = [a for a in assets if not a["next_due"] or ui.d(a["next_due"]) < today]
overdue_wo = [w for w in wos if ui.d(w["due_date"]) < today]
by_level = {lvl: sum(1 for a in assets if a["severity"] == lvl) for lvl in mt.LEVELS}

k = st.columns(6)
k[0].metric("Assets monitored", len(assets))
k[1].metric("🔴 Severe", by_level["SEVERE"])
k[2].metric("🟠 Moderate", by_level["MODERATE"])
k[3].metric("Inspections due", len(overdue_insp), help="Overdue, or never inspected")
k[4].metric("Open work orders", len(wos))
k[5].metric("Work orders overdue", len(overdue_wo))

# ── action list ────────────────────────────────────────────────────────────────
st.subheader("Needs attention")
rank = {lvl: i for i, lvl in enumerate(reversed(mt.LEVELS))}
crit_rank = {"High": 0, "Medium": 1, "Low": 2}
rows = []
for a in assets:
    t = mt.trend(db.asset_inspections(a["asset_id"]), settings["severe_from"])
    rows.append({
        "Asset": a["asset_id"],
        "Name": a["name"],
        "Criticality": a["criticality"],
        "Severity": ui.sev(a["severity"]),
        "Corroded (%)": a["corroded_pct"],
        "Growth (pp/month)": t["rate_per_month"],
        "Last inspected": ui.d(a["last_inspected"]),
        "Next inspection": ui.due_label(a["next_due"]),
        "Open WOs": a["open_work_orders"],
        "_sort": (rank[a["severity"]] if a["severity"] else 0.5,
                  crit_rank[a["criticality"]], -(a["corroded_pct"] or 0)),
    })
rows.sort(key=lambda r: r["_sort"])   # SEVERE first, then never inspected, then by criticality and %
df = pd.DataFrame(rows).drop(columns="_sort")
st.dataframe(df, hide_index=True, use_container_width=True, column_config={
    "Corroded (%)": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
    "Growth (pp/month)": st.column_config.NumberColumn(format="%+.2f",
                                                       help="Change in corroded area, percentage points per month"),
    "Last inspected": st.column_config.DateColumn(format="DD MMM YYYY"),
})
c1, c2 = st.columns([3, 1])
pick = c1.selectbox("Open an asset", [a["asset_id"] for a in assets], label_visibility="collapsed",
                    format_func=lambda i: next(f"{a['asset_id']} – {a['name']}" for a in assets if a["asset_id"] == i))
if c2.button("Open asset ➜", use_container_width=True):
    ui.open_asset(pick)

# ── charts + lists ─────────────────────────────────────────────────────────────
left, right = st.columns([3, 2], gap="large")
with left:
    st.subheader("Latest corroded area by asset")
    chart = ui.corroded_bar(assets, settings)
    if chart is None:
        st.caption("No inspections yet.")
    else:
        st.altair_chart(chart, use_container_width=True)
with right:
    st.subheader("Overdue work orders")
    if overdue_wo:
        st.dataframe(pd.DataFrame([{
            "WO": f"WO-{w['id']:05d}", "Asset": w["asset_id"], "Priority": w["priority"],
            "Due": ui.d(w["due_date"]), "Days late": (today - ui.d(w["due_date"])).days,
        } for w in overdue_wo]), hide_index=True, use_container_width=True,
            column_config={"Due": st.column_config.DateColumn(format="DD MMM YYYY")})
    else:
        st.caption("None. 👍")
    soon = [w for w in wos if today <= ui.d(w["due_date"]) <= today + timedelta(days=14)]
    st.subheader("Due in the next 14 days")
    if soon:
        st.dataframe(pd.DataFrame([{"WO": f"WO-{w['id']:05d}", "Asset": w["asset_id"], "Priority": w["priority"],
                                    "Due": ui.d(w["due_date"])} for w in soon]),
                     hide_index=True, use_container_width=True,
                     column_config={"Due": st.column_config.DateColumn(format="DD MMM YYYY")})
    else:
        st.caption("Nothing due.")

st.subheader("Recent inspections")
recent = db.recent_inspections(8)
if recent:
    st.dataframe(pd.DataFrame([{
        "Inspection": f"INS-{i['id']:05d}", "Date": ui.d(i["inspected_on"]), "Asset": i["asset_id"],
        "Name": i["asset_name"], "Inspector": i["inspector"], "Severity": ui.sev(i["severity"]),
        "Corroded (%)": round(i["corroded_pct"], 1),
    } for i in recent]), hide_index=True, use_container_width=True,
        column_config={"Date": st.column_config.DateColumn(format="DD MMM YYYY")})
else:
    st.caption("No inspections recorded yet.")
