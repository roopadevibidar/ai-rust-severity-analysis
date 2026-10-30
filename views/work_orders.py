"""Maintenance work orders raised by inspections: track, assign and close them."""

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from core import db, maintenance as mt, ui

st.title("🛠️ Work orders")
today = date.today()
all_wos = db.work_orders()
if not all_wos:
    st.info("No work orders yet. They are raised automatically when an inspection finds MILD, MODERATE "
            "or SEVERE corrosion.")
    st.stop()

open_wos = [w for w in all_wos if w["status"] in db.OPEN_STATUSES]
k = st.columns(4)
k[0].metric("Open", len(open_wos))
k[1].metric("Overdue", sum(1 for w in open_wos if ui.d(w["due_date"]) < today))
k[2].metric("Due in 7 days", sum(1 for w in open_wos if today <= ui.d(w["due_date"]) <= today + timedelta(days=7)))
k[3].metric("Closed in last 30 days", sum(1 for w in all_wos if w["status"] == "Done" and w["closed_at"]
                                          and ui.d(w["closed_at"]) >= today - timedelta(days=30)))

c1, c2 = st.columns(2)
statuses = c1.multiselect("Status", db.WO_STATUSES, default=list(db.OPEN_STATUSES))
priorities = c2.multiselect("Priority", list(mt.PRIORITY_LABEL), default=list(mt.PRIORITY_LABEL),
                            format_func=mt.PRIORITY_LABEL.get)
shown = [w for w in all_wos if w["status"] in statuses and w["priority"] in priorities]


def due_text(w):
    if w["status"] not in db.OPEN_STATUSES:
        return ""
    days = (ui.d(w["due_date"]) - today).days
    return f"⚠ {-days} days late" if days < 0 else ("Today" if days == 0 else f"in {days} days")


st.dataframe(pd.DataFrame([{
    "WO": f"WO-{w['id']:05d}", "Priority": mt.PRIORITY_LABEL[w["priority"]], "Status": w["status"],
    "Asset": w["asset_id"], "Title": w["title"], "Location": w["location"], "Due": ui.d(w["due_date"]),
    "Timing": due_text(w), "Assigned to": w["assigned_to"], "From inspection":
        f"INS-{w['inspection_id']:05d}" if w["inspection_id"] else "",
} for w in shown]), hide_index=True, use_container_width=True,
    column_config={"Due": st.column_config.DateColumn(format="DD MMM YYYY")})
if not shown:
    st.caption("No work orders match the filters.")
    st.stop()

st.subheader("Update a work order")
wo_id = st.selectbox("Work order", [w["id"] for w in shown],
                     format_func=lambda i: next(f"WO-{w['id']:05d} · {w['asset_id']} · {w['title']}"
                                                for w in shown if w["id"] == i))
w = next(x for x in shown if x["id"] == wo_id)
st.markdown(f"**{w['title']}**  \n{mt.PRIORITY_LABEL[w['priority']]} · due {ui.d(w['due_date']):%d %b %Y} · "
            f"asset {w['asset_id']} ({w['criticality']} criticality)")
st.text(w["description"] or "")
with st.form(f"wo_{wo_id}"):
    c1, c2 = st.columns(2)
    status = c1.selectbox("Status", db.WO_STATUSES, index=db.WO_STATUSES.index(w["status"]))
    assigned = c2.text_input("Assigned to", value=w["assigned_to"] or "")
    closing = st.text_area("Work done / closing notes", value=w["closing_notes"] or "", height=80)
    if st.form_submit_button("Save", type="primary"):
        db.update_work_order(wo_id, status=status, assigned_to=assigned.strip(), closing_notes=closing.strip())
        st.success("Saved.")
        st.rerun()
if w["inspection_id"]:
    ui.pdf_button(w["inspection_id"], f"wo_{wo_id}")
if st.button("🏭 Open asset"):
    ui.open_asset(w["asset_id"])
