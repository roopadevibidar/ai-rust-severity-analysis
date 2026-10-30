"""Small Streamlit helpers shared by the pages."""

from datetime import date

import altair as alt
import pandas as pd
import streamlit as st

from core import db, maintenance as mt, report

SERIES_BLUE = "#2a78d6"
SEVERITY_SCALE = alt.Scale(domain=mt.LEVELS, range=[mt.LEVEL_HEX[k] for k in mt.LEVELS])


def sev(level):
    return f"{mt.LEVEL_ICON[level]} {level}" if level else "⚪ Not inspected"


def d(value):
    """ISO date string -> date (None stays None)."""
    return date.fromisoformat(value[:10]) if value else None


def due_label(value):
    if not value:
        return "Now (never inspected)"
    days = (d(value) - date.today()).days
    if days < 0:
        return f"⚠ {-days} days overdue"
    return "Today" if days == 0 else f"in {days} days"


def pdf_button(inspection_id, key):
    """Generate the PDF on demand (it takes a moment), then offer the download."""
    state_key = f"pdf_{inspection_id}"
    if st.button("📄 Generate PDF report", key=f"gen_{key}"):
        with st.spinner("Building report…"):
            st.session_state[state_key] = report.build(inspection_id)
    if state_key in st.session_state:
        st.download_button("⬇ Download PDF report", st.session_state[state_key],
                           f"INS-{inspection_id:05d}.pdf", "application/pdf", key=f"dl_{key}")


def threshold_rules(settings, orient="y"):
    """Dashed MODERATE / SEVERE threshold lines with labels, for layering onto a chart."""
    data = pd.DataFrame({"value": [settings["moderate_from"], settings["severe_from"]],
                         "label": [f"MODERATE from {settings['moderate_from']:g}%",
                                   f"SEVERE from {settings['severe_from']:g}%"]})
    axis = "y" if orient == "y" else "x"
    pos = {axis: alt.X("value:Q", title=None) if axis == "x" else alt.Y("value:Q", title=None)}
    rules = alt.Chart(data).mark_rule(strokeDash=[4, 3], color="#9a9a94", strokeWidth=1).encode(**pos)
    if axis == "y":
        text = alt.Chart(data).mark_text(align="left", dx=4, dy=-7, fontSize=11, color="#8a8984").encode(
            text="label:N", x=alt.value(0), **pos)
    else:
        text = alt.Chart(data).mark_text(align="left", dx=4, dy=-8, fontSize=11, color="#8a8984").encode(
            text="label:N", y=alt.value(0), **pos)
    return rules + text


def trend_chart(history, settings):
    df = pd.DataFrame([{"Date": d(i["inspected_on"]), "Corroded area (%)": round(i["corroded_pct"], 1),
                        "Severity": i["severity"], "Inspection": f"INS-{i['id']:05d}",
                        "Inspector": i["inspector"]} for i in history])
    df["Date"] = pd.to_datetime(df["Date"])
    tooltip = ["Inspection", alt.Tooltip("Date:T", format="%d %b %Y"), "Corroded area (%)", "Severity",
               "Inspector"]
    top = max(df["Corroded area (%)"].max(), settings["severe_from"]) * 1.2
    y = alt.Y("Corroded area (%):Q", title="Corroded area (% of photo)", scale=alt.Scale(domain=[0, top]))
    line = alt.Chart(df).mark_line(color=SERIES_BLUE, strokeWidth=2).encode(x=alt.X("Date:T", title=None), y=y)
    points = alt.Chart(df).mark_circle(size=110, opacity=1, stroke="white", strokeWidth=2).encode(
        x="Date:T", y=y, color=alt.Color("Severity:N", scale=SEVERITY_SCALE, legend=alt.Legend(orient="top", title=None)),
        tooltip=tooltip)
    return alt.layer(line, points, threshold_rules(settings)).properties(height=300)


def corroded_bar(overview, settings):
    df = pd.DataFrame([{"Asset": a["asset_id"], "Name": a["name"], "Corroded area (%)": round(a["corroded_pct"], 1),
                        "Severity": a["severity"], "Last inspected": a["last_inspected"]}
                       for a in overview if a["severity"]])
    if df.empty:
        return None
    df = df.sort_values("Corroded area (%)", ascending=False)
    top = max(df["Corroded area (%)"].max(), settings["severe_from"]) * 1.15
    x = alt.X("Corroded area (%):Q", title="Corroded area (% of photo)", scale=alt.Scale(domain=[0, top]))
    y = alt.Y("Asset:N", sort=df["Asset"].tolist(), title=None)
    base = alt.Chart(df).encode(x=x, y=y)
    bars = base.mark_bar(cornerRadiusEnd=4, size=18).encode(
        color=alt.Color("Severity:N", scale=SEVERITY_SCALE, legend=alt.Legend(orient="top", title=None)),
        tooltip=["Asset", "Name", "Corroded area (%)", "Severity", "Last inspected"])
    labels = base.mark_text(align="left", dx=4, fontSize=11).encode(
        text=alt.Text("Corroded area (%):Q", format=".1f"))
    return alt.layer(bars, labels, threshold_rules(settings, orient="x")).properties(
        height=34 * len(df) + 20)


def open_asset(asset_id):
    st.session_state["asset_id"] = asset_id
    st.switch_page("views/assets.py")
