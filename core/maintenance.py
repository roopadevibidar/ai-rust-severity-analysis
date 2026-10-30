"""
Maintenance rules: what to do about a finding, how urgently, and when to
inspect again. Plain tables so they are easy to review and change.
"""

from datetime import date, timedelta
import numpy as np

LEVELS = ["NONE", "MILD", "MODERATE", "SEVERE"]
LEVEL_ICON = {"NONE": "🟢", "MILD": "🟡", "MODERATE": "🟠", "SEVERE": "🔴", None: "⚪"}
LEVEL_HEX = {   # status palette: good / warning / serious / critical
    "NONE": "#0ca30c", "MILD": "#fab219", "MODERATE": "#ec835a", "SEVERE": "#d03b3b",
}

# (severity, asset criticality) -> (priority, days to complete the work order)
PRIORITY_MATRIX = {
    ("SEVERE", "High"): ("P1", 3),
    ("SEVERE", "Medium"): ("P1", 7),
    ("SEVERE", "Low"): ("P2", 14),
    ("MODERATE", "High"): ("P2", 14),
    ("MODERATE", "Medium"): ("P2", 30),
    ("MODERATE", "Low"): ("P3", 60),
    ("MILD", "High"): ("P3", 60),
    ("MILD", "Medium"): ("P3", 90),
    ("MILD", "Low"): ("P4", 180),
}
PRIORITY_LABEL = {"P1": "P1 · Urgent", "P2": "P2 · High", "P3": "P3 · Planned", "P4": "P4 · Routine"}

ACTION = {
    "NONE": "No action. Continue routine inspections.",
    "MILD": "Maintain: clean the affected area and spot-coat at the next planned maintenance. Monitor.",
    "MODERATE": "Treat: surface preparation (power-tool or blast cleaning), prime and re-coat. "
                "Check for pitting.",
    "SEVERE": "Repair: remove corrosion and check for section loss (e.g. ultrasonic thickness). "
              "Repair or replace the part, then re-coat. Restrict use if load-bearing.",
}
WO_TITLE = {"MILD": "Spot-coat surface rust", "MODERATE": "Treat and re-coat corrosion",
            "SEVERE": "Repair severe corrosion"}

# Re-inspection interval as a fraction of the asset's normal interval; SEVERE is fixed
RECHECK_FACTOR = {"NONE": 1.0, "MILD": 1.0, "MODERATE": 0.5}
SEVERE_RECHECK_DAYS = 30
# Projections further out than this are not shown
PROJECTION_LIMIT_DAYS = 3650


def work_order_terms(severity, criticality):
    """(priority, days) for a finding, or None when no work order is needed."""
    return PRIORITY_MATRIX.get((severity, criticality))


def next_inspection(inspected_on, severity, interval_days):
    if severity == "SEVERE":
        return inspected_on + timedelta(days=SEVERE_RECHECK_DAYS)
    return inspected_on + timedelta(days=round(interval_days * RECHECK_FACTOR[severity]))


def worse(a, b):
    return a if LEVELS.index(a) >= LEVELS.index(b) else b


def trend(inspections, severe_from):
    """Corrosion growth from an asset's inspection history.

    Returns rate in percentage points per month (linear fit over all inspections)
    and the projected date the corroded area reaches the SEVERE threshold, or
    None values when there is not enough history.
    """
    pts = [(date.fromisoformat(i["inspected_on"]), i["corroded_pct"]) for i in inspections]
    days = np.array([(d - pts[0][0]).days for d, _ in pts], float) if pts else np.array([])
    out = {"rate_per_month": None, "severe_on": None, "points": len(pts)}
    if len(pts) < 2 or np.ptp(days) == 0:
        return out
    pct = np.array([p for _, p in pts], float)
    slope, intercept = np.polyfit(days, pct, 1)
    out["rate_per_month"] = float(slope * 30.44)
    latest_fit = slope * days[-1] + intercept
    if slope > 0 and pct[-1] < severe_from:
        days_left = (severe_from - latest_fit) / slope
        if days_left <= PROJECTION_LIMIT_DAYS:     # slower growth: no meaningful projection
            out["severe_on"] = pts[-1][0] + timedelta(days=max(0, int(days_left)))
    return out
