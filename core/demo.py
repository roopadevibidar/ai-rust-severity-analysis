"""
Demo plant: six assets with backdated inspections built from a handful of rust
photos, so the dashboard, trends and reports have history to show.

Each asset gets photos in order of increasing corrosion, so its trend rises.
All demo records are marked with the inspector name "Demo inspector".
"""

from datetime import timedelta

from core import db, inspection as ins

ASSETS = [
    # id, name, type, location, criticality, interval days,
    # inspections as (days ago, position in the photos sorted by corrosion: 0 = least, 1 = most)
    ("PIPE-101", "Cooling water line A", "Piping", "Utility block, rack 3", "High", 180,
     [(500, 0.1), (320, 0.4), (20, 0.7)]),
    ("TANK-201", "Raw water storage tank", "Tank", "Tank farm", "High", 365,
     [(560, 0.0), (330, 0.1), (45, 0.2)]),
    ("STR-301", "Pipe rack support columns", "Structure", "Rack 3", "Medium", 365,
     [(600, 0.5), (380, 0.8), (120, 1.0)]),
    ("FLG-102", "Flange joint FJ-12", "Flange", "Pump house", "Medium", 180,
     [(420, 0.2), (200, 0.3), (8, 0.3)]),
    ("BLT-401", "Anchor bolts, pump P-4", "Fasteners", "Pump house", "Medium", 180,
     [(450, 0.3), (260, 0.5), (75, 0.6)]),
    ("RAIL-501", "Platform handrail", "Structure", "Loading bay", "Low", 365,
     [(800, 0.6), (420, 0.9)]),
]
NEW_ASSET = ("PMP-601", "Pump P-6 baseplate", "Equipment base", "Pump house", "Medium", 180)
INSPECTOR = "Demo inspector"


def seed(photos, settings):
    """photos: list of (filename, image_bgr), at least 3. Returns number of inspections."""
    analysed = sorted(((name, img, ins.analyze(img, settings)) for name, img in photos),
                      key=lambda p: p[2].corroded_pct)
    n = len(analysed)
    count = 0
    for asset_id, name, kind, location, crit, interval, history in ASSETS:
        db.upsert_asset(asset_id, name, kind, location, crit, interval, "Demo asset")
        asset = db.get_asset(asset_id)
        for nth, (ago, position) in enumerate(history):
            when = db.today() - timedelta(days=ago)
            fname, img, a = analysed[round(position * (n - 1))]
            out = ins.save_inspection(asset, when, INSPECTOR, "Demo record", [(fname, img, a)], settings)
            count += 1
            if out["work_order"] and nth < len(history) - 1:
                gap = ago - history[nth + 1][0]
                closed = when + timedelta(days=min(20, gap // 2))
                db.update_work_order(out["work_order"], status="Done", closed_at=f"{closed}T16:00:00",
                                     assigned_to="Maintenance crew B",
                                     closing_notes="Cleaned and re-coated (demo record).")
    db.upsert_asset(*NEW_ASSET, notes="Demo asset, not inspected yet")
    return count
