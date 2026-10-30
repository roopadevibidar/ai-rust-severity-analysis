"""
SQLite storage for the inspection system: assets, inspections (with their
photos), work orders and plant settings.

The database and the saved photos live in data/ next to the app. Delete that
folder to start from an empty plant.
"""

from contextlib import contextmanager
from datetime import date, datetime
import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
IMAGE_DIR = DATA_DIR / "images"
DB_PATH = DATA_DIR / "inspections.db"

CRITICALITY = ["Low", "Medium", "High"]
WO_STATUSES = ["Open", "In progress", "Done", "Cancelled"]
OPEN_STATUSES = ("Open", "In progress")

DEFAULT_SETTINGS = {
    "plant_name": "Demo Plant",
    "company": "",
    "model": "v3",
    "confidence": 0.40,
    "mild_from": 0.5,
    "moderate_from": 10.0,
    "severe_from": 30.0,
    "hide_nested": True,
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
    asset_id      TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    asset_type    TEXT,
    location      TEXT,
    criticality   TEXT NOT NULL DEFAULT 'Medium',
    interval_days INTEGER NOT NULL DEFAULT 180,
    notes         TEXT,
    created_at    TEXT NOT NULL,
    active        INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS inspections (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id         TEXT NOT NULL REFERENCES assets(asset_id),
    inspected_on     TEXT NOT NULL,
    recorded_at      TEXT NOT NULL,
    inspector        TEXT NOT NULL,
    severity         TEXT NOT NULL,
    corroded_pct     REAL NOT NULL,
    box_coverage_pct REAL NOT NULL,
    regions          INTEGER NOT NULL,
    model            TEXT,
    confidence       REAL,
    thresholds       TEXT,
    next_due         TEXT,
    notes            TEXT
);
CREATE TABLE IF NOT EXISTS inspection_images (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    inspection_id      INTEGER NOT NULL REFERENCES inspections(id) ON DELETE CASCADE,
    filename           TEXT,
    original_path      TEXT NOT NULL,
    annotated_path     TEXT NOT NULL,
    severity           TEXT NOT NULL,
    corroded_pct       REAL NOT NULL,
    box_coverage_pct   REAL NOT NULL,
    regions            INTEGER NOT NULL,
    regions_json       TEXT,
    color_segmentation INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS work_orders (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id      TEXT NOT NULL REFERENCES assets(asset_id),
    inspection_id INTEGER REFERENCES inspections(id),
    priority      TEXT NOT NULL,
    title         TEXT NOT NULL,
    description   TEXT,
    status        TEXT NOT NULL DEFAULT 'Open',
    assigned_to   TEXT,
    created_at    TEXT NOT NULL,
    due_date      TEXT NOT NULL,
    closed_at     TEXT,
    closing_notes TEXT
);
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_inspections_asset ON inspections(asset_id, inspected_on);
CREATE INDEX IF NOT EXISTS idx_wo_asset ON work_orders(asset_id, status);
"""


@contextmanager
def connect():
    DATA_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    with connect() as c:
        c.executescript(SCHEMA)


def now():
    return datetime.now().isoformat(timespec="seconds")


def rows(sql, params=()):
    with connect() as c:
        return [dict(r) for r in c.execute(sql, params).fetchall()]


def row(sql, params=()):
    found = rows(sql, params)
    return found[0] if found else None


# ── settings ───────────────────────────────────────────────────────────────────
def get_settings():
    stored = {r["key"]: json.loads(r["value"]) for r in rows("SELECT key, value FROM settings")}
    return {**DEFAULT_SETTINGS, **stored}


def save_settings(values):
    with connect() as c:
        c.executemany("INSERT OR REPLACE INTO settings(key, value) VALUES (?, ?)",
                      [(k, json.dumps(v)) for k, v in values.items()])


# ── assets ─────────────────────────────────────────────────────────────────────
def upsert_asset(asset_id, name, asset_type="", location="", criticality="Medium",
                 interval_days=180, notes="", active=True):
    with connect() as c:
        c.execute("""
            INSERT INTO assets(asset_id, name, asset_type, location, criticality,
                               interval_days, notes, created_at, active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(asset_id) DO UPDATE SET
                name=excluded.name, asset_type=excluded.asset_type,
                location=excluded.location, criticality=excluded.criticality,
                interval_days=excluded.interval_days, notes=excluded.notes,
                active=excluded.active
        """, (asset_id, name, asset_type, location, criticality, int(interval_days),
              notes, now(), int(active)))


def get_asset(asset_id):
    return row("SELECT * FROM assets WHERE asset_id = ?", (asset_id,))


def asset_overview(include_inactive=False):
    """Every asset with its latest inspection, next due date and open work orders."""
    return rows(f"""
        SELECT a.*,
               li.id            AS last_inspection_id,
               li.inspected_on  AS last_inspected,
               li.severity      AS severity,
               li.corroded_pct  AS corroded_pct,
               li.next_due      AS next_due,
               (SELECT COUNT(*) FROM inspections i WHERE i.asset_id = a.asset_id) AS inspections,
               (SELECT COUNT(*) FROM work_orders w
                 WHERE w.asset_id = a.asset_id AND w.status IN {OPEN_STATUSES}) AS open_work_orders
        FROM assets a
        LEFT JOIN inspections li ON li.id = (
            SELECT id FROM inspections i
             WHERE i.asset_id = a.asset_id
             ORDER BY inspected_on DESC, id DESC LIMIT 1)
        {'' if include_inactive else 'WHERE a.active = 1'}
        ORDER BY a.asset_id
    """)


# ── inspections ────────────────────────────────────────────────────────────────
def add_inspection(asset_id, inspected_on, inspector, severity, corroded_pct, box_coverage_pct,
                   regions, model, confidence, thresholds, next_due, notes, images):
    """Insert an inspection and its photos. `images` is a list of dicts with the
    inspection_images columns except id / inspection_id. Returns the new id."""
    with connect() as c:
        cur = c.execute("""
            INSERT INTO inspections(asset_id, inspected_on, recorded_at, inspector, severity,
                                    corroded_pct, box_coverage_pct, regions, model, confidence,
                                    thresholds, next_due, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (asset_id, str(inspected_on), now(), inspector, severity, corroded_pct,
              box_coverage_pct, regions, model, confidence, json.dumps(thresholds),
              str(next_due), notes))
        inspection_id = cur.lastrowid
        for img in images:
            c.execute("""
                INSERT INTO inspection_images(inspection_id, filename, original_path, annotated_path,
                                              severity, corroded_pct, box_coverage_pct, regions,
                                              regions_json, color_segmentation)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (inspection_id, img["filename"], img["original_path"], img["annotated_path"],
                  img["severity"], img["corroded_pct"], img["box_coverage_pct"], img["regions"],
                  json.dumps(img["regions_json"]), int(img["color_segmentation"])))
        return inspection_id


def get_inspection(inspection_id):
    return row("SELECT * FROM inspections WHERE id = ?", (inspection_id,))


def inspection_images(inspection_id):
    return rows("SELECT * FROM inspection_images WHERE inspection_id = ? ORDER BY id", (inspection_id,))


def asset_inspections(asset_id):
    return rows("SELECT * FROM inspections WHERE asset_id = ? ORDER BY inspected_on, id", (asset_id,))


def recent_inspections(limit=10):
    return rows("""
        SELECT i.*, a.name AS asset_name FROM inspections i
        JOIN assets a ON a.asset_id = i.asset_id
        ORDER BY i.inspected_on DESC, i.id DESC LIMIT ?
    """, (limit,))


# ── work orders ────────────────────────────────────────────────────────────────
def add_work_order(asset_id, inspection_id, priority, title, description, due_date, created_at=None):
    with connect() as c:
        cur = c.execute("""
            INSERT INTO work_orders(asset_id, inspection_id, priority, title, description,
                                    created_at, due_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (asset_id, inspection_id, priority, title, description, created_at or now(), str(due_date)))
        return cur.lastrowid


def update_work_order(wo_id, **fields):
    if fields.get("status") in ("Done", "Cancelled") and "closed_at" not in fields:
        fields["closed_at"] = now()
    if fields.get("status") in OPEN_STATUSES:
        fields["closed_at"] = None
    cols = ", ".join(f"{k} = ?" for k in fields)
    with connect() as c:
        c.execute(f"UPDATE work_orders SET {cols} WHERE id = ?", (*fields.values(), wo_id))


def open_work_order(asset_id):
    return row(f"""
        SELECT * FROM work_orders WHERE asset_id = ? AND status IN {OPEN_STATUSES}
        ORDER BY priority, due_date LIMIT 1
    """, (asset_id,))


def work_orders(statuses=None):
    where = f"WHERE w.status IN ({','.join('?' * len(statuses))})" if statuses else ""
    return rows(f"""
        SELECT w.*, a.name AS asset_name, a.location, a.criticality FROM work_orders w
        JOIN assets a ON a.asset_id = w.asset_id
        {where}
        ORDER BY CASE WHEN w.status IN {OPEN_STATUSES} THEN 0 ELSE 1 END,
                 w.priority, w.due_date
    """, tuple(statuses or ()))


def work_order_for_inspection(inspection_id):
    return row("SELECT * FROM work_orders WHERE inspection_id = ? ORDER BY id DESC LIMIT 1",
               (inspection_id,))


def reset():
    """Delete every record and saved photo."""
    import shutil
    if DB_PATH.exists():
        DB_PATH.unlink()
    shutil.rmtree(IMAGE_DIR, ignore_errors=True)
    init()


def today():
    return date.today()
