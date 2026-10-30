"""
Inspection workflow: run the detector with the plant settings, save the photos
and results, schedule the next inspection and raise or escalate a work order.
"""

from datetime import timedelta
import re
import uuid

import cv2

from core import db, maintenance as mt
import severity as sv

MODELS = {
    "v3": ("v3 · YOLO11s (best, default)", sv.HERE / "models/v3_yolo11s.pt"),
    "v2": ("v2 · YOLO11n (fast)", sv.HERE / "models/v2_yolo11n.pt"),
    "v1": ("v1 · YOLOv8n", sv.HERE / "models/v1_yolov8n.pt"),
}
_loaded = {}


def load_model(key):
    if key not in _loaded:
        from ultralytics import YOLO
        _loaded[key] = YOLO(str(MODELS[key][1]))
    return _loaded[key]


def thresholds(settings):
    return {"mild": settings["mild_from"], "moderate": settings["moderate_from"],
            "severe": settings["severe_from"]}


def analyze(image_bgr, settings):
    return sv.analyze(image_bgr, load_model(settings["model"]), settings["confidence"],
                      settings["mild_from"], settings["moderate_from"], settings["severe_from"],
                      settings["hide_nested"])


def _region_rows(analysis):
    return [{"box": r.box, "confidence": round(r.confidence, 3), "box_area_pct": round(r.box_area_pct, 3),
             "rust_fill_pct": round(r.rust_fill_pct, 2), "severity": r.level} for r in analysis.regions]


def save_inspection(asset, inspected_on, inspector, notes, results, settings):
    """Store an inspection. `results` is a list of (filename, image_bgr, analysis).

    The inspection takes the worst photo's severity and the highest corroded %.
    Returns a dict describing what happened, including the work order outcome.
    """
    safe_id = re.sub(r"[^A-Za-z0-9_-]+", "_", asset["asset_id"])
    folder = db.IMAGE_DIR / f"{safe_id}_{inspected_on}_{uuid.uuid4().hex[:8]}"
    folder.mkdir(parents=True, exist_ok=True)

    images, severity = [], "NONE"
    for n, (filename, image, a) in enumerate(results, 1):
        orig, annot = folder / f"{n}_original.jpg", folder / f"{n}_annotated.jpg"
        cv2.imwrite(str(orig), image)
        cv2.imwrite(str(annot), sv.draw(image, a))
        severity = mt.worse(severity, a.level)
        images.append({
            "filename": filename,
            "original_path": orig.relative_to(db.ROOT).as_posix(),
            "annotated_path": annot.relative_to(db.ROOT).as_posix(),
            "severity": a.level,
            "corroded_pct": a.corroded_pct,
            "box_coverage_pct": a.box_coverage_pct,
            "regions": len(a.regions),
            "regions_json": _region_rows(a),
            "color_segmentation": a.used_color_mask,
        })

    corroded = max(i["corroded_pct"] for i in images)
    next_due = mt.next_inspection(inspected_on, severity, asset["interval_days"])
    inspection_id = db.add_inspection(
        asset_id=asset["asset_id"], inspected_on=inspected_on, inspector=inspector,
        severity=severity,
        corroded_pct=corroded,
        box_coverage_pct=max(i["box_coverage_pct"] for i in images),
        regions=sum(i["regions"] for i in images),
        model=MODELS[settings["model"]][1].stem, confidence=settings["confidence"],
        thresholds=thresholds(settings), next_due=next_due, notes=notes, images=images,
    )
    outcome = {"inspection_id": inspection_id, "severity": severity, "next_due": next_due,
               "work_order": None, "work_order_action": None}

    terms = mt.work_order_terms(severity, asset["criticality"])
    if terms is None:
        return outcome
    priority, days = terms
    due = inspected_on + timedelta(days=days)
    existing = db.open_work_order(asset["asset_id"])
    if existing is None:
        wo_id = db.add_work_order(
            asset["asset_id"], inspection_id, priority,
            f"{mt.WO_TITLE[severity]} – {asset['name']}",
            f"Raised by inspection INS-{inspection_id:05d} ({severity}, {corroded:.1f}% corroded).\n"
            f"{mt.ACTION[severity]}",
            due, created_at=f"{inspected_on}T00:00:00")
        outcome.update(work_order=wo_id, work_order_action="created")
    elif priority < existing["priority"] or str(due) < existing["due_date"]:
        db.update_work_order(
            existing["id"], priority=min(priority, existing["priority"]),
            due_date=min(str(due), existing["due_date"]), inspection_id=inspection_id,
            title=f"{mt.WO_TITLE[severity]} – {asset['name']}",
            description=(existing["description"] or "") +
            f"\nEscalated by inspection INS-{inspection_id:05d} ({severity}).\n{mt.ACTION[severity]}")
        outcome.update(work_order=existing["id"], work_order_action="escalated")
    else:
        outcome.update(work_order=existing["id"], work_order_action="existing")
    return outcome
