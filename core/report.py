"""
PDF inspection report for one inspection: asset, result, photos, corrosion
trend, recommended action, work order and sign-off lines.
"""

from datetime import date
import io
import json

import cv2
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core import db, maintenance as mt

INK = (11, 11, 11)
MUTED = (82, 81, 78)
HAIRLINE = (225, 224, 217)
LEVEL_RGB = {k: tuple(int(v[i:i + 2], 16) for i in (1, 3, 5)) for k, v in mt.LEVEL_HEX.items()}
DISCLAIMER = ("AI-assisted result for decision support. Findings must be verified by a "
              "qualified inspector before repair or safety decisions.")


def latin1(text):
    """The built-in PDF fonts are Latin-1 only."""
    text = str(text).replace("–", "-").replace("—", "-").replace("·", "-").replace("≥", ">=")
    return text.encode("latin-1", "replace").decode("latin-1")


class Report(FPDF):
    def __init__(self, title, plant):
        super().__init__(format="A4")
        self.report_title, self.plant = title, plant
        self.set_auto_page_break(True, margin=18)
        self.set_margins(15, 15, 15)

    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*INK)
        self.cell(0, 8, latin1(self.report_title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*MUTED)
        self.cell(0, 5, latin1(self.plant), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*HAIRLINE)
        self.line(15, self.get_y() + 2, 195, self.get_y() + 2)
        self.ln(6)

    def footer(self):
        self.set_y(-13)
        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(*MUTED)
        self.cell(150, 5, latin1(DISCLAIMER))
        self.cell(0, 5, f"Page {self.page_no()}/{{nb}}", align="R")

    def section(self, text):
        self.ln(2)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*INK)
        self.cell(0, 7, latin1(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def fields(self, pairs, label_w=42):
        for label, value in pairs:
            self.set_font("Helvetica", "", 9)
            self.set_text_color(*MUTED)
            self.cell(label_w, 5.5, latin1(label))
            self.set_text_color(*INK)
            self.multi_cell(0, 5.5, latin1(value if value not in (None, "") else "-"),
                            new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def table(self, headers, data, widths):
        self.set_font("Helvetica", "B", 8.5)
        self.set_text_color(*MUTED)
        self.set_draw_color(*HAIRLINE)
        for h, w in zip(headers, widths):
            self.cell(w, 6, latin1(h), border="B")
        self.ln()
        self.set_font("Helvetica", "", 8.5)
        self.set_text_color(*INK)
        for r in data:
            for v, w in zip(r, widths):
                self.cell(w, 5.5, latin1(v), border="B")
            self.ln()


def _image_bytes(path, max_side=1400):
    img = cv2.imread(str(db.ROOT / path))
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return io.BytesIO(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()), w / h


def trend_chart(history, thresholds, upto_id):
    """PNG of the asset's corroded % over time, up to and including this inspection."""
    pts = [(date.fromisoformat(i["inspected_on"]), i["corroded_pct"]) for i in history if i["id"] <= upto_id]
    fig, ax = plt.subplots(figsize=(7, 2.6), dpi=150)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.6)
    ax.set_axisbelow(True)
    for key, label in (("moderate", "MODERATE"), ("severe", "SEVERE")):
        ax.axhline(thresholds[key], color="#9a9a94", linewidth=0.8, linestyle=(0, (4, 3)))
        ax.annotate(f"{label} from {thresholds[key]:g}%", (0, thresholds[key]), xycoords=("axes fraction", "data"),
                    xytext=(4, 3), textcoords="offset points", ha="left", fontsize=7, color="#52514e")
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color="#2a78d6", linewidth=2, marker="o", markersize=5,
            markeredgecolor="white", markeredgewidth=1)
    ax.annotate(f"{ys[-1]:.1f}%", (xs[-1], ys[-1]), xytext=(6, 4), textcoords="offset points",
                fontsize=8, color="#0b0b0b")
    ax.set_ylabel("Corroded area (%)", fontsize=8, color="#52514e")
    ax.set_ylim(0, max(max(ys), thresholds["severe"]) * 1.2)
    ax.tick_params(labelsize=7, colors="#52514e")
    fig.autofmt_xdate()
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf


def build(inspection_id):
    """Return the PDF report for an inspection as bytes."""
    insp = db.get_inspection(inspection_id)
    asset = db.get_asset(insp["asset_id"])
    images = db.inspection_images(inspection_id)
    settings = db.get_settings()
    thresholds = json.loads(insp["thresholds"])
    history = db.asset_inspections(asset["asset_id"])
    wo = db.work_order_for_inspection(inspection_id) or db.open_work_order(asset["asset_id"])
    plant = " - ".join(p for p in (settings["company"], settings["plant_name"]) if p)

    pdf = Report("Corrosion Inspection Report", plant)
    pdf.alias_nb_pages()
    pdf.add_page()

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*MUTED)
    pdf.cell(0, 5, latin1(f"Report INS-{inspection_id:05d}   |   Generated {date.today():%d %b %Y}"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # result banner
    pdf.ln(2)
    level = insp["severity"]
    pdf.set_fill_color(*LEVEL_RGB[level])
    pdf.set_text_color(*((11, 11, 11) if level in ("MILD", "MODERATE") else (255, 255, 255)))
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 11, latin1(f"  {level}   |   corroded area {insp['corroded_pct']:.1f}%   |   "
                           f"{insp['regions']} rust region(s)"), fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.section("Asset")
    pdf.fields([("Asset ID", asset["asset_id"]), ("Name", asset["name"]), ("Type", asset["asset_type"]),
                ("Location", asset["location"]), ("Criticality", asset["criticality"])])

    pdf.section("Inspection")
    pdf.fields([("Date", f"{date.fromisoformat(insp['inspected_on']):%d %b %Y}"),
                ("Inspector", insp["inspector"]),
                ("Method", f"Visual, AI-assisted (model {insp['model']}, confidence {insp['confidence']:.2f})"),
                ("Severity thresholds", f"MILD from {thresholds['mild']:g}%, MODERATE from "
                                        f"{thresholds['moderate']:g}%, SEVERE from {thresholds['severe']:g}% "
                                        f"of the photo corroded"),
                ("Next inspection due", f"{date.fromisoformat(insp['next_due']):%d %b %Y}"),
                ("Notes", insp["notes"])])

    pdf.section("Recommended action")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*INK)
    pdf.multi_cell(0, 5.5, latin1(mt.ACTION[level]), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    if wo:
        pdf.ln(1)
        pdf.fields([("Work order", f"WO-{wo['id']:05d}  {wo['title']}"),
                    ("Priority / due", f"{mt.PRIORITY_LABEL[wo['priority']]}, due "
                                       f"{date.fromisoformat(wo['due_date']):%d %b %Y}"),
                    ("Status", wo["status"])])

    # trend
    if sum(1 for i in history if i["id"] <= inspection_id) >= 2:
        past = [i for i in history if i["id"] <= inspection_id]
        t = mt.trend(past, thresholds["severe"])
        pdf.section("Corrosion trend")
        pdf.image(trend_chart(history, thresholds, inspection_id), w=180)
        text = f"{t['points']} inspections. Growth {t['rate_per_month']:+.2f} percentage points per month."
        if t["severe_on"]:
            text += f" At this rate it is projected to reach the SEVERE threshold around {t['severe_on']:%b %Y}."
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(0, 5.5, latin1(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # photos
    for n, img in enumerate(images, 1):
        orig, ratio = _image_bytes(img["original_path"])
        annot, _ = _image_bytes(img["annotated_path"])
        w = 88
        h = w / ratio
        if h > 75:
            h, w = 75, 75 * ratio
        if pdf.get_y() + h + 40 > pdf.h - 18:
            pdf.add_page()
        pdf.section(f"Photo {n}: {img['filename'] or ''}")
        y = pdf.get_y()
        pdf.image(orig, x=15, y=y, w=w, h=h)
        pdf.image(annot, x=15 + w + 4, y=y, w=w, h=h)
        pdf.set_y(y + h + 1)
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(*MUTED)
        pdf.cell(w + 4, 4, "Original")
        pdf.cell(w, 4, "Detection and rust mask", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)
        pdf.fields([("Result", f"{img['severity']}, {img['corroded_pct']:.1f}% corroded, "
                               f"detected box area {img['box_coverage_pct']:.1f}%")])
        regions = json.loads(img["regions_json"] or "[]")
        if regions:
            pdf.table(["#", "Confidence", "Box area (% photo)", "Rust in box (%)", "Region severity"],
                      [[i, f"{r['confidence']:.2f}", f"{r['box_area_pct']:.1f}", f"{r['rust_fill_pct']:.0f}",
                        r["severity"]] for i, r in enumerate(regions, 1)],
                      [12, 28, 40, 34, 40])

    # sign-off
    if pdf.get_y() > pdf.h - 60:
        pdf.add_page()
    pdf.section("Sign-off")
    pdf.ln(8)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*INK)
    for label in ("Inspected by", "Reviewed by"):
        pdf.cell(60, 6, "_" * 32)
        pdf.cell(30, 6, "")
        pdf.cell(60, 6, "_" * 22, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(*MUTED)
        pdf.cell(90, 5, label)
        pdf.cell(60, 5, "Date", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(*INK)
        pdf.ln(8)
    return bytes(pdf.output())
