"""
Rust severity analysis on top of the YOLO detector.

Pipeline for one image:
    1. Localization  - the YOLO model finds rust regions (bounding boxes).
    2. Segmentation  - inside each box, rust-colored pixels (orange / brown in
                       HSV space) are kept as the rust mask. Boxes overlap, so
                       the mask is a union and no pixel is counted twice.
    3. Severity      - corroded area = rust pixels / image pixels, mapped to
                       NONE / MILD / MODERATE / SEVERE using the thresholds below.
                       Each region is also graded on its own, by how much of its
                       box is rust (REGION_* thresholds).

Boxes that lie completely inside a bigger box are hidden from the region list
and the drawing (they add no new rust area, only clutter).

Images with almost no color (grayscale) can't be segmented by color, so in that
case the whole box area is counted as rust instead.

USAGE (command line):
    python severity.py path/to/image.jpg [more images...]
"""

from dataclasses import dataclass, field
from pathlib import Path
import cv2
import numpy as np

HERE = Path(__file__).parent
DEFAULT_MODEL = HERE / "models/v3_yolo11s.pt"

# Corroded area (% of image) at which each level starts
MILD_FROM = 0.5
MODERATE_FROM = 10.0
SEVERE_FROM = 30.0

# Rust inside a region's box (% of the box) at which each region level starts
REGION_MILD_FROM = 5.0
REGION_MODERATE_FROM = 25.0
REGION_SEVERE_FROM = 60.0

# A box counts as nested when this share of its area lies inside a bigger box
NESTED_OVERLAP = 0.95

# Rust color range in OpenCV HSV (hue 0-180): red-orange to yellow-brown,
# saturated enough not to be gray, bright enough not to be shadow.
RUST_HSV_LOW = (0, 60, 30)
RUST_HSV_HIGH = (25, 255, 230)
RUST_HSV_LOW_WRAP = (170, 60, 30)    # deep red-brown wraps around hue 180
RUST_HSV_HIGH_WRAP = (180, 255, 230)

# The image is treated as grayscale when fewer than GRAYSCALE_MAX_FRACTION of its
# pixels have a saturation above GRAYSCALE_SATURATION (a plain white background
# must not hide colored objects, so this checks for any colored pixels, not the mean).
GRAYSCALE_SATURATION = 60
GRAYSCALE_MAX_FRACTION = 0.01

LEVEL_COLORS = {             # BGR, for drawing
    "NONE": (80, 175, 76),
    "MILD": (59, 235, 255),
    "MODERATE": (0, 152, 255),
    "SEVERE": (54, 67, 244),
    "N/A": (158, 158, 158),  # region level of a grayscale image (no color to measure)
}
MASK_COLOR = (0, 0, 255)


@dataclass
class Region:
    box: tuple               # x1, y1, x2, y2
    confidence: float
    box_area_pct: float      # box area, % of image
    rust_fill_pct: float     # rust pixels inside the box, % of box
    level: str               # severity of this region alone (by rust_fill_pct)


@dataclass
class Analysis:
    level: str
    corroded_pct: float      # rust pixels, % of image (the severity score)
    box_coverage_pct: float  # union of boxes, % of image
    regions: list = field(default_factory=list)
    mask: np.ndarray = None  # uint8, 255 = rust
    used_color_mask: bool = True
    nested_hidden: int = 0   # boxes dropped because they lie inside a bigger box


def severity_level(pct, mild_from=MILD_FROM, moderate_from=MODERATE_FROM, severe_from=SEVERE_FROM):
    if pct >= severe_from:
        return "SEVERE"
    if pct >= moderate_from:
        return "MODERATE"
    if pct >= mild_from:
        return "MILD"
    return "NONE"


def region_level(fill_pct):
    return severity_level(fill_pct, REGION_MILD_FROM, REGION_MODERATE_FROM, REGION_SEVERE_FROM)


def nested_indices(boxes):
    """Indices of boxes that lie (almost) completely inside a bigger box."""
    areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
    nested = set()
    for i, (ax1, ay1, ax2, ay2) in enumerate(boxes):
        for j, (bx1, by1, bx2, by2) in enumerate(boxes):
            if i == j or areas[j] <= areas[i]:
                continue
            iw = max(0, min(ax2, bx2) - max(ax1, bx1))
            ih = max(0, min(ay2, by2) - max(ay1, by1))
            if iw * ih >= NESTED_OVERLAP * max(areas[i], 1):
                nested.add(i)
                break
    return nested


def rust_color_mask(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, RUST_HSV_LOW, RUST_HSV_HIGH)
    mask |= cv2.inRange(hsv, RUST_HSV_LOW_WRAP, RUST_HSV_HIGH_WRAP)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def analyze(image_bgr, model, conf=0.4, mild_from=MILD_FROM,
            moderate_from=MODERATE_FROM, severe_from=SEVERE_FROM, hide_nested=True):
    """Detect rust in a BGR image and grade its severity."""
    h, w = image_bgr.shape[:2]
    image_area = h * w
    result = model(image_bgr, conf=conf, verbose=False)[0]
    boxes = result.boxes.xyxy.cpu().numpy().astype(int) if len(result.boxes) else np.empty((0, 4), int)
    confs = result.boxes.conf.cpu().numpy() if len(result.boxes) else np.empty(0)

    saturation = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)[..., 1]
    use_color = (saturation > GRAYSCALE_SATURATION).mean() >= GRAYSCALE_MAX_FRACTION
    color_mask = rust_color_mask(image_bgr) if use_color else np.full((h, w), 255, np.uint8)

    boxes = [(max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)) for x1, y1, x2, y2 in boxes]
    nested = nested_indices(boxes) if hide_nested else set()

    # Every box counts toward the rust area; nested ones are only left out of the list.
    box_mask = np.zeros((h, w), np.uint8)
    regions = []
    for i, ((x1, y1, x2, y2), c) in enumerate(zip(boxes, confs)):
        box_mask[y1:y2, x1:x2] = 255
        if i in nested:
            continue
        box_px = max((x2 - x1) * (y2 - y1), 1)
        fill_pct = 100 * np.count_nonzero(color_mask[y1:y2, x1:x2]) / box_px
        regions.append(Region(
            box=(int(x1), int(y1), int(x2), int(y2)),
            confidence=float(c),
            box_area_pct=100 * box_px / image_area,
            rust_fill_pct=fill_pct,
            level=region_level(fill_pct) if use_color else "N/A",
        ))

    rust_mask = color_mask & box_mask
    corroded_pct = 100 * np.count_nonzero(rust_mask) / image_area
    return Analysis(
        level=severity_level(corroded_pct, mild_from, moderate_from, severe_from),
        corroded_pct=corroded_pct,
        box_coverage_pct=100 * np.count_nonzero(box_mask) / image_area,
        regions=regions,
        mask=rust_mask,
        used_color_mask=bool(use_color),
        nested_hidden=len(nested),
    )


def draw(image_bgr, analysis, mask_style="outline"):
    """Return a copy of the image with the rust mask, boxes and a severity banner.

    mask_style: "outline" (light tint + contour, rust texture stays visible),
                "fill" (strong tint) or "none".
    """
    out = image_bgr.copy()
    h, w = out.shape[:2]
    thick = max(2, round(min(h, w) / 300))
    font_scale = max(0.5, min(h, w) / 900)

    if mask_style != "none":
        alpha = 0.25 if mask_style == "outline" else 0.45
        tint = np.zeros_like(out)
        tint[:] = MASK_COLOR
        rust = analysis.mask > 0
        out[rust] = cv2.addWeighted(out, 1 - alpha, tint, alpha, 0)[rust]
        if mask_style == "outline":
            contours, _ = cv2.findContours(analysis.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_area = 0.0002 * h * w     # skip specks so the outline stays readable
            contours = [c for c in contours if cv2.contourArea(c) >= min_area]
            cv2.drawContours(out, contours, -1, MASK_COLOR, max(1, thick - 1))

    banner = f"{analysis.level}  |  corroded {analysis.corroded_pct:.1f}%  |  regions {len(analysis.regions)}"
    (bw, bh), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.1, thick)
    banner_w, banner_h = bw + 20, bh + 20

    for i, r in enumerate(analysis.regions, 1):
        x1, y1, x2, y2 = r.box
        color = LEVEL_COLORS[r.level]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        label = f"#{i} {r.level} {r.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
        ty = max(y1, th + 6)
        if x1 < banner_w and ty - th - 6 < banner_h:   # keep the label clear of the banner
            ty = banner_h + th + 6
        cv2.rectangle(out, (x1, ty - th - 6), (x1 + tw + 6, ty), color, -1)
        cv2.putText(out, label, (x1 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thick)

    cv2.rectangle(out, (0, 0), (banner_w, banner_h), LEVEL_COLORS[analysis.level], -1)
    cv2.putText(out, banner, (10, bh + 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.1, (0, 0, 0), thick)
    return out


if __name__ == "__main__":
    import sys
    from ultralytics import YOLO

    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    model = YOLO(str(DEFAULT_MODEL))
    out_dir = HERE / "output"
    out_dir.mkdir(exist_ok=True)
    for arg in sys.argv[1:]:
        image = cv2.imread(arg)
        if image is None:
            print(f"{arg}: could not read image")
            continue
        a = analyze(image, model)
        out_path = out_dir / f"{Path(arg).stem}_severity.jpg"
        cv2.imwrite(str(out_path), draw(image, a))
        print(f"{arg}: {a.level}  corroded {a.corroded_pct:.1f}%  "
              f"boxes {a.box_coverage_pct:.1f}%  regions {len(a.regions)}  -> {out_path}")
