"""
Live rust/corrosion detector with bounding boxes drawn on screen.

All paths are relative to this script's folder — move the whole SpiderBot
folder anywhere and it still works with no changes.

USAGE:
    Set SOURCE to one of:
      - dataset test images:   SOURCE = HERE / "datasets/v1_small/test/images"
      - your own image:        SOURCE = HERE / "my_photo.jpg"
      - a folder of images:    SOURCE = HERE / "my_photos"
      - a video file:          SOURCE = HERE / "my_video.mp4"
      - webcam:                SOURCE = 0
      - ESP32-CAM stream:      SOURCE = "http://192.168.1.X:81/stream"
      - Raspberry Pi camera:   SOURCE = "http://<pi-ip>:8080"   (stream started on the Pi)

    Drop images/videos directly into the SpiderBot folder, then just reference
    them by filename.  No absolute paths needed.

CONTROLS (video/webcam):
    SPACE   - pause / resume
    ESC     - quit

OUTPUT:
    Annotated results are saved automatically to the 'output/' folder.
    Each frame shows the rust mask, the boxes and a severity banner
    (NONE / MILD / MODERATE / SEVERE, from the corroded % of the frame).
"""

from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import cv2
import severity

HERE = Path(__file__).parent

MODEL_PATH = HERE / "models/v3_yolo11s.pt"
#SOURCE = HERE / "videos/rust_7.mp4"
SOURCE = "http://172.20.10.3:8080"


CONFIDENCE_THRESHOLD = 0.4
OUTPUT_DIR = HERE / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

if not MODEL_PATH.exists():
    print(f"ERROR: trained model not found at '{MODEL_PATH}'.")
    raise SystemExit(1)

model = YOLO(str(MODEL_PATH))

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MAX_W, MAX_H = 1280, 720

def fit_to_screen(frame):
    h, w = frame.shape[:2]
    scale = min(MAX_W / w, MAX_H / h, 1.0)
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return frame

def annotate(frame):
    analysis = severity.analyze(frame, model, conf=CONFIDENCE_THRESHOLD)
    return severity.draw(frame, analysis)

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def overlay_text(frame, text, color=(0, 200, 255)):
    h, w = frame.shape[:2]
    cv2.putText(frame, text, (w // 2 - 200, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5)
    cv2.putText(frame, text, (w // 2 - 200, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    return frame

#source = Path(SOURCE) if isinstance(SOURCE, (str, Path)) and SOURCE != 0 else SOURCE
is_url = isinstance(SOURCE, str) and SOURCE.startswith("http")
source = SOURCE if (SOURCE == 0 or is_url) else Path(SOURCE)

# ── image folder mode ──────────────────────────────────────────────────────────
if isinstance(source, Path) and source.is_dir():
    images = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"ERROR: no images found in '{source}'.")
        raise SystemExit(1)
    print(f"Found {len(images)} images. Press any key to advance, ESC to quit.")
    print(f"Annotated images will be saved to '{OUTPUT_DIR}'.")
    for path in images:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        annotated = fit_to_screen(annotate(frame))
        out_path = OUTPUT_DIR / f"{path.stem}_{timestamp()}.jpg"
        cv2.imwrite(str(out_path), annotated)
        cv2.imshow(f"Spider vision  —  {path.name}", annotated)
        if cv2.waitKey(0) & 0xFF == 27:
            break
        cv2.destroyAllWindows()
    cv2.destroyAllWindows()
    print("Done.")

# ── single image mode ──────────────────────────────────────────────────────────
elif isinstance(source, Path) and source.suffix.lower() in IMAGE_EXTS:
    frame = cv2.imread(str(source))
    if frame is None:
        print(f"ERROR: could not read image '{source}'.")
        raise SystemExit(1)
    annotated = fit_to_screen(annotate(frame))
    out_path = OUTPUT_DIR / f"{source.stem}_{timestamp()}.jpg"
    cv2.imwrite(str(out_path), annotated)
    print(f"Saved to '{out_path}'.")
    cv2.imshow("Spider vision", annotated)
    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ── video / webcam / stream mode ───────────────────────────────────────────────
else:
    cap_source = 0 if SOURCE == 0 else str(source)
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        print(f"ERROR: could not open source '{SOURCE}'.")
        raise SystemExit(1)

    # Read one frame to get display dimensions
    ok, first_frame = cap.read()
    if not ok:
        print("ERROR: could not read first frame.")
        raise SystemExit(1)
    sample = fit_to_screen(first_frame)
    fh, fw = sample.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    # Output video file
    is_file = isinstance(source, Path) and source.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")
    stem = source.stem if isinstance(source, Path) else "webcam"
    out_path = OUTPUT_DIR / f"{stem}_{timestamp()}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))
    print(f"Recording output to '{out_path}'.")
    print("SPACE to start / pause / resume  |  ESC to quit")

    # Show first frame and wait for SPACE to start
    preview = fit_to_screen(first_frame.copy())
    overlay_text(preview, "Press SPACE to start")
    cv2.imshow("Spider vision", preview)
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k == 32:   # SPACE — start
            break
        if k == 27:   # ESC — quit before starting
            cap.release()
            writer.release()
            cv2.destroyAllWindows()
            raise SystemExit(0)

    # Annotate and write the first frame
    annotated = fit_to_screen(annotate(first_frame))
    writer.write(annotated)
    cv2.imshow("Spider vision", annotated)

    paused = False
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:    # ESC — quit
            break
        if key == 32:    # SPACE — toggle pause
            paused = not paused

        if paused:
            paused_frame = annotated.copy()
            overlay_text(paused_frame, "PAUSED — SPACE to resume")
            cv2.imshow("Spider vision", paused_frame)
            continue

        ok, frame = cap.read()
        if not ok:
            if is_file:
                break
            continue

        annotated = fit_to_screen(annotate(frame))
        writer.write(annotated)
        cv2.imshow("Spider vision", annotated)

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Saved to '{out_path}'.")
