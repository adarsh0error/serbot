import time
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import cv2

# --- Set these two for your rig ---
CAMERA_INDEX = 1          # confirmed working for your robot's external camera
ROTATE_180 = True         # camera is mounted upside down
# -----------------------------------

MODEL_CANDIDATES = [
    ("Nexus3/yolov8n-ppe-detection-6classes", "best.pt"),
    ("Tanishjain9/yolov8n-ppe-detection-6classes", "best.pt"),
    ("FatimaNoorAI/ppe-helmet-detection-yolov8", "best.pt"),
    ("JarvanLee/yolov8-helmet-violation-detection", "weights/best.pt"),
    ("keremberke/yolov8n-protective-equipment-detection", "best.pt"),
]

CONF_THRESHOLD = 0.3  # single fixed threshold — no cascading in live video

# What you want a status line for on screen. Matched case-insensitively
# against whatever the loaded model actually calls its classes.
MONITORED_ITEMS_WANTED = ["gloves", "goggles", "helmet", "vest", "mask"]


def load_model():
    last_error = None
    for repo_id, filename in MODEL_CANDIDATES:
        try:
            print(f"Trying {repo_id} ...")
            weights_path = hf_hub_download(repo_id=repo_id, filename=filename)
            model = YOLO(weights_path)
            print(f"  SUCCESS — loaded {repo_id}")
            print("  Classes it can detect:", list(model.names.values()))
            return model
        except Exception as exc:
            last_error = exc
            print(f"  FAILED: {type(exc).__name__}: {exc}")

    raise RuntimeError(f"No PPE model could be loaded. Last error: {last_error}")


def resolve_monitored_items(model):
    """Match MONITORED_ITEMS_WANTED against the model's real class names,
    case-insensitively, so a casing mismatch can't silently break the demo."""
    name_by_lower = {name.lower(): name for name in model.names.values()}
    resolved = []
    for wanted in MONITORED_ITEMS_WANTED:
        if wanted.lower() in name_by_lower:
            resolved.append(name_by_lower[wanted.lower()])
        else:
            print(f"  NOTE: '{wanted}' is not a class this model knows — skipping it on the status panel.")
    return resolved


def draw_status_panel(frame, monitored_items, detected_labels):
    x, y = 10, 70
    line_h = 32
    for item in monitored_items:
        present = item in detected_labels
        text = f"{item.upper()}: {'DETECTED' if present else 'ABSENT'}"
        color = (0, 200, 0) if present else (0, 0, 255)  # BGR: green / red
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y += line_h


def main():
    model = load_model()
    monitored_items = resolve_monitored_items(model)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"!! Could not open camera index {CAMERA_INDEX}.")
        print("   Run find_camera.py again if the index changed.")
        return

    print(f"\nLive feed starting on camera index {CAMERA_INDEX}. Press 'q' to quit.\n")
    prev_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("!! Failed to read frame from camera.")
            break

        if ROTATE_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        results = model.predict(source=frame, conf=CONF_THRESHOLD, iou=0.45, verbose=False)
        result = results[0]
        annotated = result.plot()

        detected_labels = {model.names[int(box.cls[0])] for box in result.boxes}

        # FPS
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        draw_status_panel(annotated, monitored_items, detected_labels)

        cv2.imshow("SerBot PPE Compliance - Live Demo (press 'q' to quit)", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()