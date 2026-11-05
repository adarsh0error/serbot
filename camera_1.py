import time
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import cv2

# --- Set these two for your rig ---
CAMERA_INDEX = 0          # confirmed working for your robot's external camera
ROTATE_180 = False        # camera is mounted upside down
# -----------------------------------

# Prioritizing SafetyVision v2 as the primary candidate
MODEL_CANDIDATES = [
    ("ayushgupta7777/safetyvision-yolov8", "v2/best.pt"),      # Primary (SafetyVision v2)
    ("baskarmother/yolov8-ppe-construction", "best.pt"),       # Fallback 1
    ("Tanishjain9/yolov8n-ppe-detection-6classes", "best.pt"), # Fallback 2
    ("Nexus3/yolov8n-ppe-detection-6classes", "best.pt"),      # Fallback 3
]

CONF_THRESHOLD = 0.30  # Filter out low-confidence false positives

# Target items mapped to SafetyVision class naming conventions
MONITORED_ITEMS_WANTED = ["Hardhat", "Safety Vest", "Goggles", "Gloves", "Mask"]


def load_model():
    last_error = None
    for repo_id, filename in MODEL_CANDIDATES:
        try:
            print(f"Trying {repo_id} ({filename}) ...")
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
    """Match MONITORED_ITEMS_WANTED against the model's actual class names,
    filtering out negative violation labels like 'NO-Hardhat'."""
    model_classes = list(model.names.values())
    resolved = []
    
    synonym_map = {
        "hardhat": ["hardhat", "helmet"],
        "safety vest": ["safety vest", "vest"],
        "goggles": ["goggles", "glasses", "eye protection"],
        "gloves": ["gloves", "glove"],
        "mask": ["mask", "face mask"]
    }

    for wanted in MONITORED_ITEMS_WANTED:
        wanted_lower = wanted.lower()
        matched_class = None
        
        # 1. Exact match (ignore violation classes like "NO-Hardhat")
        for cls_name in model_classes:
            if cls_name.lower().startswith("no-") or cls_name.lower().startswith("no_"):
                continue
            if cls_name.lower() == wanted_lower:
                matched_class = cls_name
                break
                
        # 2. Synonym match fallback
        if not matched_class and wanted_lower in synonym_map:
            targets = synonym_map[wanted_lower]
            for cls_name in model_classes:
                if cls_name.lower().startswith("no-") or cls_name.lower().startswith("no_"):
                    continue
                if any(target == cls_name.lower() for target in targets):
                    matched_class = cls_name
                    break

        if matched_class:
            resolved.append(matched_class)
        else:
            print(f"  NOTE: '{wanted}' is not explicitly tracked by this model — skipping status line.")
            
    return resolved


def draw_status_panel(frame, monitored_items, detected_labels):
    x, y = 10, 70
    line_h = 32
    for item in monitored_items:
        # Check if the positive PPE class is in the current frame
        present = item in detected_labels
        text = f"{item.upper()}: {'DETECTED' if present else 'ABSENT'}"
        color = (0, 200, 0) if present else (0, 0, 255)  # BGR: Green / Red
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

        # Calculate FPS
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Overlay status panel on top of the annotated frame
        draw_status_panel(annotated, monitored_items, detected_labels)

        cv2.imshow("SerBot PPE Compliance - Live Demo (press 'q' to quit)", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()