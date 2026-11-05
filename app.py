import os
import io
import time
import base64
import threading
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
import cv2
import numpy as np
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

BASE_DIR = Path(__file__).resolve().parent
TEST_IMAGES_DIR = BASE_DIR / "test_images"
OUTPUT_IMAGES_DIR = BASE_DIR / "output_images"

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"), static_folder=str(BASE_DIR / "static"))

# ----------------- MODEL INITIALIZATION -----------------
MODEL_CANDIDATES = [
    ("ayushgupta7777/safetyvision-yolov8", "v2/best.pt"),
    ("baskarmother/yolov8-ppe-construction", "best.pt"),
    ("Tanishjain9/yolov8n-ppe-detection-6classes", "best.pt"),
    ("Nexus3/yolov8n-ppe-detection-6classes", "best.pt"),
]

model = None
model_info = {"repo": "", "filename": "", "classes": []}

def get_ppe_model():
    global model, model_info
    if model is not None:
        return model
    
    last_err = None
    for repo_id, filename in MODEL_CANDIDATES:
        try:
            print(f"[*] Loading model from {repo_id} ({filename})...")
            weights_path = hf_hub_download(repo_id=repo_id, filename=filename)
            loaded = YOLO(weights_path)
            model = loaded
            model_info = {
                "repo": repo_id,
                "filename": filename,
                "classes": list(loaded.names.values()),
                "weights_path": str(weights_path)
            }
            print(f"[+] Loaded successfully: {repo_id}")
            return model
        except Exception as e:
            print(f"[-] Failed loading {repo_id}: {e}")
            last_err = e
            
    # Local fallback if repo fetch fails
    local_candidates = ["yolov8s.pt", "yolov8m.pt", "yolov8n.pt"]
    for lc in local_candidates:
        local_p = BASE_DIR / lc
        if local_p.exists():
            print(f"[*] Loading local weight fallback: {lc}")
            model = YOLO(str(local_p))
            model_info = {"repo": "Local", "filename": lc, "classes": list(model.names.values())}
            return model

    raise RuntimeError(f"Could not load any model: {last_err}")

# Load model at startup
get_ppe_model()

# ----------------- TRACKED CLASSES & LOGIC -----------------
MONITORED_PPE = ["Hardhat", "Safety Vest", "Goggles", "Gloves", "Mask"]

SYNONYM_MAP = {
    "hardhat": ["hardhat", "helmet"],
    "safety vest": ["safety vest", "vest"],
    "goggles": ["goggles", "glasses", "eye protection"],
    "gloves": ["gloves", "glove"],
    "mask": ["mask", "face mask"]
}

def analyze_detections(boxes, model_names):
    detected_classes = []
    item_counts = {}
    box_details = []
    
    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = model_names.get(cls_id, str(cls_id))
        conf = float(box.conf[0])
        xyxy = [round(float(coord), 1) for coord in box.xyxy[0].tolist()]
        
        detected_classes.append(cls_name)
        item_counts[cls_name] = item_counts.get(cls_name, 0) + 1
        box_details.append({
            "class": cls_name,
            "confidence": round(conf * 100, 1),
            "bbox": xyxy
        })

    # Evaluate compliance against standard 5 PPE items
    detected_lower = [c.lower() for c in detected_classes]
    
    status_summary = {}
    violations = []
    compliant_items = []
    
    for ppe in MONITORED_PPE:
        ppe_lower = ppe.lower()
        targets = SYNONYM_MAP.get(ppe_lower, [ppe_lower])
        
        # Check explicit negative label first (e.g., "no-hardhat")
        has_negative = any(any(c == f"no-{t}" or c == f"no_{t}" for t in targets) for c in detected_lower)
        # Check positive presence
        has_positive = any(any(c == t for t in targets) for c in detected_lower)
        
        if has_negative:
            status_summary[ppe] = "VIOLATION"
            violations.append(f"NO-{ppe.upper().replace(' ', '-')}")
        elif has_positive:
            status_summary[ppe] = "DETECTED"
            compliant_items.append(ppe)
        else:
            status_summary[ppe] = "ABSENT"
            violations.append(f"MISSING-{ppe.upper().replace(' ', '-')}")

    compliance_score = round((len(compliant_items) / len(MONITORED_PPE)) * 100)
    is_compliant = (len(violations) == 0) or (len(compliant_items) >= 4 and not any(v.startswith("NO-") for v in violations))

    return {
        "box_details": box_details,
        "item_counts": item_counts,
        "status_summary": status_summary,
        "compliant_items": compliant_items,
        "violations": violations,
        "compliance_score": compliance_score,
        "overall_status": "COMPLIANT" if is_compliant else "NON_COMPLIANT",
        "person_count": item_counts.get("Person", item_counts.get("person", 1 if len(box_details) > 0 else 0))
    }

# ----------------- CAMERA STREAM CONTROLLER -----------------
class CameraStreamer:
    def __init__(self):
        self.cap = None
        self.camera_index = 0
        self.rotate_180 = False
        self.conf_threshold = 0.30
        self.is_active = False
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_status = {
            "fps": 0.0,
            "overall_status": "READY",
            "compliance_score": 0,
            "status_summary": {ppe: "UNKNOWN" for ppe in MONITORED_PPE},
            "violations": [],
            "item_counts": {},
            "active_camera": 0
        }
        self.thread = None
        self.stop_event = threading.Event()

    def start(self, camera_index=0, rotate_180=False, conf=0.30):
        with self.lock:
            self.camera_index = int(camera_index)
            self.rotate_180 = bool(rotate_180)
            self.conf_threshold = float(conf)
            
            if self.is_active:
                return True
                
            self.stop_event.clear()
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                print(f"[!] Unable to open camera {self.camera_index}")
                # Try fallback camera index 1 if 0 failed, or vice versa
                alt_idx = 1 if self.camera_index == 0 else 0
                print(f"[*] Trying alternative camera index {alt_idx}...")
                self.cap = cv2.VideoCapture(alt_idx)
                if self.cap.isOpened():
                    self.camera_index = alt_idx
                else:
                    return False
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.is_active = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            print(f"[+] Camera stream started on index {self.camera_index}")
            return True

    def update_settings(self, rotate_180=None, conf=None, camera_index=None):
        with self.lock:
            if rotate_180 is not None:
                self.rotate_180 = bool(rotate_180)
            if conf is not None:
                self.conf_threshold = float(conf)
            if camera_index is not None and int(camera_index) != self.camera_index:
                new_idx = int(camera_index)
                if self.is_active:
                    self.stop()
                    self.start(camera_index=new_idx, rotate_180=self.rotate_180, conf=self.conf_threshold)

    def stop(self):
        with self.lock:
            if not self.is_active:
                return
            self.stop_event.set()
            self.is_active = False
            if self.cap:
                self.cap.release()
                self.cap = None
            print("[-] Camera stream stopped.")

    def _capture_loop(self):
        m = get_ppe_model()
        prev_time = time.time()
        
        while not self.stop_event.is_set():
            if not self.cap or not self.cap.isOpened():
                time.sleep(0.1)
                continue
                
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue
                
            if self.rotate_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-5)
            prev_time = now
            
            # Run inference
            results = m.predict(source=frame, conf=self.conf_threshold, iou=0.45, verbose=False)
            res = results[0]
            annotated_frame = res.plot()
            
            # Analyze
            eval_data = analyze_detections(res.boxes, m.names)
            
            # Draw industrial overlay directly onto frame (matching camera_1.py style)
            cv2.putText(annotated_frame, f"SERBOT LIVE | FPS: {fps:.1f}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            
            y_offset = 65
            for ppe in MONITORED_PPE:
                st = eval_data["status_summary"].get(ppe, "ABSENT")
                color = (0, 220, 0) if st == "DETECTED" else (0, 0, 240)
                cv2.putText(annotated_frame, f"{ppe.upper()}: {st}", (15, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                y_offset += 24
                
            # Encode frame
            _, buffer = cv2.imencode(".jpg", annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            self.latest_frame = buffer.tobytes()
            
            self.latest_status = {
                "fps": round(fps, 1),
                "overall_status": eval_data["overall_status"],
                "compliance_score": eval_data["compliance_score"],
                "status_summary": eval_data["status_summary"],
                "violations": eval_data["violations"],
                "item_counts": eval_data["item_counts"],
                "box_count": len(eval_data["box_details"]),
                "active_camera": self.camera_index,
                "rotate_180": self.rotate_180,
                "conf": self.conf_threshold
            }
            
            # Prevent 100% CPU lock
            time.sleep(0.01)

camera_streamer = CameraStreamer()

# ----------------- FLASK ROUTES -----------------
@app.route("/")
def index():
    return render_template("index.html", model_info=model_info)

@app.route("/api/system_info")
def system_info():
    return jsonify({
        "status": "online",
        "model": model_info,
        "monitored_ppe": MONITORED_PPE,
        "camera_active": camera_streamer.is_active,
        "active_camera_index": camera_streamer.camera_index
    })

@app.route("/api/detect", methods=["POST"])
def detect_image():
    conf = float(request.form.get("confidence", 0.30))
    sample_name = request.form.get("sample_name", None)
    
    img_bgr = None
    filename = "uploaded_image.jpg"
    
    if sample_name:
        sample_path = TEST_IMAGES_DIR / sample_name
        if sample_path.exists():
            img_bgr = cv2.imread(str(sample_path))
            filename = sample_name
    elif "image" in request.files:
        file = request.files["image"]
        filename = file.filename
        file_bytes = file.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return jsonify({"error": "No valid image provided or file could not be decoded"}), 400

    m = get_ppe_model()
    t_start = time.time()
    results = m.predict(source=img_bgr, conf=conf, iou=0.45, verbose=False)
    infer_time_ms = round((time.time() - t_start) * 1000, 1)
    
    res = results[0]
    annotated = res.plot()
    
    # Analyze detections
    eval_data = analyze_detections(res.boxes, m.names)
    eval_data["inference_time_ms"] = infer_time_ms
    eval_data["filename"] = filename
    eval_data["image_dimensions"] = [img_bgr.shape[1], img_bgr.shape[0]] # [width, height]
    
    # Generate clean Edge-to-Cloud IoT telemetry payload (from handover.md)
    telemetry_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "robot_id": "SERBOT_DIFF_01",
        "location_zone": "Floor Patrol B - Sector 4",
        "overall_status": eval_data["overall_status"],
        "compliance_score": f"{eval_data['compliance_score']}%",
        "detected_summary": eval_data["item_counts"],
        "violations_detected": eval_data["violations"],
        "person_count": eval_data["person_count"],
        "telemetry_channel": "HTTP_REST_INSPECTION"
    }
    eval_data["telemetry_payload"] = telemetry_payload

    # Base64 encode images
    _, buffer_ann = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    _, buffer_orig = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    
    eval_data["annotated_image"] = "data:image/jpeg;base64," + base64.b64encode(buffer_ann).decode("utf-8")
    eval_data["original_image"] = "data:image/jpeg;base64," + base64.b64encode(buffer_orig).decode("utf-8")

    return jsonify(eval_data)

@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            if camera_streamer.latest_frame is not None:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + camera_streamer.latest_frame + b"\r\n")
            time.sleep(0.04)
            
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/camera/control", methods=["POST"])
def camera_control():
    data = request.json or {}
    action = data.get("action", "")
    
    if action == "start":
        cam_idx = int(data.get("camera_index", 0))
        rot = bool(data.get("rotate_180", False))
        conf = float(data.get("confidence", 0.30))
        ok = camera_streamer.start(camera_index=cam_idx, rotate_180=rot, conf=conf)
        return jsonify({"status": "running" if ok else "failed", "active_camera": camera_streamer.camera_index})
        
    elif action == "stop":
        camera_streamer.stop()
        return jsonify({"status": "stopped"})
        
    elif action == "update":
        camera_streamer.update_settings(
            rotate_180=data.get("rotate_180"),
            conf=data.get("confidence"),
            camera_index=data.get("camera_index")
        )
        return jsonify({"status": "updated", "camera_status": camera_streamer.latest_status})
        
    return jsonify({"error": "Unknown action"}), 400

@app.route("/api/camera/status")
def camera_status():
    return jsonify({
        "is_active": camera_streamer.is_active,
        **camera_streamer.latest_status
    })

@app.route("/api/gallery")
def get_gallery():
    test_files = []
    if TEST_IMAGES_DIR.exists():
        test_files = [f.name for f in TEST_IMAGES_DIR.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]
        
    output_files = []
    if OUTPUT_IMAGES_DIR.exists():
        output_files = [f.name for f in OUTPUT_IMAGES_DIR.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]
        
    items = []
    for tf in test_files:
        items.append({
            "name": tf,
            "has_output": (tf in output_files),
            "test_url": f"/images/test/{tf}",
            "output_url": f"/images/output/{tf}" if (tf in output_files) else None
        })
        
    return jsonify({"gallery": items, "total": len(items)})

@app.route("/images/test/<path:filename>")
def serve_test_image(filename):
    return send_from_directory(str(TEST_IMAGES_DIR), filename)

@app.route("/images/output/<path:filename>")
def serve_output_image(filename):
    return send_from_directory(str(OUTPUT_IMAGES_DIR), filename)

if __name__ == "__main__":
    # Create required folders if missing
    TEST_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "templates").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "static" / "css").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "static" / "js").mkdir(parents=True, exist_ok=True)

    print("\n=======================================================")
    print("   SERBOT PPE COMPLIANCE PLATFORM SERVER READY")
    print("   Open in browser: http://localhost:5000")
    print("=======================================================\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
