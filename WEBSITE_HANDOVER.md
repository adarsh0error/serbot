# SerBot PPE Platform — Web Platform Handover & LLM Context Document

> **For LLMs reading this file:** This document is a complete technical handover detailing the web application, backend architecture, and design decisions implemented for the **SerBot PPE Safety Detection System**. Use this context to understand what was built, how the components integrate, and how to continue development or answer questions about this project.

---

## 1. Project Context & Objectives
- **Project Name:** SerBot (Personal Protective Equipment Compliance Monitoring Robot)
- **Robot Hardware:** 2-wheel differential drive chassis (wooden base, front castor), DC geared motors with Cytron DC driver, ESP32 microcontroller with dual-switch isolated power circuits, and a 50 cm vertical mast with a 45° downward-angled camera mount (mounted upside down to route USB cables cleanly).
- **Core ML Model:** YOLOv8s fine-tuned on 80,000+ industrial images (`ayushgupta7777/safetyvision-yolov8`, file: `v2/best.pt`). Tracked classes include Hardhat, Safety Vest, Goggles, Gloves, Mask, plus negative violation labels.
- **The Challenge:** Transition from terminal/OpenCV scripts (`camera_1.py` and `images.py`) to a unified, authentic industrial engineering web application that avoids generic "neon cyberpunk AI" tropes and provides genuine utility for live presentations and factory audits.

---

## 2. Key Accomplishments & Features Built

We built a single cohesive web application that unifies four core capabilities:

### Workspace 1: Project Overview & Mechanical Architecture
- **Purpose:** Engineering portfolio view explaining the robot's physical design and perception pipeline.
- **Components:**
  - Interactive SVG CAD schematic showing the robot's side profile, 50 cm mast, 45° optical sightline, ground clearance, and electronics layout.
  - Subsystem cards detailing drivetrain kinematics (differential drive zero-radius turning), power safety isolation (preventing motor back-EMF spikes from resetting ESP32), and sensor mast ergonomics.
  - 4-stage perception flow diagram (Optical Capture $\rightarrow$ 180° Flip $\rightarrow$ YOLOv8s Inference $\rightarrow$ Synonym & Violation Resolution $\rightarrow$ IoT Telemetry).

### Workspace 2: Live Robot Camera Feed (`camera_1.py` in Browser)
- **Purpose:** Stream what the robot sees in real time with live compliance analytics.
- **Controls & Features:**
  - **Camera Device Switcher:** Dropdown to switch between Camera 0 (internal webcam) and Camera 1 (external USB robot camera) on the fly.
  - **180° Inversion Toggle:** Real-time software rotation (`cv2.ROTATE_180`) to accommodate upside-down mast mounting.
  - **Confidence Slider:** Adjust detection threshold dynamically (0.10 to 0.90, default 0.30).
  - **Live HUD & Status Panel:** Real-time FPS counter, color-coded status badges for all 5 PPE items (Hardhat, Vest, Goggles, Gloves, Mask: Green `DETECTED` / Red `ABSENT`), overall compliance percentage, and a rolling violation log with timestamps.

### Workspace 3: Interactive Inference Lab (`images.py` in Browser)
- **Purpose:** Test and inspect static images with full computer vision analytics.
- **Controls & Features:**
  - **Drag & Drop Zone:** Drag any custom image file (JPG, PNG, WEBP) from the computer.
  - **1-Click Test Dataset Samples:** Preloaded pills for all 6 benchmark images in `test_images/`.
  - **Side-by-Side Comparison:** Raw input image vs annotated output with YOLO bounding boxes and latency in milliseconds.
  - **Audit Checklist:** 5-class PPE status cards showing detection status.
  - **Bounding Boxes Table:** Class name, confidence %, and exact pixel coordinates `[x1, y1, x2, y2]`.
  - **Edge-to-Cloud IoT JSON Telemetry:** Auto-generates a standardized REST JSON payload formatted for cloud ingestion (`timestamp`, `robot_id`, `detected_summary`, `violations_detected`).

### Workspace 4: Dataset & Historical Evaluation Gallery
- **Purpose:** Evidence gallery displaying benchmark results.
- **Controls & Features:**
  - Visual grid of the test images alongside their YOLO-annotated outputs from `output_images/`.
  - **"Inspect in Lab" Button:** Loads any gallery image directly into Workspace 3 with 1 click.

---

## 3. Hybrid Architecture (Local Server vs GitHub Pages)

The platform is engineered to run in **two modes** without code changes:

```
                          ┌─────────────────────────────┐
                          │     Web Application UI      │
                          │        (index.html)         │
                          └──────────────┬──────────────┘
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   ▼                                           ▼
       [ LOCAL PYTHON SERVER ]                    [ GITHUB PAGES HOSTING ]
          (python app.py)                         (Static / Client-Side)
  • Flask REST API (/api/detect)             • Static GitHub Pages CDN
  • In-memory YOLOv8s PyTorch model          • WebRTC (getUserMedia) for live cam
  • OpenCV MJPEG stream (/video_feed)        • Precomputed dataset (dataset.js)
  • Physical device control (Cam 0 / 1)      • Client-side inspector & gallery
```

1. **Local Mode (Full Stack):** Run via `python app.py` or double-clicking `run_dashboard.bat`. Flask loads the PyTorch YOLOv8s model once in memory for fast inference, uses OpenCV for multi-threaded camera capture, and streams MJPEG frames to the browser.
2. **GitHub Pages Mode (Standalone Client):** When accessed via `https://<user>.github.io/serbot/`, the site uses HTML5 WebRTC (`navigator.mediaDevices.getUserMedia`) to capture the user's camera directly in the browser, loads precomputed real-world inference results (`static/js/dataset.js`) for the test images, and renders all CAD schematics and galleries statically.

---

## 4. Codebase File Manifest

```
serbot/
│
├── app.py                      # Flask server: model caching, OpenCV MJPEG streamer, REST endpoints
├── run_dashboard.bat           # 1-Click Windows execution launcher for local presentation
├── index.html                  # Root dashboard entry point (used by GitHub Pages & local server)
├── generate_static_data.py     # Utility script that generates precomputed YOLO data for GitHub Pages
├── WEBSITE_HANDOVER.md         # This technical handover & LLM context file
├── README.md                   # Public repository documentation
│
├── static/
│   ├── css/
│   │   └── dashboard.css       # Industrial design system (slate/titanium, safety amber/emerald)
│   └── js/
│       ├── dashboard.js        # Main frontend controller (tabs, camera, upload, gallery, hybrid detection)
│       └── dataset.js          # Precomputed inference dataset for static GitHub Pages mode
│
├── templates/
│   └── index.html              # Jinja2 template mirror for Flask server
│
├── test_images/                # Benchmark input test images (6 images)
├── output_images/              # Ground-truth annotated YOLOv8 output images (6 images)
│
├── camera_1.py                 # Original standalone live camera script (OpenCV + YOLOv8)
├── images.py                   # Original standalone batch image testing script
├── boundingboxes.py            # Utility script for bounding box calculation
├── rotate.py                   # Camera rotation utility script
└── .gitignore                  # Ignores large .pt model weights (>50MB) for GitHub compliance
```

---

## 5. Key API Endpoints in `app.py`

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the main industrial dashboard. |
| `GET` | `/api/system_info` | Returns backend online status, model metadata, and tracked PPE classes. |
| `POST` | `/api/detect` | Accepts image file or sample name + confidence; runs YOLOv8s; returns base64 annotated image, bounding boxes table, compliance score, and IoT JSON telemetry. |
| `GET` | `/video_feed` | Multi-part MJPEG stream from OpenCV video capture with YOLO bounding box overlay. |
| `POST` | `/api/camera/control` | Actions: `start`, `stop`, `update`. Dynamically updates camera index (0/1), 180° rotation, and confidence threshold. |
| `GET` | `/api/camera/status` | Polled by client (every 350ms) to retrieve live FPS, compliance score %, and PPE checklist status. |
| `GET` | `/api/gallery` | Returns array of all available test images and matching annotated outputs. |

---

## 6. How to Run & Deploy

### Running Locally (Full Server):
```bash
# Option 1: Double-click run_dashboard.bat
# Option 2: Command line
cd "C:\Users\adars\OneDrive\Desktop\serbot"
python app.py
# Open http://localhost:5000 in your browser
```

### Pushing to GitHub & Enabling GitHub Pages:
```bash
cd "C:\Users\adars\OneDrive\Desktop\serbot"
git remote add origin https://github.com/adarsh0error/serbot.git
git push -u origin main
```
Then on GitHub: **Settings $\rightarrow$ Pages $\rightarrow$ Deploy from a branch (`main` / `root`) $\rightarrow$ Save**.  
The live site will be accessible at: `https://adarsh0error.github.io/serbot/`.
