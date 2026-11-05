# SerBot — Autonomous Mobile PPE Compliance Monitoring Platform
### Technical Project Description & System Architecture Document

---

## 1. Executive Summary
**SerBot** is an autonomous mobile robotics platform designed for real-time safety compliance monitoring across industrial manufacturing plants, hazardous fabrication environments, and construction sites. By combining a physical two-wheel differential drive robot with an elevated 45° angled perception mast and edge-accelerated YOLOv8 computer vision, SerBot patrols work zones to autonomously audit worker compliance with mandatory Personal Protective Equipment (PPE).

The system continuously identifies five core protective items: **Hardhats, High-Visibility Safety Vests, Eye Protection (Goggles), Protective Gloves, and Face Masks**, flagging non-compliance events instantaneously on an industrial dashboard and logging IoT telemetry payloads.

---

## 2. Problem Statement & Industrial Relevance
Occupational Safety and Health Administration (OSHA) statistics indicate that over **60% of construction site injuries and fatalities** stem from non-compliance with required protective equipment—most commonly head impacts, eye lacerations, and high-visibility blindspot accidents. 

### Limitations of Traditional Solutions:
1. **Fixed CCTV Blind Spots:** Static overhead cameras suffer from occlusion caused by heavy machinery, structural beams, and workers turning away from fixed viewpoints.
2. **Manual Safety Audits:** Human safety inspectors can only audit a tiny fraction of total floor area at any given moment, and manual record-keeping introduces inspection latency.
3. **High Bandwidth & Cloud Latency:** Streaming raw 1080p feeds from dozens of cameras over factory Wi-Fi creates bandwidth congestion and high cloud compute costs.

### The SerBot Approach:
SerBot solves these challenges by functioning as an **autonomous edge-computing patrol unit**. It moves dynamically to where workers are operating, evaluates compliance locally at the edge in real time (20+ FPS), and transmits only lightweight JSON telemetry and violation snapshots.

---

## 3. Hardware Architecture & Mechanical Design

```
                     [ USB Camera Module ] (Mounted inverted at 45° pitch)
                              │
                      [ 50 cm Sensor Mast ] (Vertical aluminum/structural channel)
                              │
                    ┌─────────┴─────────┐
                    │                   │
  [ Master Switch ]─┤  CHASSIS & POWER  ├─[ Logic Switch (Isolated) ]
  (Main Motor Vcc)  │                   │ (ESP32 / MCU Logic)
                    └─────────┬─────────┘
                              │
               ┌──────────────┼──────────────┐
               ▼                             ▼
       [ Cytron DC Driver ]           [ ESP32 MCU ]
               │                             │
    ┌──────────┴──────────┐                  ▼
    ▼                     ▼          [ Optical Encoders ]
[ Left DC Motor ] [ Right DC Motor ]
    │                     │
 [ 2WD Drive Wheel ]   [ 2WD Drive Wheel ]
               │
      [ Front Omnidirectional Castor ] (Balance point)
```

### 3.1 Chassis & Drivetrain
- **Chassis Base:** Custom-fabricated wooden baseplate providing structural rigidity, vibration absorption, and electrical insulation for onboard components.
- **Drive System:** 2-wheel differential drive powered by high-torque DC geared motors with high-traction rubber wheels.
- **Steering Kinematics:** Differential drive enables **zero-radius turning** (spin-in-place), allowing SerBot to navigate narrow aisles and cluttered factory floors without wide turning clearance.
- **Balance Point:** Front omnidirectional castor wheel providing 3-point planar stability across uneven surfaces.
- **Motor Control:** Cytron DC Motor Driver utilizing independent Pulse-Width Modulation (PWM) channels for smooth speed ramps and bi-directional direction control.

### 3.2 Sensor Mast & Camera Elevation (Literature Survey-Derived)
- **Mast Height:** 50 cm vertical upright channel mounted to the chassis center.
- **Camera Bracket Pitch:** Rigid **45° downward tilt** mount at the mast summit.
- **Design Rationale:** 
  - Standard ground-level robot cameras encounter **near-field perspective distortion**: when approaching a standing worker (1.6m – 1.9m tall) within 1 to 2 meters, a horizontal lens only captures boots and lower trousers, cutting off the hardhat and goggles entirely.
  - Elevating the optical sensor to 50 cm with a 45° downward pitch achieves a complete **head-to-torso field of view (FOV)** at near-field working distances (1–4 meters).
  - Keeping the mast at 50 cm preserves a low center of gravity (CoG), preventing the robot from tipping during sudden acceleration or emergency stops.
- **Inverted Hardware Mounting:** The camera is physically mounted upside-down at the bracket top to allow the USB cable to route directly down inside the mast channel without sharp bend stresses. The resulting 180° frame inversion is rectified at zero cost in software via `cv2.ROTATE_180`.

### 3.3 Power Architecture & Inductive Isolation
- **Dual-Switch Safety Isolation Circuit:**
  1. *Master Motor Switch:* Routes high-current power from the battery directly to the Cytron motor driver and DC geared motors through inline protective fuses.
  2. *Isolated Logic Switch:* Dedicated isolated supply switch powering the ESP32 microcontroller and sensor electronics.
- **Why Isolation is Essential:** Inductive kickback from DC motor commutators and rapid PWM direction changes induces voltage spikes (back-EMF). Physical separation of logic power prevents these spikes from resetting the ESP32 CPU or corrupting serial packets during motor acceleration.

---

## 4. Computer Vision & Machine Learning Pipeline

### 4.1 Model Specifications
- **Primary Model:** `ayushgupta7777/safetyvision-yolov8` (`v2/best.pt`)
- **Base Architecture:** YOLOv8s (Anchor-free, Single-Stage Real-Time Object Detector)
- **Training Corpus:** Fine-tuned on **80,000+ real-world industrial compliance images** spanning varying factory lighting, worker postures, reflective vests, and safety gear brands.
- **Tracked Classes (13 Total):**
  - Positive Classes: `Hardhat`, `Safety Vest`, `Goggles`, `Gloves`, `Mask`, `Person`
  - Negative/Violation Classes: `NO-Hardhat`, `NO-Safety Vest`, `NO-Goggles`, `NO-Gloves`, `NO-Mask`, `No_Harness`, `Fall-Detected`
- **Fallback Chain:** Automatically falls back to `baskarmother/yolov8-ppe-construction` if primary HuggingFace network weights fail to resolve.

### 4.2 Key Inference Hyperparameters
- **Confidence Threshold (`conf = 0.30`):** Empirical tuning showed that 0.30 provides optimal precision-recall balance. Lower thresholds (0.20–0.25) generated false positives on circular metallic machinery gears (mistaken for hardhats), while higher thresholds (0.45+) caused missed detections on transparent safety glasses.
- **Intersection over Union (`iou = 0.45`):** Non-Maximum Suppression (NMS) overlap threshold to eliminate redundant overlapping candidate bounding boxes while preserving adjacent equipment (e.g. goggles worn directly beneath a hardhat brim).

### 4.3 Label & Synonym Normalization Engine
Different model checkpoints use slightly different vocabulary. SerBot implements a runtime synonym resolution table:
- `"hardhat"` $\rightarrow$ `["hardhat", "helmet"]`
- `"safety vest"` $\rightarrow$ `["safety vest", "vest"]`
- `"goggles"` $\rightarrow$ `["goggles", "glasses", "eye protection"]`
- `"gloves"` $\rightarrow$ `["gloves", "glove"]`
- `"mask"` $\rightarrow$ `["mask", "face mask"]`

The engine checks both **positive detection presence** and **explicit negative violation labels** (e.g., `NO-Hardhat`), ensuring that negative violation tags take precedence.

---

## 5. Web Platform & Software Architecture

The accompanying web platform is built with an industrial dark slate aesthetic, avoiding generic cyber-neon templates in favor of a clean, technical control center.

```
serbot/
├── app.py                      # Flask backend (YOLO model cache, OpenCV streamer, REST APIs)
├── run_dashboard.bat           # 1-Click Windows execution launcher
├── index.html                  # Root dashboard entry point (GitHub Pages & standalone ready)
├── description.md              # Technical project documentation
├── README.md                   # Repository guide and quickstart
├── generate_static_data.py     # Static dataset precomputation utility
├── static/
│   ├── css/
│   │   └── dashboard.css       # Slate/titanium styling, safety amber/emerald accents
│   └── js/
│       ├── dashboard.js        # Hybrid controller (local Flask + GitHub Pages WebRTC)
│       └── dataset.js          # Verified real-world inference results for offline/static mode
├── templates/
│   └── index.html              # Jinja2 dashboard template
├── test_images/                # Raw input validation benchmark images
└── output_images/              # Ground-truth annotated YOLOv8 detection results
```

### 5.1 The 4 Integrated Workspaces
1. **Overview & System Architecture:**
   - Mechanical specifications grid.
   - Interactive SVG CAD schematic displaying SerBot's dimensions, CoG, optical cone, and ground clearance.
   - 4-stage perception pipeline walkthrough.
2. **Live Robot Camera Stream:**
   - Real-time video window streaming live from connected hardware.
   - Live hardware controls: Camera Index Selector (Camera 0 / Camera 1), 180° Inversion Switch, and Confidence Slider.
   - Live HUD: Real-time FPS readout, 5-class PPE status indicators (`DETECTED` in green vs `ABSENT` in red), overall compliance score %, and rolling event log.
3. **Interactive Inference Lab:**
   - Drag & drop file upload for custom user photos.
   - 1-click test suite selector preloaded with 6 real industrial test images.
   - Side-by-side Before (raw input) vs After (annotated bounding boxes with confidence scores) comparison.
   - Bounding boxes table showing class names, confidence percentages, and pixel coordinates `[x1, y1, x2, y2]`.
   - Real-time compliance scoring and Edge-to-Cloud REST telemetry JSON generation.
4. **Dataset & Evaluation Gallery:**
   - Benchmark gallery comparing unannotated input images against ground-truth YOLOv8s output.
   - Single-click "Inspect in Lab" action to load any gallery image directly into the interactive inspector.

---

## 6. Edge-to-Cloud IoT Telemetry Schema

When SerBot identifies non-compliance, it generates a standard JSON telemetry payload formatted for transmission to a centralized plant safety dashboard:

```json
{
  "timestamp": "2026-11-05T17:40:00Z",
  "robot_id": "SERBOT_DIFF_01",
  "location_zone": "Floor Patrol B - Sector 4",
  "overall_status": "NON_COMPLIANT",
  "compliance_score": "40%",
  "detected_summary": {
    "Hardhat": 1,
    "Safety Vest": 1,
    "Person": 1
  },
  "violations_detected": [
    "MISSING-GOGGLES",
    "MISSING-GLOVES",
    "MISSING-MASK"
  ],
  "person_count": 1,
  "telemetry_channel": "HTTP_REST_INSPECTION"
}
```

---

## 7. Dual Execution Modes

| Feature | Local Python Server (`python app.py`) | GitHub Pages (`https://<user>.github.io/serbot/`) |
| :--- | :--- | :--- |
| **Backend** | Python Flask + PyTorch YOLOv8s | Pure Client-Side HTML5 / WebRTC |
| **Camera Feed** | OpenCV `cv2.VideoCapture` (Device 0/1) | HTML5 `navigator.mediaDevices.getUserMedia` |
| **Inference Lab** | Real-time PyTorch model execution on GPU/CPU | Pre-verified dataset + client-side inspection |
| **Image Upload** | Full YOLOv8s bounding box inference | Client-side preview and safety checklist |
| **Setup Required** | Run `run_dashboard.bat` | Zero setup — open link in any browser |

---

## 8. Summary of Engineering Highlights
- **Solves the Near-Field Perception Limit:** Custom 50 cm mast with 45° downward pitch allows head-to-torso worker detection without tipping the robot.
- **Electrical Safety First:** Dual power isolation protects microcontroller logic from motor back-EMF spikes.
- **High-Accuracy Edge Vision:** YOLOv8s trained on 80,000+ real industrial images with empirical confidence tuning (`conf = 0.30`).
- **Production-Ready Web Dashboard:** Unifies hardware CAD schematics, live OpenCV video feeds, interactive inference testing, and historical evaluation galleries in a single interface.
