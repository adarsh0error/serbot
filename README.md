# SerBot — Autonomous PPE Safety Compliance Platform

[![YOLOv8](https://img.shields.io/badge/YOLOv8-SafetyVision_v2-blue.svg)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-Computer_Vision-EE4C2C.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Real--Time_Video-green.svg)](https://opencv.org/)
[![Hardware](https://img.shields.io/badge/Hardware-Differential_Drive-orange.svg)](#hardware-architecture)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An autonomous mobile robotics platform for real-time Personal Protective Equipment (PPE) compliance monitoring on construction and factory floors. Built on a 2-wheel differential drive chassis with Cytron DC motor drivers, an elevated 45° downward-pitched camera mast, and a high-accuracy YOLOv8 vision pipeline.

---

## 🌐 Live Web Platform & Dashboard

The project includes an industrial engineering dashboard with 4 core workspaces:
1. **Overview & System Architecture:** Mechanical specifications, CAD schematic, and 4-stage inference pipeline breakdown.
2. **Live Robot Camera Stream:** Real-time video feed with on-the-fly camera index selection, 180° software inversion toggle (for upside-down mast mounting), and live PPE detection status overlay.
3. **Interactive Inference Lab:** Drag & drop image testing with confidence threshold sliders, bounding box coordinate readouts, and edge-to-cloud IoT JSON telemetry generation.
4. **Dataset & Evaluation Gallery:** Historical test batch results comparing raw input against YOLO-annotated detections.

---

## ⚙️ Hardware Architecture

- **Drivetrain:** 2-Wheel Differential Drive with DC geared motors and high-traction rubber wheels.
- **Motor Driver:** Cytron DC Motor Driver for independent PWM speed and direction control.
- **Microcontroller:** ESP32 / ESP8266 handling motor control and low-level movement logic.
- **Sensor Mast:** 50 cm vertical channel with a 45° downward-angled camera mount. Elevating the camera solves near-field deadzones, capturing full head-to-torso profile without making the platform top-heavy.
- **Power & Safety Isolation:** Dual power switches (master battery cut-off + isolated microcontroller logic switch to eliminate inductive back-EMF spikes).
- **Chassis:** Custom wooden base with front omnidirectional castor wheel.

---

## 🧠 Vision & Inference Pipeline

- **Primary Model:** `ayushgupta7777/safetyvision-yolov8` (`v2/best.pt`) fine-tuned on 80,000+ real-world industrial images.
- **Tracked Classes:** `Hardhat`, `Safety Vest`, `Goggles`, `Gloves`, `Mask`, plus explicit violation tags (`NO-Hardhat`, `NO-Safety Vest`, etc.).
- **Tuned Thresholds:** `CONF_THRESHOLD = 0.30`, `IOU_THRESHOLD = 0.45`.

---

## 🚀 Quickstart

### Option A: Local Python Server (Real-Time Flask + OpenCV)
1. Install dependencies:
   ```bash
   pip install ultralytics opencv-python huggingface_hub flask
   ```
2. Double-click `run_dashboard.bat` or run:
   ```bash
   python app.py
   ```
3. Open `http://localhost:5000` in your browser.

### Option B: GitHub Pages (Standalone Web Client)
The dashboard also runs completely in the browser via GitHub Pages, utilizing WebRTC for camera streaming and pre-verified dataset models for image evaluation.
