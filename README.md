# 🚗 ADAS Vehicle Monitoring System

A real-time **Advanced Driver Assistance System (ADAS)** built using computer vision to detect, track, and analyze vehicle behavior from traffic video streams. This project simulates the **perception and decision-making layer** of autonomous driving systems.

---

## 🎯 Features

- 🔍 Vehicle Detection using YOLOv8
- 🎯 Multi-object Tracking using DeepSORT
- 📏 Speed Estimation (km/h approximation)
- 📐 Distance Estimation (approx. meters)
- ⚡ Overspeed Detection
- ⚠️ Collision Risk Warning
- 🛣️ Lane Detection
- 📊 Real-time FPS Display

---

## 🧠 How It Works

The system processes video input frame-by-frame and applies:

1. **Object Detection (YOLOv8)** → Detects vehicles in each frame
2. **Tracking (DeepSORT)** → Assigns unique IDs across frames
3. **Speed Calculation** → Based on object displacement over time
4. **Distance Estimation** → Using bounding box scaling
5. **Decision Logic** → Triggers overspeed and collision alerts
6. **Visualization** → Clean overlay UI with labels and warnings

**Pipeline:**
```bash
python main.py --source videos/sample.mp4
Video → Detection → Tracking → Speed/Distance → Alerts → Output

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| OpenCV | Video processing & visualization |
| YOLOv8 (Ultralytics) | Object detection |
| DeepSORT | Multi-object tracking |
| NumPy | Numerical computations |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ADAS-Vehicle-Monitoring-System.git
cd ADAS-Vehicle-Monitoring-System

# Install dependencies
pip install -r requirements.txt
```

### Run the Project

```bash
python main.py --source videos/sample.mp4
```

---

## 📂 Project Structure
.
├── src/
│   ├── object_detection.py   # YOLOv8 detection logic
│   ├── lane_detection.py     # Lane detection module
│   ├── utils.py              # Helper functions
│   └── config.py             # Configuration parameters
├── models/                   # YOLOv8 model weights
├── videos/                   # Input video files
├── main.py                   # Entry point
├── requirements.txt
└── README.md

---

## 📸 Sample Output

Each detected vehicle is displayed with:
ID 3 | 42 km/h | 8 m
⚠️ OVERSPEED!
⚠️ COLLISION RISK!

---

## ⚠️ Limitations

- Speed and distance values are approximate (no real-world camera calibration)
- Performance depends on camera angle and video quality
- Designed for simulation and academic/learning purposes

---

