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
