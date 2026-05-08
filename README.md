# Padel Game Analytics — Shot Classification System

**Candidate:** Tsewang Gurung (BSc Computing with Artificial Intelligence, Islington College)  
**Applying for:** AI/ML Internship (Computer Vision & R&D Track) at Layman AI  

## 🎯 Objective
A prototype computer vision pipeline designed to analyze padel gameplay footage. This system detects and tracks multiple players, extracts skeletal pose landmarks, and utilizes rule-based heuristics to classify shots (Forehand vs. Backhand), ultimately exporting the analytics to a structured dataset.

## 📦 Deliverables
* **Short Demo Video:** [Insert Google Drive Link Here]
* **Shot Analytics CSV:** [Insert Google Drive Link Here]
* **YOLOv8 Pose Model:** [Insert Google Drive Link Here]

---

## 🛠️ Tech Stack & Architecture
* **Language:** Python 3.x
* **Computer Vision:** OpenCV (Video processing, frame annotation)
* **Deep Learning / AI:** Ultralytics YOLOv8 (`yolov8n-pose.pt` for joint detection)
* **Object Tracking:** BoT-SORT (Built-in Ultralytics tracker for temporal consistency)
* **Data Manipulation:** Pandas (For structuring and exporting analytics)

---

## 🧠 Methodology & Engineering Decisions

As requested, this project prioritizes a practical, working prototype and clear problem-solving over building a heavily resource-intensive, perfect custom model from scratch.

### 1. The Pivot: Object Detection to Pose Estimation
Initially, I considered using standard object detection to track the ball, racket, and players. However, standard COCO-trained models struggle to identify solid padel rackets (they are trained on stringed tennis rackets), and the fast-moving ball suffers from motion blur on the Nano model. 

**Solution:** I pivoted to **YOLOv8 Pose**. By tracking the player's skeletal structure (specifically wrists, elbows, and shoulders), we can determine the type of shot they are hitting without needing to perfectly track the physical racket or the ball. This is a much more robust approach for a 5-day prototype.

### 2. Resolving Flickering with BoT-SORT
Processing standard frames independently caused distant players to "flicker" out of detection. 
**Solution:** I integrated the `botsort.yaml` tracker into the YOLO inference step. This assigns a unique ID to each player and uses Kalman filtering to maintain their state even if they temporarily blur or turn away from the camera. Furthermore, I increased the inference `imgsz` to 1280 to ensure players at the far end of the court were detected accurately.

### 3. Rule-Based Shot Classification
Instead of training a computationally expensive 3D-CNN for action recognition, I implemented a heuristic (rule-based) classifier based on the pose keypoints.
* **Is a shot occurring?** We check if the Right Wrist `y-coordinate` is higher than the Right Hip `y-coordinate` (indicating a raised arm).
* **Forehand:** If the Right Wrist `x-coordinate` extends outward past the Right Shoulder.
* **Backhand:** If the Right Wrist `x-coordinate` crosses the body and goes past the Left Shoulder.

---

## 🚀 Setup & Execution

### 1. Environment Setup
Clone this repository and set up a virtual environment to avoid dependency conflicts.
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt


### 2. Project Structure
padel_analytics/
├── data/
│   ├── input/
│   │   └── sample_video.mp4       <-- Place video here
│   └── output/                    <-- Video and CSV will generate here
├── src/
│   ├── __init__.py
│   ├── config.py                  <-- Adjust thresholds and paths here
│   ├── detector.py                <-- YOLOv8 Pose wrapper
│   ├── classifier.py              <-- Rule-based logic
│   └── main.py                    <-- Core pipeline script
├── requirements.txt
└── README.md

### 2. Project Structure
python -m src.main

Press q at any time to interrupt the video processing. The output video and shot_analytics.csv will automatically save to data/output/