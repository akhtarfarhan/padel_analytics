# Padel Game Analytics — Shot Classification System

**Candidate:** Tsewang Gurung (BSc Computing with Artificial Intelligence, Islington College)  
**Applying for:** AI/ML Internship (Computer Vision & R&D Track) at Layman AI  

## 🎯 Objective
A prototype computer vision pipeline designed to analyze padel gameplay footage. This system detects and tracks multiple players, extracts skeletal pose landmarks, and utilizes rule-based heuristics to classify shot types (forehand, backhand, etc.) in real-time.

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
Initially, I considered using standard object detection to track the ball, racket, and players. However, standard COCO-trained models struggle to identify solid padel rackets (they are trained on striped tennis rackets).

**Solution:** I pivoted to **YOLOv8 Pose**. By tracking the player's skeletal structure (specifically wrists, elbows, and shoulders), we can determine the type of shot they are hitting without needing explicit racket detection.

### 2. Resolving Flickering with BoT-SORT
Processing standard frames independently caused distant players to "flicker" out of detection. 
**Solution:** I integrated the `botsort.yaml` tracker into the YOLO inference step. This assigns a unique ID to each player and uses Kalman filtering to maintain their state even if they temporarily become undetected.

### 3. Rule-Based Shot Classification
Instead of training a computationally expensive 3D-CNN for action recognition, I implemented a heuristic (rule-based) classifier based on the pose keypoints.
* **Is a shot occurring?** We check if the Right Wrist `y-coordinate` is higher than the Right Hip `y-coordinate` (indicating a raised arm).
* **Forehand:** If the Right Wrist `x-coordinate` extends outward past the Right Shoulder.
* **Backhand:** If the Right Wrist `x-coordinate` crosses the body and goes past the Left Shoulder.

---

## 📋 Prerequisites

* **Python 3.8+** installed on your system
* **Git** for cloning the repository
* **4GB+ RAM** recommended for smooth video processing
* A **padel gameplay video** (MP4 format recommended)

---

## 🚀 Setup & Execution

### 1. Environment Setup

Clone this repository and set up a virtual environment to avoid dependency conflicts.

```bash
# Clone the repository
git clone https://github.com/akhtarfarhan/padel_analytics.git
cd padel_analytics

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Project Structure

```
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
├── LICENSE
└── README.md
```

### 3. Running the Pipeline

Place your padel gameplay video in `data/input/sample_video.mp4`, then run:

```bash
python -m src.main
```

**Controls:**
- Press `q` at any time to interrupt the video processing
- The output video and `shot_analytics.csv` will automatically save to `data/output/`

---

## 📊 Output Format

The generated `shot_analytics.csv` contains:
* Frame number
* Player ID
* Shot type (forehand, backhand, other)
* Confidence score
* Pose keypoint coordinates (x, y, confidence)

---

## 🔧 Configuration

Edit `src/config.py` to customize:
* **Detection thresholds** (confidence, IoU)
* **Shot classification thresholds** (wrist/shoulder coordinate thresholds)
* **Input/output paths**
* **Video processing parameters** (FPS, resolution)

---

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please fork this repository and submit a pull request with your improvements.

---

## 📧 Contact

For questions or feedback, please reach out via GitHub Issues.
