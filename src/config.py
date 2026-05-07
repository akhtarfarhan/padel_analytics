import os

# Base Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_VIDEO_PATH = os.path.join(BASE_DIR, "data", "input", "sample_video.mp4")
OUTPUT_VIDEO_PATH = os.path.join(BASE_DIR, "data", "output", "processed_video.mp4")
OUTPUT_CSV_PATH = os.path.join(BASE_DIR, "data", "output", "shot_analytics.csv")

# Model Settings
YOLO_MODEL_PATH = "yolov8n-pose.pt" # Keep the pose model, it's better for our final goal
CONFIDENCE_THRESHOLD = 0.25         # Lowered from 0.5 to catch less obvious detections
INFERENCE_SIZE = 1280               # Force YOLO to process at higher resolution (Standard HD width)