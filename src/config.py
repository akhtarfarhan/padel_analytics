import os

# Base Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_VIDEO_PATH = os.path.join(BASE_DIR, "data", "input", "sample_video.mp4")
OUTPUT_VIDEO_PATH = os.path.join(BASE_DIR, "data", "output", "processed_video.mp4")
OUTPUT_CSV_PATH = os.path.join(BASE_DIR, "data", "output", "shot_analytics.csv")

# Model Settings
YOLO_MODEL_PATH = "yolov8n.pt" # We will use the Nano model for speed
CONFIDENCE_THRESHOLD = 0.5