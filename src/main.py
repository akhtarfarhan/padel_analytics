"""
Main pipeline entry point.
"""

from config import YOLO_MODEL
from detector import Detector
from pose_tracker import PoseTracker
from classifier import ShotClassifier

def main():
    detector = Detector(YOLO_MODEL)
    pose_tracker = PoseTracker()
    classifier = ShotClassifier()

    print("Pipeline initialized successfully.")

if __name__ == "__main__":
    main()
