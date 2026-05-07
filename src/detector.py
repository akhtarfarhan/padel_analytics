from ultralytics import YOLO

class YOLOv8Detector:
    def __init__(self, model_path):
        """Initializes the YOLOv8 model."""
        # This will automatically download the model weights the first time it runs
        self.model = YOLO(model_path)
        
        # COCO Dataset class IDs:
        # 0: person
        # 32: sports ball
        # 38: tennis racket (we'll use this to detect the padel racket)
        self.target_classes = [0, 32, 38]

    def detect(self, frame, conf_threshold=0.3):
        """
        Runs inference on a single frame.
        We keep confidence a bit lower (0.3) because the ball moves fast and blurs.
        """
        # verbose=False keeps our terminal clean from YOLO's default printing
        results = self.model(frame, classes=self.target_classes, conf=conf_threshold, verbose=False)
        
        return results[0] # Return the first (and only) frame's results