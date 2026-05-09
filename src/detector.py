from ultralytics import YOLO

class YOLOv8Detector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.target_classes = [0] 

    def detect(self, frame, conf_threshold=0.25, imgsz=1280):
        """
        Runs tracking inference on a single frame.
        We swapped .predict() / () for .track() to maintain temporal consistency.
        """
        results = self.model.track(
            frame, 
            persist=True, 
            tracker="botsort.yaml", 
            classes=self.target_classes, 
            conf=conf_threshold, 
            imgsz=imgsz, 
            verbose=False
        )
        return results[0]