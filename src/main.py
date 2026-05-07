import cv2
from src.config import INPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH, YOLO_MODEL_PATH, CONFIDENCE_THRESHOLD
from src.detector import YOLOv8Detector
from src.video_utils import create_video_writer

def main():
    print("[INFO] Initializing pipeline...")
    detector = YOLOv8Detector(YOLO_MODEL_PATH)
    
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video at {INPUT_VIDEO_PATH}")
        return

    # Set up our video writer
    writer = create_video_writer(cap, OUTPUT_VIDEO_PATH)
    
    print("[INFO] Processing video... Press 'q' to stop early.")
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # End of video
            
        # 1. Run detection
        results = detector.detect(frame, conf_threshold=CONFIDENCE_THRESHOLD)
        
        # 2. Draw bounding boxes
        # Ultralytics provides a handy plot() method to draw the boxes for us quickly
        annotated_frame = results.plot()
        
        # 3. Save the frame to our output video
        writer.write(annotated_frame)
        
        # Optional: Show the video on screen while processing (can be slow)
        cv2.imshow("Padel Analytics - Detection", cv2.resize(annotated_frame, (1280, 720)))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] User interrupted processing.")
            break
            
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"[INFO] Processed {frame_count} frames...")

    # Clean up resources
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Processing complete! Output saved to {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    main()