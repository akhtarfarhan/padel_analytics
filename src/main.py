import cv2
import pandas as pd
from src.config import INPUT_VIDEO_PATH, OUTPUT_VIDEO_PATH, OUTPUT_CSV_PATH, YOLO_MODEL_PATH, CONFIDENCE_THRESHOLD, INFERENCE_SIZE
from src.detector import YOLOv8Detector
from src.classifier import RuleBasedClassifier
from src.video_utils import create_video_writer

def main():
    print("[INFO] Initializing pipeline...")
    detector = YOLOv8Detector(YOLO_MODEL_PATH)
    classifier = RuleBasedClassifier()
    
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video at {INPUT_VIDEO_PATH}")
        return

    writer = create_video_writer(cap, OUTPUT_VIDEO_PATH)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # This list will hold all our analytics data
    analytics_data = []
    
    print("[INFO] Processing video... Press 'q' to stop early.")
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Run tracking & pose detection
        results = detector.detect(frame, conf_threshold=CONFIDENCE_THRESHOLD, imgsz=INFERENCE_SIZE)
        annotated_frame = results.plot()
        
        # 2. Extract Data for Classification
        # We must check if the tracker assigned IDs AND if pose keypoints exist
        if results.boxes.id is not None and results.keypoints is not None:
            # Convert GPU tensors to standard Python/Numpy lists
            track_ids = results.boxes.id.int().cpu().tolist()
            all_keypoints = results.keypoints.xy.cpu().numpy()
            boxes = results.boxes.xyxy.cpu().numpy()
            
            # Loop through every person detected in this specific frame
            for i in range(len(track_ids)):
                player_id = track_ids[i]
                keypoints = all_keypoints[i]
                
                # Classify the pose
                shot_type = classifier.classify_shot(keypoints)
                
                if shot_type in ["Forehand", "Backhand"]:
                    # Calculate timestamp
                    timestamp_sec = round(frame_count / fps, 2)
                    
                    # Log the data for our CSV
                    analytics_data.append({
                        "Frame": frame_count,
                        "Timestamp (s)": timestamp_sec,
                        "Player_ID": player_id,
                        "Shot_Type": shot_type
                    })
                    
                    # Draw a label above the player making the shot
                    x1, y1, x2, y2 = boxes[i]
                    cv2.putText(
                        annotated_frame, 
                        f"P{player_id}: {shot_type}", 
                        (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (0, 255, 0), # Green text
                        2
                    )
        
        # 3. Save and display
        writer.write(annotated_frame)
        cv2.imshow("Padel Analytics", cv2.resize(annotated_frame, (1280, 720)))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] User interrupted processing.")
            break
            
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"[INFO] Processed {frame_count} frames...")

    # Clean up video resources
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    
    # 4. Export the Data to CSV
    print("\n[INFO] Generating Analytics Report...")
    if len(analytics_data) > 0:
        df = pd.DataFrame(analytics_data)
        df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"[SUCCESS] Saved analytics to {OUTPUT_CSV_PATH}")
        
        # Bonus Task: Print a quick summary to the terminal!
        print("\n--- Quick Match Stats ---")
        print(df['Shot_Type'].value_counts())
    else:
        print("[WARNING] No shots were detected to save.")

if __name__ == "__main__":
    main()