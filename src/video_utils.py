import cv2

def create_video_writer(cap, output_path):
    """Creates a cv2.VideoWriter object matching the input video's properties."""
    # Extract properties from the input video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # 'mp4v' is a highly compatible codec for .mp4 files
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    return writer