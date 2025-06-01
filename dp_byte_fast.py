import cv2
import time
import os
import numpy as np
import torch
from collections import defaultdict
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from ultralytics import YOLO
from utils import annotate

# Create a class to convert dictionary to object with attributes
class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

# Configuration dictionary
config = {
    'input': 'input/a.mp4',  # Your input video
    'imgsz': None,
    'model': 'yolo12m.pt',  # You can use any YOLO model you have
    'threshold': 0.5,
    'show': True,
    'cls': [0], # YOLO's person class
    # ByteTrack parameters
    'track_thresh': 0.45,  # Detection threshold for tracking
    'track_buffer': 60,   # Frames to keep tracks alive without detection
    'match_thresh': 0.75,  # Matching threshold for association
    'mot20': False,       # Whether to use MOT20 settings
    'frame_rate': 30      # Approximate FPS of your video
}

# Convert dictionary to object with attributes
tracker_args = Args(**config)

# Setup directories and device
np.random.seed(42)
OUT_DIR = 'outputs'
os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize YOLO detector
print(f"Loading YOLO model: {config['model']}")
model = YOLO(config['model'])

# Initialize ByteTracker with the object that has attributes
tracker = BYTETracker(tracker_args)

# Video processing
cap = cv2.VideoCapture(config['input'])
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_fps = int(cap.get(5))
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
save_name = os.path.basename(config['input']).split('.')[0]

# Define output video writer
out = cv2.VideoWriter(
    f"{OUT_DIR}/{save_name}_bytetrack.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'), frame_fps,
    (frame_width, frame_height)
)

frame_count = 0
total_fps = 0
all_tracklets = defaultdict(list)

print(f"Processing video: {config['input']}")
print(f"Tracking classes: {config['cls']}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()
    
    # Step 1: Detect objects with YOLO
    results = model(frame, classes=config['cls'], conf=config['threshold'], verbose=False)
    
    # Step 2: Extract detection information
    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    
    # Step 3: Format detections for ByteTrack
    # ByteTrack expects detections as a numpy array with shape (N, 5)
    # where each row is [x1, y1, x2, y2, score]
    detections = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        detections.append([x1, y1, x2, y2, scores[i]])
    
    # Convert list to numpy array
    if len(detections) > 0:
        detections = np.array(detections, dtype=np.float64)  # Explicitly use float64
        
        # Update ByteTrack with detections
        online_targets = tracker.update(
            detections, [frame_height, frame_width], [frame_height, frame_width]
            )
    else:
        online_targets = []
    
    # Step 4: Store results and annotate
    for track in online_targets:
        track_id = track.track_id
        tlwh = track.tlwh
        all_tracklets[str(track_id)].append({
            "frame": frame_count,
            "bbox": tlwh.tolist(),
            "confidence": float(track.score)
        })
    
    # Annotate frame with tracking results
    frame = annotate(online_targets, frame)
    
    # Calculate FPS
    fps = 1 / (time.time() - start_time)
    total_fps += fps
    frame_count += 1
    
    # Display FPS on frame
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )
    
    print(f"Frame {frame_count}/{frames} | FPS: {fps:.1f} | Tracks: {len(online_targets)}")
    
    # Write frame to output video
    out.write(frame)
    
    # Display frame if show is enabled
    if config['show']:
        cv2.imshow("ByteTrack", frame)
        if cv2.waitKey(1) == ord('q'):
            break

# Save tracking results to JSON
import json
# check if the output directory exists
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
with open(f"{OUT_DIR}/{save_name}_tracks.json", "w") as f:
    json.dump(all_tracklets, f, indent=2)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Tracking completed. Average FPS: {total_fps / frame_count:.1f}")
print(f"Output saved to {OUT_DIR}/{save_name}_bytetrack.mp4")