import torch
import torchvision
import cv2
import os
import time
import numpy as np
import json
from collections import defaultdict
from ultralytics import YOLO
 
from torchvision.transforms import ToTensor
from deep_sort_realtime.deepsort_tracker import DeepSort
from utils.anotate_video import convert_detections, annotate
from utils.dataloader import save_tracklets_to_json

# Update args to use yolov11n.pt
args = {
    'input': 'videos/2.mp4',
    'imgsz': None,
    'model': 'models/yolo11l.pt',  # Use the local model file
    'threshold': 0.4,
    'embedder': 'torchreid',
    'show': True,
    'cls': [0],  
    'embedder_name': 'osnet_x1_0',
    'save_interval': 100, # save every 100 frames
    'tracklets_output_path': 'results/2_row_tracks2.json',
    'video_output_name': '2_row2.mp4',
    'video_output_dir': 'outputs/',
}
np.random.seed(42)
 
os.makedirs(args['video_output_dir'], exist_ok=True)
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COLORS = np.random.randint(0, 255, size=(90, 3))
 
print(f"Tracking: Humans")
print(f"Detector: {args['model']}")
print(f"Re-ID embedder: {args['embedder']}")

model = YOLO(args['model'], verbose=True)
print(f"Successfully loaded {args['model']} with Ultralytics YOLO")

# Initialize a SORT tracker object
tracker = DeepSort(max_age=45, embedder=args['embedder'], 
                   embedder_gpu=False, embedder_model_name=args['embedder_name'],
                   )

VIDEO_PATH = args['input']
cap = cv2.VideoCapture(VIDEO_PATH)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_fps = int(cap.get(5))
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Define codec and create VideoWriter object
out = cv2.VideoWriter(
    f"{args['video_output_dir']}/{args['video_output_name']}.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'), frame_fps,
    (frame_width, frame_height)
)

frame_count = 0  # To count total frames
total_fps = 0    # To get the final frames per second

# Dictionary to store all tracklets
all_tracklets = defaultdict(list)

while cap.isOpened():
    # Read a frame
    ret, frame = cap.read()
    if ret:
        start_time = time.time()
        # Feed frame to model and get detections
        det_start_time = time.time()

        results = model(frame, classes=args['cls'], verbose=False, iou=0.5, 
                        conf=args['threshold'])
        
        # Extract detection information
        boxes = []
        scores = []
        labels = []
        
        for r in results:
            if hasattr(r, 'boxes'):
                for box in r.boxes:
                    boxes.append(box.xyxy[0].cpu())
                    scores.append(box.conf[0].cpu())
                    labels.append(box.cls[0].cpu())
        
        if boxes:
            boxes = torch.stack(boxes)
            scores = torch.tensor(scores)
            labels = torch.tensor(labels).int()
            
            detections = {
                "boxes": boxes,
                "labels": labels + 1,  # Convert from 0-indexed to 1-indexed for COCO_91_CLASSES
                "scores": scores
            }
        else:
            detections = {"boxes": torch.tensor([]), "labels": torch.tensor([]), "scores": torch.tensor([])}
    
        det_end_time = time.time()
        det_fps = 1 / (det_end_time - det_start_time)
    
        # Convert detections to Deep SORT format
        cls_filter = [c+1 for c in args['cls']]  # Convert to 1-indexed for COCO_91_CLASSES
        detections = convert_detections(detections, args['threshold'], cls_filter)
    
        # Update tracker with detections
        track_start_time = time.time()
        tracks = tracker.update_tracks(detections, frame=frame)
        track_end_time = time.time()
        track_fps = 1 / (track_end_time - track_start_time)
 
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        # Add `fps` to `total_fps`
        total_fps += fps
        # Increment frame count
        frame_count += 1
 
        print(f"Frame {frame_count}/{frames}",
              f"Detection FPS: {det_fps:.1f},",
              f"Tracking FPS: {track_fps:.1f}, Total FPS: {fps:.1f}")
        
        # Store track information for JSON output
        for track in tracks:
            if track.is_confirmed():
                track_id = str(track.track_id)
                ltrb = track.to_ltrb()
                
                # Store track data in the format shown in the example
                track_data = {
                    "frame": frame_count,
                    "bbox": [
                        float(ltrb[0]),  # left
                        float(ltrb[1]),  # top
                        float(ltrb[2]),  # right
                        float(ltrb[3])   # bottom
                    ],
                    "confidence": float(track.get_det_conf()) if track.get_det_conf() is not None else 0.0,
                }
                
                all_tracklets[track_id].append(track_data)
        if frame_count % args['save_interval'] == 0:
            save_tracklets_to_json(all_tracklets, args['tracklets_output_path'])
            print(f"Tracklets updated and saved at {frame_count}/{frames}")
        if frame_count % 150 == 0: 
            save_tracklets_to_json(all_tracklets, f'results/checkpoints/{frame_count}_2row_tracks.json')
            print(f"Tracks updated up to {frame_count}/{frames} and saved as checkpoint")
    
        # Draw bounding boxes and labels on frame.
        if len(tracks) > 0:
            frame = annotate(
                tracks,
                frame,
            )
        out.write(frame)
        if args['show']:
            # Display or save output frame.
            cv2.imshow("Output", frame)
            # Press q to quit.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    else:
        break

save_tracklets_to_json(all_tracklets, args['tracklets_output_path'])
print(f"Tracklets saved to {args['tracklets_output_path']}")

print(f"Total frames processed: {frame_count}")
print(f"Average frames per second: {total_fps / frame_count:.1f}")
    
# Release resources
cap.release()
cv2.destroyAllWindows()