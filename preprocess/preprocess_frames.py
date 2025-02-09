import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize YOLO model with verbose=False
yolo = YOLO('models/yolo11x.pt').to(device)
yolo.conf = 0.35  # Set confidence threshold
yolo.verbose = False  # Disable verbose output

def preprocess_frames(video_path, start_frame=0, num_frames=None):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    if num_frames is None:
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - start_frame
    
    frames = []
    detections = []
    
    for _ in tqdm(range(num_frames), desc="Preprocessing frames", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
        
        results = yolo(frame, verbose=False)  # Add verbose=False here as well
        
        frame_detections = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                if box.cls[0] == 0:  # class 0 is person
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    frame_detections.append([x1, y1, x2, y2, conf])
        
        detections.append(np.array(frame_detections))
    
    cap.release()
    return frames, detections