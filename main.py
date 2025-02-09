import json
import argparse
from collections import defaultdict
from tqdm import tqdm
import os

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter

class Track:
    def __init__(self, detection):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)  # 7 state variables, 4 measurement variables
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0]])
        self.kf.R *= 10
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = detection.reshape(4, 1)
        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.id = None  # Will be set when the track is added to tracking

def parse_arguments():
    parser = argparse.ArgumentParser(description="YOLO object tracking with Kalman Filter")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--tracklets_path", type=str, required=True, help="Path to save the output tracklets JSON file")
    parser.add_argument("--model_index", type=str, choices=['n', 's', 'm', 'l', 'x'], required=True, help="YOLO model index (n, s, m, l, x)")
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the YOLO model with verbose set to False
    model = YOLO(f"models/yolo11{args.model_index}.pt")
    model.to(device)  # Move model to GPU if available
    model.verbose = False

    # Open the video file
    cap = cv2.VideoCapture(args.video_path)

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Store the tracklets
    tracklets = defaultdict(list)
    tracks = []

    # Loop through the video frames
    for frame_number in tqdm(range(total_frames), desc="Processing frames"):
        # Read a frame from the video
        success, frame = cap.read()

        if not success:
            break

        # Run YOLO tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, classes=[0], tracker="botsort.yaml", iou=0.45, verbose=False)

        # Get the boxes, track IDs, and confidence scores
        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu().tolist()

        # Predict step for all existing tracks
        for track in tracks:
            track.kf.predict()

        # Update step
        for box, track_id, confidence in zip(boxes, track_ids, confidences):
            # Find existing track or create new one
            track = next((t for t in tracks if t.id == track_id), None)
            if track is None:
                track = Track(box)
                track.id = track_id
                tracks.append(track)
            
            # Update Kalman filter with new measurement
            track.kf.update(box)

            # Use the Kalman filter's estimate for the tracklet
            estimate = track.kf.x[:4].flatten()
            x, y, w, h = estimate
            tracklets[track_id].append({
                'frame': frame_number,
                'bbox': [float(x - w / 2), float(y - h / 2), float(x + w / 2), float(y + h / 2)],
                'confidence': float(confidence)
            })

    # Release the video capture object
    cap.release()

    # Save tracklets to a JSON file
    print(f"Saving tracklets to JSON file: {args.tracklets_path}")

    if not os.path.exists(os.path.dirname(args.tracklets_path)):
        os.makedirs(os.path.dirname(args.tracklets_path))
    with open(args.tracklets_path, 'w') as f:
        json.dump({'video_path': args.video_path, 'fps': fps, 'tracklets': tracklets}, f)

    print(f"Processing complete. Tracklets saved to {args.tracklets_path}")

if __name__ == "__main__":
    main()