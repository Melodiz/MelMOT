from track import track_objects
from utils.load_detections import load_preprocessed_data
from utils.video_utils import process_and_play_video
from preprocess.preprocess_frames import preprocess_frames
import cv2
import numpy as np
from tqdm import tqdm
import json
import argparse

def sliding_window_pipeline(video_path, output_path, window_size=300, A_initial_frames=600):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    print("Processing initial frames with YOLO...")
    initial_frames, initial_detections = preprocess_frames(video_path, num_frames=A_initial_frames)

    print("Processing first window with tracker...")
    all_tracks, processed_frames, tracklets = track_objects_wrapper(initial_frames[:window_size], initial_detections[:window_size])
    
    all_tracklets = {}

    for start_frame in tqdm(range(0, total_frames, window_size), desc="Processing sliding windows"):
        end_frame = min(start_frame + window_size, total_frames)
        
        # Load next batch of frames
        frames = []
        for i in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        print(f"Processing frames {start_frame} to {end_frame} with YOLO...")
        _, new_detections = preprocess_frames(video_path, start_frame=start_frame, num_frames=len(frames))
        
        print(f"Processing frames {start_frame} to {end_frame} with tracker...")
        all_tracks, processed_frames, new_tracklets = track_objects(frames, new_detections, start_frame=start_frame, previous_tracklets=all_tracklets)
        
        # Update all_tracklets with new_tracklets
        all_tracklets.update(new_tracklets)
        
        # Write processed frames to output video
        for frame in tqdm(processed_frames, desc=f"Writing processed frames {start_frame} to {end_frame}"):
            out.write(frame)

    cap.release()
    out.release()
    print("Processing complete. Output saved to:", output_path)

    # Save all tracklets to a JSON file
    tracklets_output_path = output_path.rsplit('.', 1)[0] + '_tracklets.json'
    with open(tracklets_output_path, 'w') as f:
        json.dump(all_tracklets, f)
    print("All tracklets saved to:", tracklets_output_path)


def track_objects_wrapper(frames, detections):
    all_tracks, processed_frames, tracklets = track_objects(frames, detections)
    return all_tracks, processed_frames, tracklets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run sliding window pipeline for object tracking")
    parser.add_argument("--video_path", type=str, default='videos/huge_tsum.mp4', help="Path to the input video file")
    parser.add_argument("--output_path", type=str, default='results/sliding_window_output.mp4', help="Path to save the output video file")
    
    args = parser.parse_args()
    
    sliding_window_pipeline(args.video_path, args.output_path)