import json
import cv2
import numpy as np
from tqdm import tqdm
from utils.video_utils import play_video

def load_tracklets(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def get_color(track_id):
    # Use a hash function to generate a consistent color for each ID
    color = hash(str(track_id))
    return ((color & 0xFF), ((color >> 8) & 0xFF), ((color >> 16) & 0xFF))

def draw_tracks(frame, tracklets, frame_idx, track_history=10):
    for track_id, track_data in tracklets.items():
        # Filter tracklets for the current frame and recent history
        recent_tracklets = [t for t in track_data if frame_idx - track_history <= t['frame'] <= frame_idx]
        
        if recent_tracklets:
            # Get a unique color for this track ID
            color = get_color(track_id)
            
            # Draw the current bounding box
            current_bbox = recent_tracklets[-1]['bbox']
            x1, y1, x2, y2 = map(int, current_bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, str(track_id), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Draw the track history
            points = [(int((t['bbox'][0] + t['bbox'][2]) / 2), 
                       int((t['bbox'][1] + t['bbox'][3]) / 2)) for t in recent_tracklets]
            for i in range(1, len(points)):
                cv2.line(frame, points[i-1], points[i], color, 2)
    
    return frame

def visualize_tracklets(video_path, tracklet_file, output_path, play_video_after=True, frame_offset=0):
    tracklets_data = load_tracklets(tracklet_file)
    tracklets = tracklets_data['tracklets']
    
    cap = cv2.VideoCapture(video_path)  
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Find the last frame with tracklets, handling empty track data
    last_tracklet_frames = [max((t['frame'] for t in track_data), default=0) 
                            for track_data in tracklets.values() if track_data]
    last_tracklet_frame = max(last_tracklet_frames) if last_tracklet_frames else 0

    for video_frame_idx in tqdm(range(total_frames), desc="Visualizing tracklets"):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Adjust the frame index for the tracklets
        tracklet_frame_idx = video_frame_idx - frame_offset
        
        if 0 <= tracklet_frame_idx <= last_tracklet_frame:
            frame_with_tracks = draw_tracks(frame, tracklets, tracklet_frame_idx)
        else:
            frame_with_tracks = frame  # No tracks to draw for this frame
        
        out.write(frame_with_tracks)
    
    cap.release()
    out.release()
    print(f"Tracklet visualization saved to {output_path}")
    print(f"Processed {video_frame_idx + 1} frames out of {total_frames} total frames")

    if play_video_after:
        print("Playing the resulting video...")
        play_video(output_path, speed=2)

if __name__ == "__main__":
    video_path = 'videos/tsum_test.mp4'
    tracklet_file = 'results/tsum_test_traks.json'
    output_path = 'results/tracklets_visualization.mp4'
    frame_offset = 0  # Adjust this value if needed
    visualize_tracklets(video_path, tracklet_file, output_path, play_video_after=True, frame_offset=frame_offset)