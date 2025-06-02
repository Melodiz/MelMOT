import json
import cv2
import numpy as np
from tqdm import tqdm
from utils.video_utils import play_video
from utils.anotate_video import annotate


def read_clusters(clusters_file):
    with open(clusters_file, 'r') as f:
        data = json.load(f)
    return data['clusters']


def load_tracklets(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['tracklets']


def get_color(track_id):
    # Use a hash function to generate a consistent color for each ID
    color = hash(str(track_id))
    return ((color & 0xFF), ((color >> 8) & 0xFF), ((color >> 16) & 0xFF))


def annotate_tracklets(frame, tracklets, frame_idx):
    # banned = [4, 5, 9, 12, 13] # for 1
    banned = [2, 6, 5, 3, 10, 41, 39, 49, 35, 36] 
    # filted banned ids
    for track_id, track_data in tracklets.items():
        # if not (int(track_id) in banned): continue
        # Find detection for current frame
        current_detections = [d for d in track_data if d['frame'] == frame_idx]
        if not current_detections:
            continue
            
        # Get the detection for this frame
        detection = current_detections[0]
        bbox = detection['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        
        # Generate a unique color for this track ID
        # Use a hash function to ensure consistent colors for the same ID
        color_hash = hash(str(track_id)) % 0xFFFFFF
        color = (color_hash & 0xFF, (color_hash >> 8) & 0xFF, (color_hash >> 16) & 0xFF)
        
        # Draw bounding box with the unique color
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color=color,
            thickness=2
        )
        
        # Create label with ID
        label = f"{track_id}"
        
        # Calculate text size for better positioning
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # Draw background rectangle for text
        cv2.rectangle(
            frame,
            (x1, y1 - text_size[1] - 10),
            (x1 + text_size[0] + 10, y1),
            color,
            -1  # Fill the rectangle
        )
        
        # Draw ID text with white color for better visibility
        cv2.putText(
            frame,
            label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),  # White text
            2,
            lineType=cv2.LINE_AA
        )
        
        # Check if lower_bbox_point exists in detection, otherwise calculate it
        if 'lower_bbox_point' in detection:
            # Make sure the point is a tuple of integers
            lower_bbox_point = tuple(map(int, detection['lower_bbox_point']))
        else:
            # Calculate the lower midpoint
            lower_bbox_point = (int((x1 + x2) / 2), int(y2))
        
        # Draw point at the lower midpoint
        cv2.circle(frame, lower_bbox_point, 3, (0, 0, 255), -1)  # Draw point in red color
        
    return frame

def visualize_tracklets(video_path, tracklet_file, output_path, play_video_after=True, frame_offset=0, 
                        start_frame=0, end_frame=None):
    tracklets = load_tracklets(tracklet_file)
    # tracklets = tracklets_data['tracklets']

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # If end_frame is not specified, use the total number of frames
    if end_frame is None or end_frame > total_frames:
        end_frame = total_frames

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Find the last frame with tracklets, handling empty track data
    last_tracklet_frames = [max((t['frame'] for t in track_data), default=0)
                            for track_data in tracklets.values() if track_data]
    last_tracklet_frame = max(
        last_tracklet_frames) if last_tracklet_frames else 0

    # Skip to start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for video_frame_idx in tqdm(range(start_frame, end_frame), desc="Visualizing tracklets"):
        ret, frame = cap.read()
        if not ret:
            break

        # Adjust the frame index for the tracklets
        tracklet_frame_idx = video_frame_idx - frame_offset

        if start_frame <= video_frame_idx < end_frame and 0 <= tracklet_frame_idx <= last_tracklet_frame:
            # frame_with_tracks = draw_tracks(
            #     frame, tracklets, tracklet_frame_idx)
            frame_with_tracks = annotate_tracklets(frame, tracklets, tracklet_frame_idx)
        else:
            frame_with_tracks = frame  # No tracks to draw for this frame

        out.write(frame_with_tracks)

    cap.release()
    out.release()
    print(f"Tracklet visualization saved to {output_path}")
    print(
        f"Processed frames {start_frame} to {end_frame - 1} out of {total_frames} total frames")

    if play_video_after:
        print("Playing the resulting video...")
        play_video(output_path, speed=2)


if __name__ == "__main__":
    video_path = f'videos/1.mp4'
    tracklet_file = f'results/row_tracks/1_row_tracks2.json'
    output_path = f'row1.mp4'
    frame_offset = 0  # Adjust this value if needed
    start_frame = 0  # Start drawing boxes from this frame
    end_frame = 1200  # Stop drawing boxes at this frame (exclusive)
    visualize_tracklets(video_path, tracklet_file, output_path, play_video_after=False,
                        frame_offset=frame_offset, start_frame=start_frame, end_frame=end_frame)
