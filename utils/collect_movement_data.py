import json
import numpy as np
from tqdm import tqdm

def collect_movement_data(tracklet_file):
    with open(tracklet_file, 'r') as f:
        data = json.load(f)

    tracklets = data['tracklets']
    movement_data = {}

    # Define the threshold for movement (in pixels) and the frame window
    movement_threshold = 50  # Adjust this value as needed
    frame_window = 50

    for track_id, track_info in tqdm(tracklets.items(), desc="Processing tracklets"):
        track_history = {
            'min_x': float('inf'),
            'max_x': float('-inf'),
            'min_y': float('inf'),
            'max_y': float('-inf'),
            'total_movement': 0,
            'frames_seen': len(track_info),
            'total_confidence': 0,
            'stationary_frames': 0,
            'moving_frames': 0
        }

        prev_center = None
        centers = []

        for frame_data in track_info:
            bbox = frame_data['bbox']
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            centers.append(center)
            
            track_history['min_x'] = min(track_history['min_x'], center[0])
            track_history['max_x'] = max(track_history['max_x'], center[0])
            track_history['min_y'] = min(track_history['min_y'], center[1])
            track_history['max_y'] = max(track_history['max_y'], center[1])
            track_history['total_confidence'] += frame_data['confidence']

            if prev_center:
                movement = np.sqrt((center[0] - prev_center[0])**2 + (center[1] - prev_center[1])**2)
                track_history['total_movement'] += movement
            
            prev_center = center

            # Check for stationary/moving frames
            if len(centers) >= frame_window:
                start_center = centers[-frame_window]
                end_center = centers[-1]
                distance = np.sqrt((end_center[0] - start_center[0])**2 + (end_center[1] - start_center[1])**2)
                if distance < movement_threshold:
                    track_history['stationary_frames'] += 1
                else:
                    track_history['moving_frames'] += 1

        # Calculate additional metrics
        track_history['avg_movement'] = track_history['total_movement'] / (track_history['frames_seen'] - 1) if track_history['frames_seen'] > 1 else 0
        track_history['bounding_box_area'] = (track_history['max_x'] - track_history['min_x']) * (track_history['max_y'] - track_history['min_y'])
        track_history['max_x_change'] = track_history['max_x'] - track_history['min_x']
        track_history['max_y_change'] = track_history['max_y'] - track_history['min_y']
        track_history['avg_confidence'] = track_history['total_confidence'] / track_history['frames_seen']

        # Calculate stationary and moving durations
        total_analyzed_frames = track_history['stationary_frames'] + track_history['moving_frames']
        stationary_duration = track_history['stationary_frames'] / total_analyzed_frames if total_analyzed_frames > 0 else 0
        moving_duration = track_history['moving_frames'] / total_analyzed_frames if total_analyzed_frames > 0 else 0

        # Keep only the specified data
        movement_data[track_id] = {
            'frames_seen': track_history['frames_seen'],
            'total_movement': track_history['total_movement'],
            'avg_movement': track_history['avg_movement'],
            'bounding_box_area': track_history['bounding_box_area'],
            'max_x_change': track_history['max_x_change'],
            'max_y_change': track_history['max_y_change'],
            'avg_confidence': track_history['avg_confidence'],
            'stationary_duration': stationary_duration,
            'moving_duration': moving_duration
        }

    return movement_data

def main(silent=True):
    names = [3235, 32107, 32134, 32148]
    for ind in names:
        tracklet_file = f'results/{ind}_traks.json'
        movement_data = collect_movement_data(tracklet_file)

        # Save movement data to movement.json
        output_file = f'metrics/{ind}_movement.json'
        with open(output_file, 'w') as f:
            json.dump(movement_data, f, indent=2)

        print(f"Movement data has been saved to {output_file}")

        # Print summary of movement data
        if not silent:
            for track_id, data in movement_data.items():
                print(f"Track ID: {track_id}")
                print(f"  Frames seen: {data['frames_seen']}")
                print(f"  Total movement: {data['total_movement']:.2f}")
                print(f"  Average movement per frame: {data['avg_movement']:.2f}")
                print(f"  Bounding box area: {data['bounding_box_area']:.2f}")
                print(f"  Max X change: {data['max_x_change']:.2f}")
                print(f"  Max Y change: {data['max_y_change']:.2f}")
                print(f"  Average confidence: {data['avg_confidence']:.2f}")
                print(f"  Stationary duration: {data['stationary_duration']:.2f}")
                print(f"  Moving duration: {data['moving_duration']:.2f}")
                print()

if __name__ == "__main__":
    main(silent=True)