import numpy as np
from tqdm import tqdm

def track_movement(frames, detections, mot_tracker):
    track_history = {}
    all_tracks = []

    for frame, frame_detections in tqdm(zip(frames, detections), total=len(frames), desc="Tracking movement"):
        tracks = mot_tracker.update(frame_detections)
        all_tracks.append(tracks)

        for track in tracks:
            track_id = int(track[4])
            bbox = track[:4]
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

            if track_id not in track_history:
                track_history[track_id] = {
                    'positions': [center],
                    'min_x': center[0],
                    'max_x': center[0],
                    'min_y': center[1],
                    'max_y': center[1],
                    'total_movement': 0,
                    'frames_seen': 1
                }
            else:
                history = track_history[track_id]
                history['positions'].append(center)
                history['min_x'] = min(history['min_x'], center[0])
                history['max_x'] = max(history['max_x'], center[0])
                history['min_y'] = min(history['min_y'], center[1])
                history['max_y'] = max(history['max_y'], center[1])
                history['frames_seen'] += 1

                if len(history['positions']) > 1:
                    prev_pos = history['positions'][-2]
                    history['total_movement'] += np.sqrt((center[0] - prev_pos[0])**2 + (center[1] - prev_pos[1])**2)

    return track_history, all_tracks