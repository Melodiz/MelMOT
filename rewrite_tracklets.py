from utils.remap_ids_tracklets import create_remap
import json
import numpy as np

def load_tracklets(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['tracklets']

def merge_traks_info(info1, info2):
    # Combine both track info
    all_detections = info1 + info2
    
    # Group detections by frame
    frame_detections = {}
    for detection in all_detections:
        frame = detection['frame']
        if frame not in frame_detections:
            frame_detections[frame] = detection
        else:
            # If we already have a detection for this frame, keep the one with higher confidence
            if detection['confidence'] > frame_detections[frame]['confidence']:
                frame_detections[frame] = detection
    
    # Convert back to list and sort by frame
    merged_tracklets = list(frame_detections.values())
    merged_tracklets.sort(key=lambda x: x['frame'])
    
    return merged_tracklets

def remap_tracklets(tracklets, id_remap):
    remapped_tracklets = {}
    for original_id, track_info in tracklets.items():
        reid = id_remap.get(int(original_id), original_id)
        if reid not in remapped_tracklets:
            remapped_tracklets[reid] = track_info
        else:
            remapped_tracklets[reid] = merge_traks_info(remapped_tracklets[reid], track_info)
    return remapped_tracklets

def rem_small_detections(tracklets_data, min_frames=10):
    # Filter out tracks with less than min_frames detections
    filtered_tracklets = {
        reid: track_info for reid, track_info in tracklets_data.items()
        if len(track_info) >= min_frames
    }
    return filtered_tracklets

def save_remapped_tracklets(remapped_tracklets, output_path):
    with open(output_path, 'w') as f:
        json.dump({'tracklets': remapped_tracklets}, f, indent=2)
    print(f"Remapped tracklets saved to {output_path}")

def rem_zero_confidence(tracklets_data, threshold=0.95):
    percet_of_zero_conf = {
        reid: (len([d for d in track_info if d['confidence'] == 0]) / len(track_info))
        for reid, track_info in tracklets_data.items()
    }
    remapped_tracklets = {
        reid: track_info for reid, track_info in tracklets_data.items()
        if percet_of_zero_conf[reid] < threshold
    }
    return remapped_tracklets

def rem_zero_conf_ends(tracklets_data):
    cleared_tracklets = tracklets_data.copy()
    rem_count = 0
    total_frames = sum(len(track_info) for track_info in tracklets_data.values())
    for id, track_info in tracklets_data.items():
        last_frame_with_conf=max(d['frame'] for d in track_info if d['confidence'] > 0)
        # Remove detections after last_frame_with_conf
        cleared_tracklets[id] = [d for d in track_info if d['frame'] <= last_frame_with_conf]
        cleared_tracklets[id].sort(key=lambda x: x['frame'])
        rem_count += len(track_info) - len(cleared_tracklets[id])
    print(f"{rem_count/total_frames*100:.2f}% of detections removed due to zero confidence or ends after last detection with confidence")
    return cleared_tracklets, rem_count

    
def improve_tracklets(tracklets_path, clusters_file,output_path,  min_frames=10):
    id_remap = create_remap(tracklets_path, clusters_file)
    tracklets = load_tracklets(tracklets_path)
    remapped_tracklets = remap_tracklets(tracklets, id_remap)
    remapped_tracklets = rem_small_detections(remapped_tracklets, min_frames)
    save_remapped_tracklets(remapped_tracklets, output_path)

def bbox_distance(bbox1, bbox2):
    if bbox1 is None or bbox2 is None:
        return 0
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (w1 - w2) ** 2 + (h1 - h2) ** 2)
def remove_manikin(tracklets_data, movement_threshold=0.5):
    """
    Remove tracks with minimal movement (likely mannequins or stationary objects).
    Movement is normalized by the average bounding box size to account for perspective.
    
    Args:
        tracklets_data (dict): Dictionary of track data keyed by track ID
        movement_threshold (float): Threshold for minimum movement as a ratio of average bbox size
        
    Returns:
        dict: Filtered tracklets with stationary objects removed
    """
    filtered_tracklets = {}
    removed_count = 0
    
    for track_id, track_info in tracklets_data.items():
        if not track_info:  # Skip empty tracks
            continue
            
        # Extract all x and y coordinates and calculate bbox sizes
        x_coords = []
        y_coords = []
        bbox_sizes = []
        
        for detection in track_info:
            bbox = detection['bbox']
            # Calculate center point of bbox
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            x_coords.append(center_x)
            y_coords.append(center_y)
            
            # Calculate bbox size (average of width and height)
            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]
            bbox_size = (bbox_width + bbox_height) / 2
            bbox_sizes.append(bbox_size)
        
        # Calculate average bbox size for normalization
        avg_bbox_size = sum(bbox_sizes) / len(bbox_sizes) if bbox_sizes else 1
        
        # Calculate maximum movement in x and y directions
        max_x_change = max(x_coords) - min(x_coords) if x_coords else 0
        max_y_change = max(y_coords) - min(y_coords) if y_coords else 0
        
        # Normalize movement by average bbox size
        normalized_x_change = max_x_change / avg_bbox_size
        normalized_y_change = max_y_change / avg_bbox_size
        
        # If both normalized x and y movement are below threshold, consider it a mannequin/stationary object
        if normalized_x_change < movement_threshold and normalized_y_change < movement_threshold:
            removed_count += 1
            continue
        
        # Keep this track if it has sufficient movement
        filtered_tracklets[track_id] = track_info
    
    print(f"Removed {removed_count} tracks as potential mannequins/stationary objects")
    return filtered_tracklets

if __name__ == "__main__":
    video_path = 'videos/2.mp4'
    tracklets_path = 'results/row_tracks/2_row_tracks2.json'
    # clusters_path = 'simple_clusters.json'
    output_path = 'results/2_clear2.json'
    # improve_tracklets(tracklets_path, clusters_path, output_path, min_frames=10)
    tracklets = load_tracklets(tracklets_path)
    print(f'Initial unique tracklets: {len(tracklets.keys())}')

    tracklets, removed_frames = rem_zero_conf_ends(tracklets)
    print(f'removed {removed_frames} frames with zero confidence detections', len(tracklets.keys()))

    tracklets = rem_small_detections(tracklets, 15)
    print(f'After removing small detections: {len(tracklets.keys())}')

    tracklets = rem_zero_confidence(tracklets, 0.7)
    print(f'After removing zero confidence detections: {len(tracklets.keys())}')

    tracklets = remove_manikin(tracklets, movement_threshold=0.3)  
    print(f'After removing stationary objects: {len(tracklets.keys())}')

    save_remapped_tracklets(tracklets, output_path)

