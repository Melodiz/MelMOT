import numpy as np
import cv2
import json
from utils.video_utils import plot_two_crops


def load_tracklets(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['tracklets']


def remap_loss(val):
    """
    Remap loss values logarithmically from [0, inf) to [100, 0]
    where lower loss values are remapped to higher scores (better matches).
    
    Examples:
    0.0 -> 100
    0.1 -> 60
    0.2 -> 30
    0.3 -> 20
    0.5 -> 10
    1 -> 5
    2 -> 2
    3 -> 1
    5 -> 0.1
    7+ -> 0
    """
    if val <= 0:
        return 100  # Perfect match
    elif val >= 7:
        return 0    # No match
    
    # Use an exponential decay function to map values
    # The formula is: score = a * exp(-b * val)
    # where a and b are constants chosen to fit the desired mapping
    a = 100  # Maximum score
    b = 2  # Decay rate (adjusted to fit the desired mapping)
    
    score = a * np.exp(-b * val)
    score = min(score, 100)
    return max(0, score)
    

def process_query_crop(query_track, base_tracklets, time_threshold=7, dist_threshold=0.7):
    """
    Find the best matching track from base_tracklets for the given query_track
    for each id in the base tracks
    filter out the candidates based on time abs(query_timestamp - candidate_timestamp) <= time_threshold
    within the filtered candidates, filter out the ones based on distance abs(query_bbox - candidate_bbox) <= dist_threshold
    return the top 1 match: match_score = - distance_diff - time_diff
    top 1 match for query is the one with the highest match_score

    Args:
        query_track (dict): A single detection from a query tracklet
        base_tracklets (dict): Dictionary of all base tracklets
        time_threshold (int): Maximum allowed time difference in frames
        dist_threshold (float): Maximum allowed distance difference in world coordinates

    Returns:
        tuple: (best_match_id, best_match_frame, best_match_loss) or (None, None, None) if no match found
    """
    matches_id = []
    matches_loss = []
    best_match_id = None
    best_match_frame = None
    best_match_loss = float('inf')

    query_timestamp = query_track['timestamp']
    query_coords = query_track['real_coordinates']

    # Iterate through all base tracklets
    for track_id, track_detections in base_tracklets.items():
        # Iterate through all detections in this tracklet
        for i, detection in enumerate(track_detections):
            # Calculate time difference
            time_diff = abs(detection['timestamp'] - query_timestamp)

            # Skip if time difference exceeds threshold
            if time_diff > time_threshold:
                continue
            # Calculate distance between real-world coordinates
            base_coords = detection['real_coordinates']
            dist_diff = np.sqrt((query_coords[0] - base_coords[0])**2 +
                                (query_coords[1] - base_coords[1])**2)

            # Skip if distance exceeds threshold
            if dist_diff > dist_threshold:
                continue

            # Calculate match score (less is better)
            loss = dist_diff + (time_diff)
            
            # Update best match if this score is better
            if loss < best_match_loss:
                best_match_loss = loss
                best_match_id = track_id
                best_match_frame = i
    
    # Return the best matching track detection or None if no match found
    if best_match_id is None:
        return None, None, None
    return best_match_id, best_match_frame, best_match_loss


def process_query(query_tracks, base_tracklets, time_threshold=10, dist_threshold=0.5):
    """
    Find the best matching track from base_tracklets for the given query_tracks
    using weighted voting across all detections in the query track.
    
    Args:
        query_tracks (list): List of detections in a query tracklet
        base_tracklets (dict): Dictionary of all base tracklets
        time_threshold (int): Maximum allowed time difference in frames
        dist_threshold (float): Maximum allowed distance difference in world coordinates
        
    Returns:
        tuple: (best_match_id, best_match_frame, best_match_score, best_match_query_frame_index)
               or (None, None, None, None) if no match found
    """
    # Store matches for each query detection
    found_matches = []
    
    # For each query detection, find the best match
    for i, detection in enumerate(query_tracks):
        match_id, match_frame, match_loss = process_query_crop(
            detection, base_tracklets, time_threshold, dist_threshold)
        
        if match_id is not None:
            found_matches.append((i, match_id, match_frame, match_loss))
    
    if not found_matches:
        return None, None, None, None
    
    # Group matches by track ID and calculate weighted votes
    track_votes = {}
    for query_idx, track_id, frame_idx, loss in found_matches:
        # Remap loss to a score where higher is better
        score = remap_loss(loss)
        
        if track_id not in track_votes:
            track_votes[track_id] = {
                'total_score': 0,
                'matches': []
            }
        
        # Add this match to the track's matches
        track_votes[track_id]['total_score'] += score
        track_votes[track_id]['matches'].append((query_idx, frame_idx, loss, score))
    
    # Find the track with the highest total weighted score
    if not track_votes:
        return None, None, None, None
    
    best_match_id = max(track_votes.keys(), key=lambda k: track_votes[k]['total_score'])
    
    # For the best match ID, find the detection with the lowest loss
    best_matches = track_votes[best_match_id]['matches']
    best_match = min(best_matches, key=lambda x: x[2])  # Sort by loss (lower is better)
    
    best_match_query_frame_index, best_match_frame, best_match_loss, _ = best_match
    
    # Calculate the confidence score based on voting
    total_votes = len(found_matches)
    votes_for_winner = len(track_votes[best_match_id]['matches'])
    confidence = votes_for_winner / total_votes if total_votes > 0 else 0
    
    # Print some debug information
    print(f"Found {len(found_matches)} potential matches across {len(track_votes)} different tracks")
    print(f"Best match ID {best_match_id} received {votes_for_winner}/{total_votes} votes ({confidence:.2f})")
    print(f"Best match has loss: {best_match_loss:.2f}, remapped score: {remap_loss(best_match_loss):.2f}")
    
    return best_match_id, best_match_frame, best_match_loss, best_match_query_frame_index


def visualize_all_reid(query_tracklets_path, base_tracklets_path, query_video, base_video):
    """
    Visualize all ReID matches between query and base tracklets
    
    Args:
        query_tracklets_path (str): Path to query tracklets JSON file
        base_tracklets_path (str): Path to base tracklets JSON file
        query_video (str): Path to query video file
        base_video (str): Path to base video file
    """
    # Load tracklets
    base_tracks = load_tracklets(base_tracklets_path)
    query_tracks = load_tracklets(query_tracklets_path)

    # Process each query track
    q_ids = list(query_tracks.keys())
    print(f"Processing {len(q_ids)} query tracks...")
    
    for q_id in q_ids:
        print(f"\nQuery track id: {q_id} has: {len(query_tracks[q_id])} frames")
        
        # Find best match using weighted voting
        match_id, match_frame, loss, query_frame_index = process_query(
            query_tracks[q_id], base_tracks, time_threshold=7, dist_threshold=5)
        
        if not match_id:
            print("No match found")
            continue
        
        # Calculate and display the remapped score (higher is better)
        remapped_score = remap_loss(loss)
        print(f"Best match: Track {match_id}, Frame {base_tracks[match_id][match_frame]['frame']}")
        print(f"Loss: {loss:.2f}, Remapped Score: {remapped_score:.2f}")
        
        # Get the matching frames
        query_track_match = query_tracks[q_id][query_frame_index]
        base_track_match = base_tracks[match_id][match_frame]
        
        # Visualize the match
        print(query_track_match, base_track_match, query_video,base_video, q_id, match_id)
        plot_two_crops(
            track1=query_track_match,
            track2=base_track_match,
            video_path1=query_video,
            video_path2=base_video,
            query_id=q_id,
            base_id=match_id, 
            save_silent=True
        )


if __name__ == "__main__":
    # Define paths
    base_tracklets_path = "results/2_ext.json"
    base_video = "videos/2.mp4"
    query_tracklets_path = "results/1_ext.json"
    query_video = "videos/1.mp4"
    
    # Run the visualization
    visualize_all_reid(query_tracklets_path, base_tracklets_path, query_video, base_video)