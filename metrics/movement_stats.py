import json
import os
from typing import Dict, Any

def save_movement_statistics(track_history: Dict[int, Dict[str, Any]], output_dir: str = 'results'):
    """
    Save movement statistics for all tracked IDs to a JSON file.

    Args:
    track_history (Dict[int, Dict[str, Any]]): Dictionary containing movement history for each track ID.
    output_dir (str): Directory to save the output file. Defaults to 'results'.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Prepare the statistics
    movement_stats = {}
    for track_id, history in track_history.items():
        area_width = history['max_x'] - history['min_x']
        area_height = history['max_y'] - history['min_y']
        avg_movement = history['total_movement'] / history['frames_seen']

        movement_stats[track_id] = {
            'total_movement': history['total_movement'],
            'avg_movement': avg_movement,
            'area_width': area_width,
            'area_height': area_height,
            'frames_seen': history['frames_seen']
        }

    # Save to file
    output_path = os.path.join(output_dir, 'movement_statistics.json')
    with open(output_path, 'w') as f:
        json.dump(movement_stats, f, indent=2)

    print(f"Movement statistics saved to {output_path}")