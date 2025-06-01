import json
import numpy as np
import cv2
from get_coordinates import pixel_to_world_cam1, pixel_to_world_cam2

def load_tracklets(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['tracklets']

def save_tracklets(tracklets, output_file):
    with open(output_file, 'w') as f:
        json.dump({'tracklets': tracklets}, f, indent=2)
    print(f"Tracklets have been saved to {output_file}")
    return tracklets

def get_lower_bbox_points(bbox):
    left, top, right, bottom = bbox
    # Midpoint x is halfway between left and right
    # y is the bottom of the box
    return (int((left + right) // 2), int(bottom))


def add_lower_bbox_points(tracklets):
    for track_id, track_info in tracklets.items():
        for frame_data in track_info:
            bbox = frame_data['bbox']
            frame_data['lower_bbox_point'] = get_lower_bbox_points(bbox)
    print(f'Lower bounding box points have been added.')
    return tracklets

def add_timestamps(tracklets, video_name = 1):
    for track_id, track_info in tracklets.items():
        for frame_data in track_info:
            if video_name == 1:
                frame_data['timestamp'] = (frame_data['frame']-76)
            else:
                frame_data['timestamp'] = (frame_data['frame'])
    print('Adding timestamps to each track.')
    return tracklets

def add_real_coordinates(tracklets, video_name=1):
    for track_id, track_info in tracklets.items():
        for frame_data in track_info:
            if video_name == 1:
                x_pixel, y_pixel = frame_data['lower_bbox_point']
                x_world, y_world = pixel_to_world_cam1(x_pixel, y_pixel)
                frame_data['real_coordinates'] = (float(x_world), float(y_world))
            else:
                x_pixel, y_pixel = frame_data['lower_bbox_point']
                x_world, y_world = pixel_to_world_cam2(x_pixel, y_pixel)
                frame_data['real_coordinates'] = (float(x_world), float(y_world))
    print('Adding real coordinates to each track.')
    return tracklets

def add_coordinate_time_info(tracklets_path, tracklets_out, video_name=1):
    """
    Combined function to add lower bbox points, timestamps, and real-world coordinates to tracklets
    
    Args:
        tracklets_path (str): Path to input tracklets JSON file
        tracklets_out (str): Path to save the processed tracklets
        video_name (int): Video identifier (1 or 2) to determine which coordinate conversion to use
        
    Returns:
        dict: Processed tracklets with added information
    """
    print(f"Processing tracklets from {tracklets_path}")
    
    # Load tracklets
    tracklets = load_tracklets(tracklets_path)
    print(f"Loaded {len(tracklets)} tracklets")
    
    # Process each track
    for track_id, track_info in tracklets.items():
        for frame_data in track_info:
            # Add lower bbox point
            bbox = frame_data['bbox']
            frame_data['lower_bbox_point'] = get_lower_bbox_points(bbox)
            
            # Add timestamp based on video
            if video_name == 1:
                frame_data['timestamp'] = (frame_data['frame'])
            else:
                frame_data['timestamp'] = (frame_data['frame'] - 56)
            
            # Add real-world coordinates
            x_pixel, y_pixel = frame_data['lower_bbox_point']
            if video_name == 1:
                x_world, y_world = pixel_to_world_cam1(x_pixel, y_pixel)
            else:
                x_world, y_world = pixel_to_world_cam2(x_pixel, y_pixel)
            
            # Convert NumPy values to Python floats for JSON serialization
            frame_data['real_coordinates'] = (float(x_world), float(y_world))
    
    # Save the processed tracklets
    save_tracklets(tracklets, tracklets_out)
    
    print(f"Added lower bbox points, timestamps, and real coordinates to {len(tracklets)} tracklets")
    print(f"Saved processed tracklets to {tracklets_out}")
    
    return tracklets


if __name__ == "__main__":
    row_tracklets_path = 'results/2_clear2.json'
    output_tracklets_path = 'results/2_ext.json'
    video_name = 2

    add_coordinate_time_info(row_tracklets_path, output_tracklets_path, video_name)
