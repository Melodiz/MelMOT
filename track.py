import cv2
import numpy as np
from utils.sort import Sort
from tqdm import tqdm
from metrics.movement_stats import save_movement_statistics
from utils.movement_tracker import track_movement
from utils.appearance_model import AppearanceModel
from utils.filtering import initialize_filters, apply_filters
from utils.occlusion_handler import OcclusionHandler
import json

def initialize_trackers(max_age, min_hits, iou_threshold):
    mot_tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
    appearance_model = AppearanceModel()
    occlusion_handler = OcclusionHandler(max_age, appearance_model)
    return mot_tracker, appearance_model, occlusion_handler

def analyze_movement(track_history, static_area_threshold, movement_threshold):
    ids_to_keep = set()
    for track_id, history in track_history.items():
        area_width = history['max_x'] - history['min_x']
        area_height = history['max_y'] - history['min_y']
        avg_movement = history['total_movement'] / history['frames_seen']

        if ((area_width > static_area_threshold or 
            area_height > static_area_threshold) or
            avg_movement > movement_threshold):
            ids_to_keep.add(track_id)
    return ids_to_keep

def process_frame(frame, tracks, ids_to_keep, filters, kalman_filters, appearance_model):
    frame_with_boxes = frame.copy()
    active_tracks = set()

    for track in tracks:
        track_id = int(track[4])
        if track_id in ids_to_keep:
            active_tracks.add(track_id)
            bbox = track[:4].astype(int)
            
            if track_id not in filters:
                filters[track_id], kalman_filters[track_id] = initialize_filters(track_id, bbox)
            
            filtered_bbox, filtered_center, filtered_size = apply_filters(filters[track_id], kalman_filters[track_id], bbox)
            
            appearance_model.update(frame, filtered_bbox, track_id)
            
            cv2.rectangle(frame_with_boxes, (filtered_bbox[0], filtered_bbox[1]), (filtered_bbox[2], filtered_bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame_with_boxes, str(track_id), (filtered_bbox[0], filtered_bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    return frame_with_boxes, active_tracks

def handle_occlusions(frame_with_boxes, active_tracks, tracks, ids_to_keep, occlusion_handler, kalman_filters, filters):
    frame_with_boxes = occlusion_handler.handle_occlusions(frame_with_boxes, active_tracks, kalman_filters, filters)

    for track in tracks:
        track_id = int(track[4])
        if track_id in ids_to_keep and track_id not in active_tracks and track_id not in occlusion_handler.occluded_tracks:
            occlusion_handler.add_new_occluded_track(track_id, track[:4])

    occlusion_handler.merge_nearby_tracks()

    return frame_with_boxes
def track_objects(frames, detections, max_age=100, min_hits=3, iou_threshold=0.3, 
                  static_area_threshold=0, frames_to_static=200, 
                  movement_threshold=0, start_frame=0, previous_tracklets=None):
    mot_tracker, appearance_model, occlusion_handler = initialize_trackers(max_age, min_hits, iou_threshold)
    
    filters, kalman_filters = {}, {}
    
    track_history, all_tracks = track_movement(frames, detections, mot_tracker)
    save_movement_statistics(track_history)

    ids_to_keep = analyze_movement(track_history, static_area_threshold, movement_threshold)

    processed_frames = []
    tracklets = previous_tracklets or {}

    for frame_idx, (frame, tracks) in enumerate(tqdm(zip(frames, all_tracks), total=len(frames), desc="Processing frames")):
        frame_with_boxes, active_tracks = process_frame(frame, tracks, ids_to_keep, filters, kalman_filters, appearance_model)
        frame_with_boxes = handle_occlusions(frame_with_boxes, active_tracks, tracks, ids_to_keep, occlusion_handler, kalman_filters, filters)
        processed_frames.append(frame_with_boxes)

        # Save tracklets
        for track in tracks:
            track_id = int(track[4])
            if track_id in ids_to_keep:
                bbox = track[:4].tolist()  # Convert numpy array to list
                if track_id not in tracklets:
                    tracklets[track_id] = []
                tracklets[track_id].append({"frame": start_frame + frame_idx, "bbox": bbox})

    return all_tracks, processed_frames, tracklets