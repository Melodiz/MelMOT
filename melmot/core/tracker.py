"""
Single Camera Multiple Object Tracker

This module implements the core tracking functionality for single-camera MOT,
combining YOLO detection with BoT-SORT tracking and post-processing heuristics.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json
from tqdm import tqdm

from ..detection.yolo_detector import YOLODetector
from ..utils.tracking_utils import Tracklet, Detection
from ..utils.post_processing import PostProcessor


@dataclass
class TrackingConfig:
    """Configuration for the tracking system."""
    max_age: int = 100
    min_hits: int = 3
    iou_threshold: float = 0.3
    static_threshold: float = 2.0
    movement_threshold: float = 0.1
    confidence_threshold: float = 0.5


class SingleCameraTracker:
    """
    Single camera multiple object tracker using YOLO + BoT-SORT.
    
    This class implements the complete tracking pipeline including:
    - Object detection with YOLO
    - Multi-object tracking with BoT-SORT
    - Post-processing heuristics
    - Tracklet management
    """
    
    def __init__(
        self,
        model_path: str = "yolov12n.pt",
        config: Optional[TrackingConfig] = None,
        device: str = "auto"
    ):
        """
        Initialize the tracker.
        
        Args:
            model_path: Path to YOLO model weights
            config: Tracking configuration
            device: Device to run inference on ('cpu', 'cuda', or 'auto')
        """
        self.config = config or TrackingConfig()
        self.detector = YOLODetector(model_path, device=device)
        self.post_processor = PostProcessor(self.config)
        
        # Initialize tracking state
        self.tracklets: Dict[int, Tracklet] = {}
        self.frame_count = 0
        
    def track_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        save_tracklets: bool = True
    ) -> Dict[str, Any]:
        """
        Track objects in a video file.
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            save_tracklets: Whether to save tracklet data
            
        Returns:
            Dictionary containing tracking results and metadata
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize output video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        results = {
            'tracklets': {},
            'metadata': {
                'fps': fps,
                'width': width,
                'height': height,
                'total_frames': total_frames
            }
        }
        
        print(f"Processing video: {video_path}")
        print(f"FPS: {fps}, Resolution: {width}x{height}, Frames: {total_frames}")
        
        try:
            for frame_idx in tqdm(range(total_frames), desc="Tracking frames"):
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Track objects in current frame
                frame_results = self.track_frame(frame, frame_idx)
                
                # Draw results on frame
                annotated_frame = self._draw_tracking_results(frame, frame_results)
                
                # Write to output video
                if writer:
                    writer.write(annotated_frame)
                    
                # Update results
                self._update_tracklets(frame_results, frame_idx)
                
        finally:
            cap.release()
            if writer:
                writer.release()
        
        # Apply post-processing
        self._apply_post_processing()
        
        # Save tracklets
        if save_tracklets:
            results['tracklets'] = self._get_tracklet_data()
            
        return results
    
    def track_frame(
        self,
        frame: np.ndarray,
        frame_idx: int
    ) -> Dict[str, Any]:
        """
        Track objects in a single frame.
        
        Args:
            frame: Input frame as numpy array
            frame_idx: Frame index
            
        Returns:
            Dictionary containing detections and tracking results
        """
        # Detect objects
        detections = self.detector.detect(frame)
        
        # Filter detections by confidence
        detections = [
            det for det in detections 
            if det.confidence >= self.config.confidence_threshold
        ]
        
        # Update tracking
        tracking_results = self._update_tracking(detections, frame_idx)
        
        return {
            'detections': detections,
            'tracks': tracking_results,
            'frame_idx': frame_idx
        }
    
    def _update_tracking(
        self,
        detections: List[Detection],
        frame_idx: int
    ) -> List[Dict[str, Any]]:
        """
        Update tracking state with new detections.
        
        Args:
            detections: List of detections for current frame
            frame_idx: Current frame index
            
        Returns:
            List of active tracks
        """
        # Convert detections to format expected by tracker
        detection_array = np.array([
            [det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3], det.confidence]
            for det in detections
        ])
        
        # Update tracker (placeholder for BoT-SORT integration)
        # In the actual implementation, this would call the BoT-SORT tracker
        tracks = self._run_tracker(detection_array)
        
        return tracks
    
    def _run_tracker(self, detections: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run the tracking algorithm (placeholder for BoT-SORT).
        
        Args:
            detections: Detection array in format [x1, y1, x2, y2, confidence]
            
        Returns:
            List of track dictionaries
        """
        # TODO: Integrate BoT-SORT here
        # For now, return placeholder results
        tracks = []
        for i, det in enumerate(detections):
            tracks.append({
                'id': i,
                'bbox': det[:4].tolist(),
                'confidence': det[4],
                'age': 1
            })
        
        return tracks
    
    def _draw_tracking_results(
        self,
        frame: np.ndarray,
        results: Dict[str, Any]
    ) -> np.ndarray:
        """
        Draw tracking results on the frame.
        
        Args:
            frame: Input frame
            results: Tracking results for current frame
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Draw detections
        for det in results['detections']:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(
                annotated_frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )
        
        # Draw tracks
        for track in results['tracks']:
            track_id = track['id']
            bbox = track['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bounding box
            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )
            
            # Draw track ID
            cv2.putText(
                annotated_frame,
                f"ID: {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )
        
        return annotated_frame
    
    def _update_tracklets(
        self,
        frame_results: Dict[str, Any],
        frame_idx: int
    ):
        """Update tracklet data with current frame results."""
        for track in frame_results['tracks']:
            track_id = track['id']
            bbox = track['bbox']
            
            if track_id not in self.tracklets:
                self.tracklets[track_id] = Tracklet(track_id)
            
            self.tracklets[track_id].add_detection(
                frame_idx, bbox, track['confidence']
            )
    
    def _apply_post_processing(self):
        """Apply post-processing heuristics to clean up tracklets."""
        self.post_processor.process(self.tracklets)
    
    def _get_tracklet_data(self) -> Dict[str, Any]:
        """Convert tracklets to serializable format."""
        return {
            str(track_id): tracklet.to_dict()
            for track_id, tracklet in self.tracklets.items()
        }
    
    def save_tracklets(self, output_path: str):
        """Save tracklet data to JSON file."""
        tracklet_data = self._get_tracklet_data()
        
        with open(output_path, 'w') as f:
            json.dump(tracklet_data, f, indent=2)
        
        print(f"Tracklets saved to: {output_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tracking statistics."""
        total_tracks = len(self.tracklets)
        total_detections = sum(
            len(tracklet.detections) for tracklet in self.tracklets.values()
        )
        
        return {
            'total_tracks': total_tracks,
            'total_detections': total_detections,
            'average_track_length': total_detections / total_tracks if total_tracks > 0 else 0
        }
