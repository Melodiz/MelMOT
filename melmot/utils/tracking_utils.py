"""
Tracking Utilities

This module provides utility classes and data structures for tracking,
including Detection, Tracklet, and related helper functions.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import json


@dataclass
class Detection:
    """
    Represents a single object detection.
    
    Attributes:
        bbox: Bounding box coordinates [x1, y1, x2, y2]
        confidence: Detection confidence score
        class_id: Class ID of the detected object
        features: Optional appearance features for Re-ID
    """
    bbox: List[float]
    confidence: float
    class_id: int
    features: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Validate detection data."""
        if len(self.bbox) != 4:
            raise ValueError("Bbox must have exactly 4 coordinates [x1, y1, x2, y2]")
        
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get the center point of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def width(self) -> float:
        """Get the width of the bounding box."""
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> float:
        """Get the height of the bounding box."""
        return self.bbox[3] - self.bbox[1]
    
    @property
    def area(self) -> float:
        """Get the area of the bounding box."""
        return self.width * self.height
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert detection to dictionary."""
        return {
            'bbox': self.bbox,
            'confidence': self.confidence,
            'class_id': self.class_id,
            'features': self.features.tolist() if self.features is not None else None
        }


@dataclass
class Tracklet:
    """
    Represents a tracklet (trajectory) of an object across multiple frames.
    
    Attributes:
        track_id: Unique identifier for the track
        detections: List of detections for this track
        start_frame: First frame where this track appears
        end_frame: Last frame where this track appears
        total_frames: Total number of frames this track appears in
    """
    track_id: int
    detections: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize computed properties."""
        self._update_computed_properties()
    
    def add_detection(
        self,
        frame_idx: int,
        bbox: List[float],
        confidence: float,
        features: Optional[np.ndarray] = None
    ):
        """
        Add a detection to this tracklet.
        
        Args:
            frame_idx: Frame index
            bbox: Bounding box coordinates
            confidence: Detection confidence
            features: Optional appearance features
        """
        detection_data = {
            'frame': frame_idx,
            'bbox': bbox,
            'confidence': confidence,
            'features': features.tolist() if features is not None else None
        }
        
        self.detections.append(detection_data)
        self._update_computed_properties()
    
    def _update_computed_properties(self):
        """Update computed properties based on current detections."""
        if not self.detections:
            self.start_frame = None
            self.end_frame = None
            self.total_frames = 0
            return
        
        frames = [det['frame'] for det in self.detections]
        self.start_frame = min(frames)
        self.end_frame = max(frames)
        self.total_frames = len(self.detections)
    
    def get_detection_at_frame(self, frame_idx: int) -> Optional[Dict[str, Any]]:
        """
        Get detection data for a specific frame.
        
        Args:
            frame_idx: Frame index to search for
            
        Returns:
            Detection data if found, None otherwise
        """
        for detection in self.detections:
            if detection['frame'] == frame_idx:
                return detection
        return None
    
    def get_bbox_at_frame(self, frame_idx: int) -> Optional[List[float]]:
        """
        Get bounding box for a specific frame.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            Bounding box coordinates if found, None otherwise
        """
        detection = self.get_detection_at_frame(frame_idx)
        return detection['bbox'] if detection else None
    
    def get_features_at_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Get appearance features for a specific frame.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            Appearance features if found, None otherwise
        """
        detection = self.get_detection_at_frame(frame_idx)
        if detection and detection['features']:
            return np.array(detection['features'])
        return None
    
    def get_average_features(self) -> Optional[np.ndarray]:
        """
        Get average appearance features across all frames.
        
        Returns:
            Average features if available, None otherwise
        """
        features_list = []
        for detection in self.detections:
            if detection['features']:
                features_list.append(detection['features'])
        
        if not features_list:
            return None
        
        return np.mean(features_list, axis=0)
    
    def get_movement_statistics(self) -> Dict[str, float]:
        """
        Calculate movement statistics for this tracklet.
        
        Returns:
            Dictionary containing movement statistics
        """
        if len(self.detections) < 2:
            return {
                'total_movement': 0.0,
                'average_movement': 0.0,
                'max_movement': 0.0
            }
        
        movements = []
        for i in range(1, len(self.detections)):
            prev_bbox = self.detections[i-1]['bbox']
            curr_bbox = self.detections[i]['bbox']
            
            # Calculate center point movement
            prev_center = ((prev_bbox[0] + prev_bbox[2]) / 2, (prev_bbox[1] + prev_bbox[3]) / 2)
            curr_center = ((curr_bbox[0] + curr_bbox[2]) / 2, (curr_bbox[1] + curr_bbox[3]) / 2)
            
            movement = np.sqrt((curr_center[0] - prev_center[0])**2 + (curr_center[1] - prev_center[1])**2)
            movements.append(movement)
        
        return {
            'total_movement': sum(movements),
            'average_movement': np.mean(movements),
            'max_movement': max(movements)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tracklet to dictionary."""
        return {
            'track_id': self.track_id,
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'total_frames': self.total_frames,
            'detections': self.detections,
            'movement_stats': self.get_movement_statistics()
        }
    
    def __len__(self) -> int:
        """Return the number of detections in this tracklet."""
        return len(self.detections)


def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: First bounding box [x1, y1, x2, y2]
        bbox2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        IoU score between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def calculate_distance(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate Euclidean distance between centers of two bounding boxes.
    
    Args:
        bbox1: First bounding box [x1, y1, x2, y2]
        bbox2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        Euclidean distance between centers
    """
    center1 = Detection(bbox1, 0.0, 0).center
    center2 = Detection(bbox2, 0.0, 0).center
    
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
