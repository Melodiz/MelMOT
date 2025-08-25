"""
Post-Processing Heuristics

This module implements post-processing heuristics for improving tracking quality,
including phantom track removal and manikin misclassification filtering.
"""

import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass

from .tracking_utils import Tracklet, calculate_iou


@dataclass
class PostProcessingConfig:
    """Configuration for post-processing heuristics."""
    static_threshold: float = 2.0
    movement_threshold: float = 0.1
    phantom_track_threshold: int = 10
    min_track_length: int = 5


class PostProcessor:
    """
    Post-processor for cleaning up tracking results.
    
    Implements heuristics for:
    1. Phantom track removal
    2. Manikin misclassification filtering
    3. Track quality assessment
    """
    
    def __init__(self, config: PostProcessingConfig):
        """
        Initialize the post-processor.
        
        Args:
            config: Post-processing configuration
        """
        self.config = config
        self.removed_tracks = []
        
    def process(self, tracklets: Dict[int, Tracklet]) -> Dict[int, Tracklet]:
        """
        Apply all post-processing heuristics to the tracklets.
        
        Args:
            tracklets: Dictionary of tracklets to process
            
        Returns:
            Cleaned dictionary of tracklets
        """
        print(f"Starting post-processing on {len(tracklets)} tracklets...")
        
        # Apply heuristics
        tracklets = self._remove_phantom_tracks(tracklets)
        tracklets = self._remove_static_objects(tracklets)
        tracklets = self._remove_short_tracks(tracklets)
        
        print(f"Post-processing complete. {len(tracklets)} tracklets remaining.")
        print(f"Removed {len(self.removed_tracks)} tracks.")
        
        return tracklets
    
    def _remove_phantom_tracks(self, tracklets: Dict[int, Tracklet]) -> Dict[int, Tracklet]:
        """
        Remove phantom tracks that extend beyond the last detection.
        
        Phantom tracks arise when the tracker compensates for missed
        detections by predicting bounding boxes based on motion models.
        These predictions can become unreliable over time.
        """
        print("Removing phantom tracks...")
        
        cleaned_tracklets = {}
        removed_count = 0
        
        for track_id, tracklet in tracklets.items():
            if len(tracklet.detections) < 2:
                # Skip very short tracks
                cleaned_tracklets[track_id] = tracklet
                continue
            
            # Check if this track has phantom predictions
            if self._is_phantom_track(tracklet):
                self.removed_tracks.append({
                    'track_id': track_id,
                    'reason': 'phantom_track',
                    'length': len(tracklet.detections)
                })
                removed_count += 1
            else:
                cleaned_tracklets[track_id] = tracklet
        
        print(f"Removed {removed_count} phantom tracks")
        return cleaned_tracklets
    
    def _is_phantom_track(self, tracklet: Tracklet) -> bool:
        """
        Determine if a tracklet contains phantom predictions.
        
        Args:
            tracklet: Tracklet to check
            
        Returns:
            True if tracklet contains phantom predictions
        """
        if len(tracklet.detections) < self.config.phantom_track_threshold:
            return False
        
        # Check for rapid bounding box size changes (indicates prediction instability)
        bbox_sizes = []
        for detection in tracklet.detections:
            bbox = detection['bbox']
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            bbox_sizes.append(width * height)
        
        # Calculate size stability
        size_changes = np.diff(bbox_sizes)
        size_stability = np.std(size_changes)
        
        # If size changes are very unstable, likely phantom
        if size_stability > np.mean(bbox_sizes) * 0.5:
            return True
        
        return False
    
    def _remove_static_objects(self, tracklets: Dict[int, Tracklet]) -> Dict[int, Tracklet]:
        """
        Remove tracks that represent static objects (e.g., manikins).
        
        Static objects show minimal displacement and are often
        misclassified as people by detection models.
        """
        print("Removing static objects...")
        
        cleaned_tracklets = {}
        removed_count = 0
        
        for track_id, tracklet in tracklets.items():
            if self._is_static_object(tracklet):
                self.removed_tracks.append({
                    'track_id': track_id,
                    'reason': 'static_object',
                    'length': len(tracklet.detections)
                })
                removed_count += 1
            else:
                cleaned_tracklets[track_id] = tracklet
        
        print(f"Removed {removed_count} static objects")
        return cleaned_tracklets
    
    def _is_static_object(self, tracklet: Tracklet) -> bool:
        """
        Determine if a tracklet represents a static object.
        
        Args:
            tracklet: Tracklet to check
            
        Returns:
            True if tracklet represents a static object
        """
        if len(tracklet.detections) < 3:
            return False
        
        # Calculate movement statistics
        movement_stats = tracklet.get_movement_statistics()
        
        # Check if movement is below threshold
        if movement_stats['average_movement'] < self.config.movement_threshold:
            return True
        
        # Check normalized movement range
        bbox_sizes = []
        for detection in tracklet.detections:
            bbox = detection['bbox']
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            bbox_sizes.append((width, height))
        
        # Calculate average bbox size
        avg_width = np.mean([size[0] for size in bbox_sizes])
        avg_height = np.mean([size[1] for size in bbox_sizes])
        
        # Calculate movement range
        x_coords = [det['bbox'][0] for det in tracklet.detections]
        y_coords = [det['bbox'][1] for det in tracklet.detections]
        
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        
        # Normalize by average bbox size
        normalized_x_range = x_range / avg_width if avg_width > 0 else 0
        normalized_y_range = y_range / avg_height if avg_height > 0 else 0
        
        # If normalized movement is very small, likely static
        if (normalized_x_range < self.config.static_threshold and 
            normalized_y_range < self.config.static_threshold):
            return True
        
        return False
    
    def _remove_short_tracks(self, tracklets: Dict[int, Tracklet]) -> Dict[int, Tracklet]:
        """
        Remove tracks that are too short to be reliable.
        
        Args:
            tracklets: Dictionary of tracklets
            
        Returns:
            Filtered dictionary of tracklets
        """
        cleaned_tracklets = {}
        removed_count = 0
        
        for track_id, tracklet in tracklets.items():
            if len(tracklet.detections) >= self.config.min_track_length:
                cleaned_tracklets[track_id] = tracklet
            else:
                self.removed_tracks.append({
                    'track_id': track_id,
                    'reason': 'short_track',
                    'length': len(tracklet.detections)
                })
                removed_count += 1
        
        if removed_count > 0:
            print(f"Removed {removed_count} short tracks")
        
        return cleaned_tracklets
    
    def get_removal_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about removed tracks.
        
        Returns:
            Dictionary containing removal statistics
        """
        if not self.removed_tracks:
            return {'total_removed': 0, 'reasons': {}}
        
        reasons = {}
        for track in self.removed_tracks:
            reason = track['reason']
            if reason not in reasons:
                reasons[reason] = 0
            reasons[reason] += 1
        
        return {
            'total_removed': len(self.removed_tracks),
            'reasons': reasons,
            'removed_tracks': self.removed_tracks
        }
    
    def apply_reid_post_processing(
        self,
        tracklets: Dict[int, Tracklet],
        similarity_threshold: float = 0.85
    ) -> Dict[int, Tracklet]:
        """
        Apply Re-ID based post-processing to re-link fragmented tracks.
        
        This implements the appearance-based re-linking strategy described
        in the paper for mitigating identity switches.
        
        Args:
            tracklets: Dictionary of tracklets
            similarity_threshold: Threshold for re-linking tracks
            
        Returns:
            Updated tracklets with re-linked fragments
        """
        print("Applying Re-ID post-processing...")
        
        # Group tracklets by camera and time
        tracklet_groups = self._group_tracklets_by_time(tracklets)
        
        # Find potential re-linking candidates
        re_link_candidates = self._find_relink_candidates(tracklet_groups)
        
        # Apply re-linking
        updated_tracklets = self._apply_relinking(
            tracklets, re_link_candidates, similarity_threshold
        )
        
        return updated_tracklets
    
    def _group_tracklets_by_time(self, tracklets: Dict[int, Tracklet]) -> Dict[str, List[Tracklet]]:
        """Group tracklets by time intervals for efficient processing."""
        groups = {}
        
        for tracklet in tracklets.values():
            # Create time-based grouping key
            start_time = tracklet.start_frame
            end_time = tracklet.end_frame
            
            # Group by 30-frame intervals (1 second at 30 FPS)
            interval = start_time // 30
            key = f"interval_{interval}"
            
            if key not in groups:
                groups[key] = []
            groups[key].append(tracklet)
        
        return groups
    
    def _find_relink_candidates(
        self,
        tracklet_groups: Dict[str, List[Tracklet]]
    ) -> List[Dict[str, Any]]:
        """Find potential candidates for re-linking fragmented tracks."""
        candidates = []
        
        for group_key, tracklets in tracklet_groups.items():
            # Sort tracklets by start time
            sorted_tracklets = sorted(tracklets, key=lambda t: t.start_frame)
            
            for i, tracklet1 in enumerate(sorted_tracklets):
                for j, tracklet2 in enumerate(sorted_tracklets[i+1:], i+1):
                    # Check if these could be the same person
                    if self._are_relink_candidates(tracklet1, tracklet2):
                        candidates.append({
                            'track1_id': tracklet1.track_id,
                            'track2_id': tracklet2.track_id,
                            'similarity': 0.0,  # Will be computed later
                            'reason': 'spatial_temporal_proximity'
                        })
        
        return candidates
    
    def _are_relink_candidates(self, tracklet1: Tracklet, tracklet2: Tracklet) -> bool:
        """Check if two tracklets are candidates for re-linking."""
        # Check temporal proximity
        time_gap = abs(tracklet1.end_frame - tracklet2.start_frame)
        if time_gap > 60:  # More than 2 seconds gap
            return False
        
        # Check spatial proximity (if overlapping time)
        if tracklet1.end_frame < tracklet2.start_frame:
            # Track1 ends before Track2 starts
            bbox1 = tracklet1.get_bbox_at_frame(tracklet1.end_frame)
            bbox2 = tracklet2.get_bbox_at_frame(tracklet2.start_frame)
        else:
            # Some overlap
            bbox1 = tracklet1.get_bbox_at_frame(tracklet1.end_frame)
            bbox2 = tracklet2.get_bbox_at_frame(tracklet2.start_frame)
        
        if bbox1 and bbox2:
            distance = self._calculate_bbox_distance(bbox1, bbox2)
            if distance > 100:  # More than 100 pixels apart
                return False
        
        return True
    
    def _calculate_bbox_distance(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate distance between two bounding boxes."""
        center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
        center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
        
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def _apply_relinking(
        self,
        tracklets: Dict[int, Tracklet],
        candidates: List[Dict[str, Any]],
        similarity_threshold: float
    ) -> Dict[int, Tracklet]:
        """Apply re-linking to merge fragmented tracks."""
        # TODO: Implement actual re-linking logic
        # This would involve:
        # 1. Computing appearance similarity between candidates
        # 2. Merging tracks that exceed similarity threshold
        # 3. Updating track IDs and merging detection data
        
        print(f"Found {len(candidates)} re-linking candidates")
        print("Re-linking implementation pending...")
        
        return tracklets
