"""
Cross-Camera Person Re-identification

This module implements the cross-camera Re-ID methodology described in the paper,
using a robust spatiotemporal matching strategy with homography-based coordinate
alignment and appearance-based fallback.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
from pathlib import Path

from ..utils.tracking_utils import Tracklet


@dataclass
class ReIDConfig:
    """Configuration for cross-camera Re-ID system."""
    appearance_threshold: float = 0.8
    spatial_tolerance: float = 0.5  # meters
    temporal_window: float = 5.0    # seconds
    homography_enabled: bool = True
    appearance_enabled: bool = True


@dataclass
class CameraInfo:
    """Information about a camera in the system."""
    camera_id: str
    position: Tuple[float, float, float]  # x, y, height in meters
    homography_matrix: Optional[np.ndarray] = None
    homography_file: Optional[str] = None


class CrossCameraReID:
    """
    Cross-camera person Re-identification system.
    
    This class implements the multi-stage matching process described in the paper:
    1. Spatiotemporal constraints using homography for coordinate alignment
    2. Appearance-based matching as a fallback mechanism
    3. Novel exponential loss function for similarity scoring
    """
    
    def __init__(
        self,
        cameras: List[CameraInfo],
        config: Optional[ReIDConfig] = None
    ):
        """
        Initialize the cross-camera Re-ID system.
        
        Args:
            cameras: List of camera information
            config: Re-ID configuration
        """
        self.cameras = {cam.camera_id: cam for cam in cameras}
        self.config = config or ReIDConfig()
        
        # Load homography matrices
        self._load_homography_matrices()
        
        # Initialize Re-ID state
        self.global_trajectories: Dict[str, Dict[str, Any]] = {}
        self.next_global_id = 1
        
    def _load_homography_matrices(self):
        """Load homography matrices for cameras."""
        for camera in self.cameras.values():
            if camera.homography_file and Path(camera.homography_file).exists():
                try:
                    with open(camera.homography_file, 'r') as f:
                        homography_data = json.load(f)
                        camera.homography_matrix = np.array(homography_data['matrix'])
                    print(f"Loaded homography for camera {camera.camera_id}")
                except Exception as e:
                    print(f"Failed to load homography for camera {camera.camera_id}: {e}")
    
    def link_tracklets(
        self,
        camera_tracklets: Dict[str, Dict[int, Tracklet]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Link tracklets across cameras to form global trajectories.
        
        Args:
            camera_tracklets: Dictionary mapping camera IDs to tracklet dictionaries
            
        Returns:
            Dictionary of global trajectories
        """
        print("Starting cross-camera tracklet linking...")
        
        # Initialize global trajectories
        self.global_trajectories = {}
        self.next_global_id = 1
        
        # Convert camera tracklets to list format for processing
        all_tracklets = []
        for camera_id, tracklets in camera_tracklets.items():
            for track_id, tracklet in tracklets.items():
                all_tracklets.append({
                    'camera_id': camera_id,
                    'track_id': track_id,
                    'tracklet': tracklet
                })
        
        # Sort tracklets by start time
        all_tracklets.sort(key=lambda x: x['tracklet'].start_frame)
        
        # Process tracklets sequentially
        for tracklet_info in all_tracklets:
            self._process_tracklet(tracklet_info)
        
        print(f"Cross-camera linking complete. {len(self.global_trajectories)} global trajectories created.")
        
        return self.global_trajectories
    
    def _process_tracklet(self, tracklet_info: Dict[str, Any]):
        """
        Process a single tracklet for cross-camera linking.
        
        Args:
            tracklet_info: Information about the tracklet to process
        """
        camera_id = tracklet_info['camera_id']
        track_id = tracklet_info['track_id']
        tracklet = tracklet_info['tracklet']
        
        # Try to find matching global trajectory
        matched_global_id = self._find_matching_trajectory(tracklet_info)
        
        if matched_global_id:
            # Add to existing trajectory
            self._add_to_trajectory(matched_global_id, camera_id, track_id, tracklet)
        else:
            # Create new global trajectory
            self._create_new_trajectory(camera_id, track_id, tracklet)
    
    def _find_matching_trajectory(
        self,
        tracklet_info: Dict[str, Any]
    ) -> Optional[str]:
        """
        Find a matching global trajectory for the given tracklet.
        
        Args:
            tracklet_info: Information about the tracklet
            
        Returns:
            Global trajectory ID if match found, None otherwise
        """
        camera_id = tracklet_info['camera_id']
        tracklet = tracklet_info['tracklet']
        
        best_match_id = None
        best_score = 0.0
        
        for global_id, trajectory in self.global_trajectories.items():
            # Calculate matching score
            score = self._calculate_matching_score(tracklet_info, trajectory)
            
            if score > best_score and score > self.config.appearance_threshold:
                best_score = score
                best_match_id = global_id
        
        return best_match_id
    
    def _calculate_matching_score(
        self,
        tracklet_info: Dict[str, Any],
        trajectory: Dict[str, Any]
    ) -> float:
        """
        Calculate matching score between a tracklet and a global trajectory.
        
        Args:
            tracklet_info: Information about the tracklet
            trajectory: Global trajectory information
            
        Returns:
            Matching score between 0 and 1
        """
        camera_id = tracklet_info['camera_id']
        tracklet = tracklet_info['tracklet']
        
        # Initialize score components
        spatiotemporal_score = 0.0
        appearance_score = 0.0
        
        # Calculate spatiotemporal score
        if self.config.homography_enabled:
            spatiotemporal_score = self._calculate_spatiotemporal_score(
                tracklet_info, trajectory
            )
        
        # Calculate appearance score
        if self.config.appearance_enabled:
            appearance_score = self._calculate_appearance_score(
                tracklet_info, trajectory
            )
        
        # Combine scores using weighted average
        if self.config.homography_enabled and self.config.appearance_enabled:
            # Use spatiotemporal as primary, appearance as fallback
            if spatiotemporal_score > 0.7:
                final_score = 0.8 * spatiotemporal_score + 0.2 * appearance_score
            else:
                final_score = 0.3 * spatiotemporal_score + 0.7 * appearance_score
        elif self.config.homography_enabled:
            final_score = spatiotemporal_score
        else:
            final_score = appearance_score
        
        return final_score
    
    def _calculate_spatiotemporal_score(
        self,
        tracklet_info: Dict[str, Any],
        trajectory: Dict[str, Any]
    ) -> float:
        """
        Calculate spatiotemporal matching score using homography.
        
        Args:
            tracklet_info: Information about the tracklet
            trajectory: Global trajectory information
            
        Returns:
            Spatiotemporal score between 0 and 1
        """
        camera_id = tracklet_info['camera_id']
        tracklet = tracklet_info['tracklet']
        
        # Get camera homography
        camera = self.cameras.get(camera_id)
        if not camera or camera.homography_matrix is None:
            return 0.0
        
        # Check temporal overlap with existing trajectories
        temporal_scores = []
        
        for traj_camera_id, traj_info in trajectory['trajectories'].items():
            if traj_camera_id == camera_id:
                continue  # Skip same camera
            
            # Check if there's temporal overlap
            overlap_score = self._calculate_temporal_overlap(
                tracklet, traj_info
            )
            
            if overlap_score > 0:
                # Calculate spatial consistency using homography
                spatial_score = self._calculate_spatial_consistency(
                    tracklet, traj_info, camera, traj_camera_id
                )
                
                # Combine temporal and spatial scores
                combined_score = overlap_score * spatial_score
                temporal_scores.append(combined_score)
        
        if not temporal_scores:
            return 0.0
        
        # Return maximum score
        return max(temporal_scores)
    
    def _calculate_temporal_overlap(
        self,
        tracklet: Tracklet,
        traj_info: Dict[str, Any]
    ) -> float:
        """
        Calculate temporal overlap between tracklet and trajectory.
        
        Args:
            tracklet: Current tracklet
            traj_info: Trajectory information
            
        Returns:
            Temporal overlap score between 0 and 1
        """
        # Get frame ranges
        tracklet_start = tracklet.start_frame
        tracklet_end = tracklet.end_frame
        traj_start = traj_info['start_frame']
        traj_end = traj_info['end_frame']
        
        # Calculate overlap
        overlap_start = max(tracklet_start, traj_start)
        overlap_end = min(tracklet_end, traj_end)
        
        if overlap_end <= overlap_start:
            return 0.0
        
        overlap_duration = overlap_end - overlap_start
        tracklet_duration = tracklet_end - tracklet_start
        traj_duration = traj_end - traj_start
        
        # Calculate overlap ratio
        overlap_ratio = overlap_duration / min(tracklet_duration, traj_duration)
        
        # Apply temporal window constraint
        if overlap_duration > self.config.temporal_window * 30:  # Convert to frames
            overlap_ratio *= 0.5  # Penalize very long overlaps
        
        return overlap_ratio
    
    def _calculate_spatial_consistency(
        self,
        tracklet: Tracklet,
        traj_info: Dict[str, Any],
        camera: CameraInfo,
        traj_camera_id: str
    ) -> float:
        """
        Calculate spatial consistency using homography transformation.
        
        Args:
            tracklet: Current tracklet
            traj_info: Trajectory information
            camera: Current camera information
            traj_camera_id: Trajectory camera ID
            
        Returns:
            Spatial consistency score between 0 and 1
        """
        # Get trajectory camera homography
        traj_camera = self.cameras.get(traj_camera_id)
        if not traj_camera or traj_camera.homography_matrix is None:
            return 0.0
        
        # Transform tracklet coordinates to common ground plane
        tracklet_ground_coords = self._transform_to_ground_plane(
            tracklet, camera
        )
        
        # Transform trajectory coordinates to common ground plane
        traj_ground_coords = self._transform_to_ground_plane(
            traj_info, traj_camera
        )
        
        if tracklet_ground_coords is None or traj_ground_coords is None:
            return 0.0
        
        # Calculate spatial distance
        spatial_distance = np.linalg.norm(
            np.array(tracklet_ground_coords) - np.array(traj_ground_coords)
        )
        
        # Convert to score using exponential loss function
        # This implements the novel exponential loss function mentioned in the paper
        spatial_score = np.exp(-spatial_distance / self.config.spatial_tolerance)
        
        return max(0.0, spatial_score)
    
    def _transform_to_ground_plane(
        self,
        tracklet_or_traj: Any,
        camera: CameraInfo
    ) -> Optional[Tuple[float, float]]:
        """
        Transform coordinates to common ground plane using homography.
        
        Args:
            tracklet_or_traj: Tracklet or trajectory information
            camera: Camera information with homography matrix
            
        Returns:
            Ground plane coordinates (x, y) or None if transformation fails
        """
        if camera.homography_matrix is None:
            return None
        
        try:
            # Get representative coordinates (center of tracklet/trajectory)
            if hasattr(tracklet_or_traj, 'detections'):
                # It's a tracklet
                bbox = tracklet_or_traj.get_bbox_at_frame(
                    tracklet_or_traj.start_frame
                )
                if bbox is None:
                    return None
                
                # Use center of bounding box
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
            else:
                # It's trajectory info
                center_x = tracklet_or_traj.get('center_x', 0)
                center_y = tracklet_or_traj.get('center_y', 0)
            
            # Apply homography transformation
            point_homogeneous = np.array([center_x, center_y, 1.0])
            transformed_point = camera.homography_matrix @ point_homogeneous
            
            # Normalize homogeneous coordinates
            if transformed_point[2] != 0:
                ground_x = transformed_point[0] / transformed_point[2]
                ground_y = transformed_point[1] / transformed_point[2]
                return (ground_x, ground_y)
            
        except Exception as e:
            print(f"Error in homography transformation: {e}")
        
        return None
    
    def _calculate_appearance_score(
        self,
        tracklet_info: Dict[str, Any],
        trajectory: Dict[str, Any]
    ) -> float:
        """
        Calculate appearance-based matching score.
        
        Args:
            tracklet_info: Information about the tracklet
            trajectory: Global trajectory information
            
        Returns:
            Appearance score between 0 and 1
        """
        tracklet = tracklet_info['tracklet']
        
        # Get tracklet features
        tracklet_features = tracklet.get_average_features()
        if tracklet_features is None:
            return 0.0
        
        # Compare with trajectory features
        appearance_scores = []
        
        for traj_camera_id, traj_info in trajectory['trajectories'].items():
            # TODO: Implement actual feature comparison
            # This would require loading the actual trajectory tracklets
            # and computing feature similarity
            
            # Placeholder: return moderate score
            appearance_scores.append(0.5)
        
        if not appearance_scores:
            return 0.0
        
        return max(appearance_scores)
    
    def _add_to_trajectory(
        self,
        global_id: str,
        camera_id: str,
        track_id: int,
        tracklet: Tracklet
    ):
        """Add tracklet to existing global trajectory."""
        trajectory = self.global_trajectories[global_id]
        
        # Add camera trajectory information
        trajectory['trajectories'][camera_id] = {
            'track_id': track_id,
            'start_frame': tracklet.start_frame,
            'end_frame': tracklet.end_frame,
            'total_frames': len(tracklet.detections)
        }
        
        # Update trajectory metadata
        trajectory['total_cameras'] = len(trajectory['trajectories'])
        trajectory['total_detections'] += len(tracklet.detections)
        
        print(f"Added camera {camera_id} track {track_id} to global trajectory {global_id}")
    
    def _create_new_trajectory(
        self,
        camera_id: str,
        track_id: int,
        tracklet: Tracklet
    ):
        """Create a new global trajectory."""
        global_id = f"person_{self.next_global_id:04d}"
        self.next_global_id += 1
        
        # Create trajectory structure
        trajectory = {
            'global_id': global_id,
            'trajectories': {
                camera_id: {
                    'track_id': track_id,
                    'start_frame': tracklet.start_frame,
                    'end_frame': tracklet.end_frame,
                    'total_frames': len(tracklet.detections)
                }
            },
            'total_cameras': 1,
            'total_detections': len(tracklet.detections),
            'creation_time': tracklet.start_frame
        }
        
        self.global_trajectories[global_id] = trajectory
        
        print(f"Created new global trajectory {global_id} for camera {camera_id} track {track_id}")
    
    def save_trajectories(self, output_path: str):
        """Save global trajectories to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.global_trajectories, f, indent=2)
        
        print(f"Global trajectories saved to: {output_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get Re-ID system statistics."""
        total_trajectories = len(self.global_trajectories)
        total_cameras = sum(
            traj['total_cameras'] for traj in self.global_trajectories.values()
        )
        total_detections = sum(
            traj['total_detections'] for traj in self.global_trajectories.values()
        )
        
        return {
            'total_trajectories': total_trajectories,
            'total_cameras': total_cameras,
            'total_detections': total_detections,
            'average_cameras_per_trajectory': total_cameras / total_trajectories if total_trajectories > 0 else 0,
            'average_detections_per_trajectory': total_detections / total_trajectories if total_trajectories > 0 else 0
        }
