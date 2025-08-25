#!/usr/bin/env python3
"""
Simple MelMOT Usage Example

This script demonstrates basic usage of the MelMOT system for
single-camera tracking and cross-camera Re-identification.
"""

import sys
from pathlib import Path

# Add the melmot package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from melmot.core.tracker import SingleCameraTracker, TrackingConfig
from melmot.reid.cross_camera import CrossCameraReID, ReIDConfig, CameraInfo
from melmot.utils.tracking_utils import Tracklet


def main():
    """Main example function."""
    print("MelMOT Simple Example")
    print("=" * 50)
    
    # Example 1: Single Camera Tracking
    print("\n1. Single Camera Tracking Example")
    print("-" * 30)
    
    # Create tracking configuration
    tracking_config = TrackingConfig(
        confidence_threshold=0.5,
        static_threshold=2.0,
        movement_threshold=0.1
    )
    
    # Initialize tracker
    tracker = SingleCameraTracker(
        model_path="yolov12n.pt",
        config=tracking_config,
        device="auto"
    )
    
    print("Tracker initialized successfully!")
    print(f"Model info: {tracker.get_model_info()}")
    
    # Example 2: Cross-Camera Re-ID Setup
    print("\n2. Cross-Camera Re-ID Example")
    print("-" * 30)
    
    # Create camera configurations
    cameras = [
        CameraInfo(
            camera_id="camera_001",
            position=(0.0, 0.0, 3.5),
            homography_file="melmot/homography/entrance_cam.json"
        ),
        CameraInfo(
            camera_id="camera_002",
            position=(15.0, 0.0, 3.5),
            homography_file=None
        )
    ]
    
    # Initialize Re-ID system
    reid_config = ReIDConfig(
        appearance_threshold=0.8,
        spatial_tolerance=0.5,
        temporal_window=5.0
    )
    
    reid_system = CrossCameraReID(cameras, reid_config)
    
    print("Re-ID system initialized successfully!")
    print(f"Number of cameras: {len(cameras)}")
    
    # Example 3: Create Sample Tracklets
    print("\n3. Sample Tracklet Creation")
    print("-" * 30)
    
    # Create sample tracklets
    tracklet1 = Tracklet(track_id=1)
    tracklet1.add_detection(100, [100, 200, 150, 300], 0.95)
    tracklet1.add_detection(101, [105, 205, 155, 305], 0.92)
    tracklet1.add_detection(102, [110, 210, 160, 310], 0.88)
    
    tracklet2 = Tracklet(track_id=2)
    tracklet2.add_detection(100, [200, 150, 250, 250], 0.91)
    tracklet2.add_detection(101, [205, 155, 255, 255], 0.89)
    
    print(f"Created tracklet 1: {len(tracklet1)} detections")
    print(f"Created tracklet 2: {len(tracklet2)} detections")
    
    # Example 4: Movement Statistics
    print("\n4. Movement Statistics")
    print("-" * 30)
    
    stats1 = tracklet1.get_movement_statistics()
    stats2 = tracklet2.get_movement_statistics()
    
    print(f"Tracklet 1 - Total movement: {stats1['total_movement']:.2f}")
    print(f"Tracklet 1 - Average movement: {stats1['average_movement']:.2f}")
    print(f"Tracklet 2 - Total movement: {stats2['total_movement']:.2f}")
    print(f"Tracklet 2 - Average movement: {stats2['average_movement']:.2f}")
    
    # Example 5: Post-Processing Simulation
    print("\n5. Post-Processing Simulation")
    print("-" * 30)
    
    # Simulate post-processing
    from melmot.utils.post_processing import PostProcessor, PostProcessingConfig
    
    post_config = PostProcessingConfig(
        static_threshold=2.0,
        movement_threshold=0.1,
        min_track_length=3
    )
    
    post_processor = PostProcessor(post_config)
    
    # Create sample tracklets dictionary
    sample_tracklets = {1: tracklet1, 2: tracklet2}
    
    # Apply post-processing
    cleaned_tracklets = post_processor.process(sample_tracklets)
    
    print(f"Original tracklets: {len(sample_tracklets)}")
    print(f"Cleaned tracklets: {len(cleaned_tracklets)}")
    
    # Get removal statistics
    removal_stats = post_processor.get_removal_statistics()
    print(f"Removed tracks: {removal_stats['total_removed']}")
    
    print("\nExample completed successfully!")
    print("\nTo run actual tracking:")
    print("  python -m melmot.cli track --video videos/input.mp4 --output results/output.mp4")
    print("\nTo run cross-camera Re-ID:")
    print("  python -m melmot.cli reid --tracklets results/tracklets.json --output results/global_trajectories.json")


if __name__ == "__main__":
    main()
