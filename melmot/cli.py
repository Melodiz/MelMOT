"""
Command Line Interface for MelMOT

This module provides a command-line interface for the MelMOT system,
allowing users to run tracking and Re-ID tasks from the command line.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .core.tracker import SingleCameraTracker, TrackingConfig
from .reid.cross_camera import CrossCameraReID, ReIDConfig, CameraInfo
from .utils.tracking_utils import Tracklet


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MelMOT: Multi-Object Tracking for Retail Spaces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single camera tracking
  python -m melmot.cli track --video videos/input.mp4 --output results/output.mp4
  
  # Cross-camera Re-ID
  python -m melmot.cli reid --tracklets results/tracklets.json --output results/global_trajectories.json
  
  # Full pipeline
  python -m melmot.cli pipeline --config config/retail_mall.yaml
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Track command
    track_parser = subparsers.add_parser('track', help='Single camera tracking')
    track_parser.add_argument('--video', required=True, help='Input video file path')
    track_parser.add_argument('--output', help='Output video file path')
    track_parser.add_argument('--tracklets', help='Output tracklets JSON file path')
    track_parser.add_argument('--model', default='yolov12n.pt', help='YOLO model path')
    track_parser.add_argument('--device', default='auto', help='Device to run on (cpu/cuda/auto)')
    track_parser.add_argument('--confidence', type=float, default=0.5, help='Detection confidence threshold')
    
    # Re-ID command
    reid_parser = subparsers.add_parser('reid', help='Cross-camera Re-identification')
    reid_parser.add_argument('--tracklets', required=True, help='Input tracklets JSON file path')
    reid_parser.add_argument('--output', required=True, help='Output global trajectories JSON file path')
    reid_parser.add_argument('--config', help='Camera configuration file path')
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Full tracking and Re-ID pipeline')
    pipeline_parser.add_argument('--config', required=True, help='Configuration file path')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    try:
        if args.command == 'track':
            run_tracking(args)
        elif args.command == 'reid':
            run_reid(args)
        elif args.command == 'pipeline':
            run_pipeline(args)
        else:
            print(f"Unknown command: {args.command}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def run_tracking(args):
    """Run single camera tracking."""
    print("Starting single camera tracking...")
    
    # Create tracking configuration
    config = TrackingConfig(
        confidence_threshold=args.confidence
    )
    
    # Initialize tracker
    tracker = SingleCameraTracker(
        model_path=args.model,
        config=config,
        device=args.device
    )
    
    # Run tracking
    results = tracker.track_video(
        video_path=args.video,
        output_path=args.output,
        save_tracklets=True
    )
    
    # Save tracklets if requested
    if args.tracklets:
        tracker.save_tracklets(args.tracklets)
    
    # Print statistics
    stats = tracker.get_statistics()
    print("\nTracking Statistics:")
    print(f"  Total tracks: {stats['total_tracks']}")
    print(f"  Total detections: {stats['total_detections']}")
    print(f"  Average track length: {stats['average_track_length']:.2f}")
    
    print("Tracking complete!")


def run_reid(args):
    """Run cross-camera Re-identification."""
    print("Starting cross-camera Re-identification...")
    
    # Load tracklets
    tracklets = load_tracklets(args.tracklets)
    
    # Load camera configuration
    cameras = load_camera_config(args.config) if args.config else create_default_cameras()
    
    # Initialize Re-ID system
    reid_config = ReIDConfig()
    reid_system = CrossCameraReID(cameras, reid_config)
    
    # Run Re-ID
    global_trajectories = reid_system.link_tracklets(tracklets)
    
    # Save results
    reid_system.save_trajectories(args.output)
    
    # Print statistics
    stats = reid_system.get_statistics()
    print("\nRe-ID Statistics:")
    print(f"  Total trajectories: {stats['total_trajectories']}")
    print(f"  Total cameras: {stats['total_cameras']}")
    print(f"  Total detections: {stats['total_detections']}")
    print(f"  Average cameras per trajectory: {stats['average_cameras_per_trajectory']:.2f}")
    
    print("Re-identification complete!")


def run_pipeline(args):
    """Run full tracking and Re-ID pipeline."""
    print("Starting full pipeline...")
    
    # Load configuration
    config = load_pipeline_config(args.config)
    
    # Run tracking for each camera
    all_tracklets = {}
    
    for camera_config in config['cameras']:
        print(f"\nProcessing camera: {camera_config['id']}")
        
        # Initialize tracker
        tracker = SingleCameraTracker(
            model_path=config['tracking']['model_path'],
            device=config['tracking']['device']
        )
        
        # Run tracking
        results = tracker.track_video(
            video_path=camera_config['video_path'],
            output_path=camera_config.get('output_path'),
            save_tracklets=True
        )
        
        # Store tracklets
        all_tracklets[camera_config['id']] = results['tracklets']
    
    # Run cross-camera Re-ID
    print("\nRunning cross-camera Re-identification...")
    
    cameras = [CameraInfo(**cam) for cam in config['cameras']]
    reid_system = CrossCameraReID(cameras, ReIDConfig(**config['reid']))
    
    global_trajectories = reid_system.link_tracklets(all_tracklets)
    
    # Save results
    output_path = config['output']['global_trajectories']
    reid_system.save_trajectories(output_path)
    
    print("Full pipeline complete!")


def load_tracklets(file_path: str) -> dict:
    """Load tracklets from JSON file."""
    import json
    
    with open(file_path, 'r') as f:
        tracklet_data = json.load(f)
    
    # Convert to Tracklet objects
    tracklets = {}
    for camera_id, camera_tracklets in tracklet_data.items():
        tracklets[camera_id] = {}
        for track_id, track_data in camera_tracklets.items():
            tracklet = Tracklet(int(track_id))
            for detection in track_data['detections']:
                tracklet.add_detection(
                    detection['frame'],
                    detection['bbox'],
                    detection.get('confidence', 1.0)
                )
            tracklets[camera_id][int(track_id)] = tracklet
    
    return tracklets


def load_camera_config(file_path: str) -> list:
    """Load camera configuration from JSON file."""
    import json
    
    with open(file_path, 'r') as f:
        config = json.load(f)
    
    cameras = []
    for camera_config in config['cameras']:
        camera = CameraInfo(
            camera_id=camera_config['id'],
            position=tuple(camera_config['position']),
            homography_file=camera_config.get('homography')
        )
        cameras.append(camera)
    
    return cameras


def create_default_cameras() -> list:
    """Create default camera configuration."""
    return [
        CameraInfo(
            camera_id="camera_001",
            position=(0.0, 0.0, 3.5),
            homography_file=None
        )
    ]


def load_pipeline_config(file_path: str) -> dict:
    """Load pipeline configuration from YAML file."""
    try:
        import yaml
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except ImportError:
        print("PyYAML not installed. Install with: pip install pyyaml")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
