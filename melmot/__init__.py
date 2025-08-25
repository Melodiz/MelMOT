"""
MelMOT: Multi-Object Tracking for Retail Spaces

A research project implementing innovative approaches to multi-user tracking
in retail environments, focusing on robust single-camera tracking and
effective cross-camera re-identification (Re-ID).
"""

__version__ = "2.0.0"
__author__ = "Ivan Novosad"
__email__ = "ivan.novosad@example.com"

# Core imports
from .core.tracker import SingleCameraTracker
from .reid.cross_camera import CrossCameraReID
from .detection.yolo_detector import YOLODetector

__all__ = [
    "SingleCameraTracker",
    "CrossCameraReID", 
    "YOLODetector",
]
