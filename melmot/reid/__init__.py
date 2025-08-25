"""
Re-identification modules for MelMOT.
"""

from .cross_camera import CrossCameraReID, ReIDConfig, CameraInfo

__all__ = [
    'CrossCameraReID',
    'ReIDConfig',
    'CameraInfo'
]
