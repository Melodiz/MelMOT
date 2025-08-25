"""
YOLO Object Detector

This module provides a clean interface to the YOLO object detection model
using the ultralytics library.
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path
import torch

from ..utils.tracking_utils import Detection


class YOLODetector:
    """
    YOLO-based object detector for person detection.
    
    This class provides a clean interface to YOLO models for detecting
    people in video frames, with configurable confidence thresholds
    and class filtering.
    """
    
    def __init__(
        self,
        model_path: str = "yolov12n.pt",
        device: str = "auto",
        confidence_threshold: float = 0.5,
        class_filter: Optional[List[int]] = None
    ):
        """
        Initialize the YOLO detector.
        
        Args:
            model_path: Path to YOLO model weights
            device: Device to run inference on ('cpu', 'cuda', or 'auto')
            confidence_threshold: Minimum confidence for detections
            class_filter: List of class IDs to detect (None for all classes)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.class_filter = class_filter or [0]  # Default to person class
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Load model
        self._load_model()
        
    def _load_model(self):
        """Load the YOLO model."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            print(f"YOLO model loaded from {self.model_path}")
            print(f"Running on device: {self.device}")
        except ImportError:
            raise ImportError(
                "ultralytics package not found. Install with: pip install ultralytics"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")
    
    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: Optional[float] = None
    ) -> List[Detection]:
        """
        Detect objects in a single frame.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            confidence_threshold: Override default confidence threshold
            
        Returns:
            List of Detection objects
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
            
        # Run inference
        results = self.model(frame, verbose=False)
        
        # Extract detections
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get confidence and class
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                
                # Apply confidence threshold
                if confidence < confidence_threshold:
                    continue
                    
                # Apply class filter
                if self.class_filter and class_id not in self.class_filter:
                    continue
                    
                # Create detection object
                detection = Detection(
                    bbox=[x1, y1, x2, y2],
                    confidence=confidence,
                    class_id=class_id
                )
                
                detections.append(detection)
        
        return detections
    
    def detect_batch(
        self,
        frames: List[np.ndarray],
        confidence_threshold: Optional[float] = None
    ) -> List[List[Detection]]:
        """
        Detect objects in multiple frames (batch processing).
        
        Args:
            frames: List of input frames
            confidence_threshold: Override default confidence threshold
            
        Returns:
            List of detection lists, one per frame
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
            
        # Run batch inference
        results = self.model(frames, verbose=False)
        
        # Process results
        all_detections = []
        
        for result in results:
            frame_detections = []
            boxes = result.boxes
            
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Get confidence and class
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Apply confidence threshold
                    if confidence < confidence_threshold:
                        continue
                        
                    # Apply class filter
                    if self.class_filter and class_id not in self.class_filter:
                        continue
                        
                    # Create detection object
                    detection = Detection(
                        bbox=[x1, y1, x2, y2],
                        confidence=confidence,
                        class_id=class_id
                    )
                    
                    frame_detections.append(detection)
            
            all_detections.append(frame_detections)
        
        return all_detections
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        try:
            return {
                'model_path': self.model_path,
                'device': self.device,
                'confidence_threshold': self.confidence_threshold,
                'class_filter': self.class_filter,
                'model_type': type(self.model).__name__
            }
        except Exception:
            return {'error': 'Could not retrieve model info'}
    
    def set_confidence_threshold(self, threshold: float):
        """Set the confidence threshold for detections."""
        if 0.0 <= threshold <= 1.0:
            self.confidence_threshold = threshold
        else:
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
    
    def set_class_filter(self, class_ids: List[int]):
        """Set the class filter for detections."""
        self.class_filter = class_ids
    
    def warmup(self, input_shape: Tuple[int, int] = (640, 640)):
        """
        Warm up the model with dummy input.
        
        Args:
            input_shape: Shape of dummy input (height, width)
        """
        print("Warming up YOLO model...")
        dummy_input = np.random.randint(0, 255, (*input_shape, 3), dtype=np.uint8)
        
        # Run inference multiple times to warm up
        for _ in range(3):
            _ = self.detect(dummy_input)
        
        print("Model warmup complete")
