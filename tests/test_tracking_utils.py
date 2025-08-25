"""
Tests for tracking utilities.
"""

import pytest
import numpy as np
from melmot.utils.tracking_utils import Detection, Tracklet, calculate_iou, calculate_distance


class TestDetection:
    """Test Detection class."""
    
    def test_detection_creation(self):
        """Test creating a detection."""
        detection = Detection(
            bbox=[100, 200, 150, 300],
            confidence=0.95,
            class_id=0
        )
        
        assert detection.bbox == [100, 200, 150, 300]
        assert detection.confidence == 0.95
        assert detection.class_id == 0
    
    def test_detection_validation(self):
        """Test detection validation."""
        # Test invalid bbox length
        with pytest.raises(ValueError):
            Detection(bbox=[100, 200], confidence=0.95, class_id=0)
        
        # Test invalid confidence
        with pytest.raises(ValueError):
            Detection(bbox=[100, 200, 150, 300], confidence=1.5, class_id=0)
    
    def test_detection_properties(self):
        """Test detection properties."""
        detection = Detection(
            bbox=[100, 200, 150, 300],
            confidence=0.95,
            class_id=0
        )
        
        assert detection.center == (125.0, 250.0)
        assert detection.width == 50.0
        assert detection.height == 100.0
        assert detection.area == 5000.0
    
    def test_detection_to_dict(self):
        """Test detection serialization."""
        detection = Detection(
            bbox=[100, 200, 150, 300],
            confidence=0.95,
            class_id=0
        )
        
        detection_dict = detection.to_dict()
        assert detection_dict['bbox'] == [100, 200, 150, 300]
        assert detection_dict['confidence'] == 0.95
        assert detection_dict['class_id'] == 0


class TestTracklet:
    """Test Tracklet class."""
    
    def test_tracklet_creation(self):
        """Test creating a tracklet."""
        tracklet = Tracklet(track_id=1)
        
        assert tracklet.track_id == 1
        assert len(tracklet.detections) == 0
        assert tracklet.start_frame is None
        assert tracklet.end_frame is None
        assert tracklet.total_frames == 0
    
    def test_adding_detections(self):
        """Test adding detections to tracklet."""
        tracklet = Tracklet(track_id=1)
        
        # Add first detection
        tracklet.add_detection(100, [100, 200, 150, 300], 0.95)
        
        assert len(tracklet.detections) == 1
        assert tracklet.start_frame == 100
        assert tracklet.end_frame == 100
        assert tracklet.total_frames == 1
        
        # Add second detection
        tracklet.add_detection(101, [105, 205, 155, 305], 0.92)
        
        assert len(tracklet.detections) == 2
        assert tracklet.start_frame == 100
        assert tracklet.end_frame == 101
        assert tracklet.total_frames == 2
    
    def test_get_detection_at_frame(self):
        """Test getting detection at specific frame."""
        tracklet = Tracklet(track_id=1)
        tracklet.add_detection(100, [100, 200, 150, 300], 0.95)
        tracklet.add_detection(101, [105, 205, 155, 305], 0.92)
        
        # Test existing frame
        detection = tracklet.get_detection_at_frame(100)
        assert detection is not None
        assert detection['frame'] == 100
        assert detection['bbox'] == [100, 200, 150, 300]
        
        # Test non-existing frame
        detection = tracklet.get_detection_at_frame(102)
        assert detection is None
    
    def test_movement_statistics(self):
        """Test movement statistics calculation."""
        tracklet = Tracklet(track_id=1)
        tracklet.add_detection(100, [100, 200, 150, 300], 0.95)
        tracklet.add_detection(101, [105, 205, 155, 305], 0.92)
        tracklet.add_detection(102, [110, 210, 160, 310], 0.88)
        
        stats = tracklet.get_movement_statistics()
        
        assert 'total_movement' in stats
        assert 'average_movement' in stats
        assert 'max_movement' in stats
        assert stats['total_movement'] > 0
        assert stats['average_movement'] > 0
        assert stats['max_movement'] > 0
    
    def test_tracklet_serialization(self):
        """Test tracklet serialization."""
        tracklet = Tracklet(track_id=1)
        tracklet.add_detection(100, [100, 200, 150, 300], 0.95)
        tracklet.add_detection(101, [105, 205, 155, 305], 0.92)
        
        tracklet_dict = tracklet.to_dict()
        
        assert tracklet_dict['track_id'] == 1
        assert tracklet_dict['start_frame'] == 100
        assert tracklet_dict['end_frame'] == 101
        assert tracklet_dict['total_frames'] == 2
        assert 'detections' in tracklet_dict
        assert 'movement_stats' in tracklet_dict


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_calculate_iou(self):
        """Test IoU calculation."""
        # Test overlapping boxes
        bbox1 = [100, 100, 200, 200]
        bbox2 = [150, 150, 250, 250]
        
        iou = calculate_iou(bbox1, bbox2)
        assert 0 < iou < 1
        
        # Test non-overlapping boxes
        bbox3 = [300, 300, 400, 400]
        iou = calculate_iou(bbox1, bbox3)
        assert iou == 0.0
        
        # Test identical boxes
        iou = calculate_iou(bbox1, bbox1)
        assert iou == 1.0
    
    def test_calculate_distance(self):
        """Test distance calculation."""
        bbox1 = [100, 100, 200, 200]
        bbox2 = [300, 300, 400, 400]
        
        distance = calculate_distance(bbox1, bbox2)
        assert distance > 0
        
        # Test distance between centers
        # bbox1 center: (150, 150), bbox2 center: (350, 350)
        expected_distance = np.sqrt((350-150)**2 + (350-150)**2)
        assert abs(distance - expected_distance) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__])
