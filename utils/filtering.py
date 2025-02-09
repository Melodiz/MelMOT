import numpy as np
from filterpy.kalman import KalmanFilter
from utils.filters import MovingAverageFilter, ExponentialMovingAverageFilter, LowPassFilter, MedianFilter

def initialize_filters(track_id, bbox):
    filters = {
        'position': MovingAverageFilter(window_size=5),
        'velocity': ExponentialMovingAverageFilter(alpha=0.3),
        'size': LowPassFilter(alpha=0.7),
        'outlier': MedianFilter(window_size=3)
    }
    
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([bbox[0], bbox[1], 0., 0.])  # Initial state
    kf.F = np.array([[1, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])  # State transition matrix
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])  # Measurement function
    kf.P *= 1000.  # Covariance matrix
    kf.R = np.eye(2) * 50  # Measurement noise
    kf.Q = np.eye(4) * 0.1  # Process noise
    
    return filters, kf

def apply_filters(filters, kalman_filter, bbox):
    center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
    size = np.array([bbox[2] - bbox[0], bbox[3] - bbox[1]])
    
    kalman_filter.predict()
    kalman_filter.update(center)
    
    filtered_center = kalman_filter.x[:2]
    filtered_size = filters['size'].update(size)
    
    # Outlier rejection
    if np.linalg.norm(filtered_center - center) > 50:  # Threshold for outlier
        filtered_center = filters['outlier'].update(center)
    
    # Calculate velocity
    if 'prev_center' in filters:
        velocity = filtered_center - filters['prev_center']
        filtered_velocity = filters['velocity'].update(velocity)
    else:
        filtered_velocity = np.array([0, 0])
    
    filters['prev_center'] = filtered_center
    
    # Reconstruct bounding box
    filtered_bbox = np.array([
        filtered_center[0] - filtered_size[0] / 2,
        filtered_center[1] - filtered_size[1] / 2,
        filtered_center[0] + filtered_size[0] / 2,
        filtered_center[1] + filtered_size[1] / 2
    ]).astype(int)
    
    return filtered_bbox, filtered_center, filtered_size