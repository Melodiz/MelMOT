import cv2
import numpy as np

class AppearanceModel:
    def __init__(self):
        self.features = {}

    def update(self, frame, bbox, track_id):
        roi = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        feature = cv2.mean(roi)[:3]  # Simple color histogram
        self.features[track_id] = feature

    def match(self, frame, bbox, track_id):
        if track_id not in self.features:
            return False
        
        roi = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        feature = cv2.mean(roi)[:3]
        
        distance = np.linalg.norm(np.array(feature) - np.array(self.features[track_id]))
        return distance < 50  # Adjust this threshold as needed