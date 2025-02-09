import numpy as np
import cv2

class OcclusionHandler:
    def __init__(self, max_age, appearance_model):
        self.max_age = max_age
        self.appearance_model = appearance_model
        self.occluded_tracks = {}

    def handle_occlusions(self, frame, active_tracks, kalman_filters, filters):
        frame_with_boxes = frame.copy()

        # Handle existing occluded tracks
        for track_id in list(self.occluded_tracks.keys()):
            if track_id in active_tracks:
                del self.occluded_tracks[track_id]
            else:
                self.occluded_tracks[track_id]['frames'] += 1
                if self.occluded_tracks[track_id]['frames'] > self.max_age:
                    del self.occluded_tracks[track_id]
                    if track_id in filters:
                        del filters[track_id]
                    if track_id in kalman_filters:
                        del kalman_filters[track_id]
                else:
                    self._handle_occluded_track(frame_with_boxes, track_id, kalman_filters)

        return frame_with_boxes

    def _handle_occluded_track(self, frame, track_id, kalman_filters):
        # Predict new position using Kalman filter
        kalman_filters[track_id].predict()
        predicted_center = kalman_filters[track_id].x[:2]
        
        size = self.occluded_tracks[track_id]['size']
        predicted_bbox = np.array([
            predicted_center[0] - size[0] / 2,
            predicted_center[1] - size[1] / 2,
            predicted_center[0] + size[0] / 2,
            predicted_center[1] + size[1] / 2
        ]).astype(int)
        
        # Try to re-identify the object
        reidentified, confidence = self.appearance_model.match(frame, predicted_bbox, track_id)
        
        if reidentified:
            color = (0, int(255 * confidence), int(255 * (1 - confidence)))
            cv2.rectangle(frame, (predicted_bbox[0], predicted_bbox[1]), 
                          (predicted_bbox[2], predicted_bbox[3]), color, 2)
            cv2.putText(frame, f"{track_id} (occluded, conf: {confidence:.2f})", 
                        (predicted_bbox[0], predicted_bbox[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Update Kalman filter with the re-identified position
            kalman_filters[track_id].update(predicted_center)
        
        # Update predicted position for next frame
        self.occluded_tracks[track_id]['center'] = predicted_center
        self.occluded_tracks[track_id]['size'] = size
        self.occluded_tracks[track_id]['confidence'] = confidence

    def add_new_occluded_track(self, track_id, bbox):
        center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        size = np.array([bbox[2] - bbox[0], bbox[3] - bbox[1]])
        self.occluded_tracks[track_id] = {'center': center, 'size': size, 'frames': 0, 'confidence': 1.0}

    def merge_nearby_tracks(self, merge_threshold=10):
        merged_tracks = set()
        for track_id1, track1 in self.occluded_tracks.items():
            if track_id1 in merged_tracks:
                continue
            for track_id2, track2 in self.occluded_tracks.items():
                if track_id1 != track_id2 and track_id2 not in merged_tracks:
                    distance = np.linalg.norm(track1['center'] - track2['center'])
                    if distance < merge_threshold:
                        # Merge tracks
                        new_center = (track1['center'] + track2['center']) / 2
                        new_size = (track1['size'] + track2['size']) / 2
                        new_confidence = max(track1['confidence'], track2['confidence'])
                        self.occluded_tracks[track_id1] = {'center': new_center, 'size': new_size, 
                                                           'frames': min(track1['frames'], track2['frames']),
                                                           'confidence': new_confidence}
                        merged_tracks.add(track_id2)
        
        # Remove merged tracks
        for track_id in merged_tracks:
            del self.occluded_tracks[track_id]