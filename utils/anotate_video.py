import cv2
import numpy as np
 
 
# Define a function to convert detections to SORT format.
def convert_detections(detections, threshold, classes):
    # Get the bounding boxes, labels and scores from the detections dictionary.
    boxes = detections["boxes"].cpu().numpy()
    labels = detections["labels"].cpu().numpy()
    scores = detections["scores"].cpu().numpy()
    lbl_mask = np.isin(labels, classes)
    scores = scores[lbl_mask]
    # Filter out low confidence scores and non-person classes.
    mask = scores > threshold
    boxes = boxes[lbl_mask][mask]
    scores = scores[mask]
    labels = labels[lbl_mask][mask]
 
 
    # Convert boxes to [x1, y1, w, h, score] format.
    final_boxes = []
    for i, box in enumerate(boxes):
        # Append ([x, y, w, h], score, label_string).
        final_boxes.append(
            (
                [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                scores[i],
                str(labels[i])
            )
        )
 
    return final_boxes


# Function for bounding box and ID annotation.
def annotate(tracks, frame):
    """
    Draw bounding boxes and labels on the frame for each track.
    Uses different colors for different track IDs.
    
    Args:
        tracks: List of DeepSort Track objects
        frame: The frame to annotate
        
    Returns:
        Annotated frame with colored boxes and ID labels
    """
    for track in tracks:
        if not track.is_confirmed():
            continue
            
        # Get track ID and bounding box coordinates
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        
        # Generate a unique color for this track ID
        # Use a hash function to ensure consistent colors for the same ID
        color_hash = hash(str(track_id)) % 0xFFFFFF
        color = (color_hash & 0xFF, (color_hash >> 8) & 0xFF, (color_hash >> 16) & 0xFF)
        
        # Draw bounding box with the unique color
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color=color,
            thickness=2
        )
        
        # Create label with ID
        label = f"{track_id}"
        
        # Calculate text size for better positioning
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # Draw background rectangle for text
        cv2.rectangle(
            frame,
            (x1, y1 - text_size[1] - 10),
            (x1 + text_size[0] + 10, y1),
            color,
            -1  # Fill the rectangle
        )
        
        # Draw ID text with white color for better visibility
        cv2.putText(
            frame,
            label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),  # White text
            2,
            lineType=cv2.LINE_AA
        )
        
    return frame