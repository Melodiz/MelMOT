import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import os

# You'll need to install torchreid first:
# pip install torchreid

def load_tracklets(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['tracklets']

def load_reid_model():
    """
    Load a pre-trained ReID model
    """
    try:
        import torchreid
        from torchreid.utils import FeatureExtractor
        
        # Check if CUDA is available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize feature extractor
        # Using OSNet, a lightweight and effective ReID model
        extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path='',  # Use the default pre-trained weights
            device=device
        )
        
        return extractor
    except ImportError:
        print("torchreid not installed. Please install it with: pip install torchreid")
        return None

def choose_top_detections(track_info, gallery_size, threshold=0.35):
    # filter out detections with confidence less than threshold
    detections = [d for d in track_info if d['confidence'] > threshold]
    # choose random detections if there are less than gallery_size
    return np.random.choice(detections, size=min(gallery_size, len(detections)), replace=False)

def create_a_gallery(video_path, tracklets_path, video_name=None, gallery_size=10):
    tracklets_data = load_tracklets(tracklets_path)
    # for each user, choose the top <gallery_size detections
    total_gallery_size = 0
    gallery_tracklets = {}
    for user_id, track_info in tqdm(tracklets_data.items(), desc="Creating gallery"):
        gallery_tracklets[user_id] = choose_top_detections(track_info, gallery_size)
        total_gallery_size += len(gallery_tracklets[user_id])
    del(tracklets_data) # free up memory
    print('gallery_tracklets successfully created. Total gallery size:', total_gallery_size)
    print('loading a ReID model...')
    # for each id, extract features and save them
    feature_extractor = load_reid_model()
    if feature_extractor is None:
        print("Feature extraction will be skipped.")
    else:
        print("Model for feature extraction loaded successfully.")
    
    # Open the video
    cap = cv2.VideoCapture(video_path)

    galary_features = {}
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []
    for user_id, detections in tqdm(gallery_tracklets.items(), desc="Extracting features for gallery"):
        galary_features[user_id] = []
        for detection in detections:
            frame_number = detection['frame']
            bbox = detection['bbox']
            
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            # Read the frame
            success, frame = cap.read()
            if not success:
                print(f"Error: Could not read frame {frame_number}")
                continue
            
            # Extract the bounding box region
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            cropped_img = frame[y1:y2, x1:x2]
            
            # Convert from BGR to RGB for feature extraction
            cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
            
            # Resize image to match model's expected input size
            resized_img = cv2.resize(cropped_img_rgb, (128, 256))
            
            # Extract features
            feature_tensor = feature_extractor(resized_img)
            features_np = feature_tensor.cpu().numpy()[0]  # Get the first batch item
            galary_features[user_id].append(features_np)
        # save the features of user to a file
        # check if the directory exists, otherwise create it
        if not os.path.exists(f'gallery/{video_name}'):
            os.makedirs(f'gallery/{video_name}')
        np.save(f'gallery/{video_name}{user_id}.npy', np.array(galary_features[user_id]))
    print('Gallery features successfully saved.')
    return galary_features

def create_a_gallery_pictures(video_path, tracklets_path, video_name=None, gallery_size=32):
    """
    Create a gallery of pictures for each track ID by cropping from video frames
    
    Args:
        video_path (str): Path to the video file
        tracklets_path (str): Path to the tracklets JSON file
        video_name (str, optional): Name of the video/subfolder to save images
        gallery_size (int, optional): Number of images to save per track ID
    
    Returns:
        dict: Dictionary mapping track IDs to lists of saved image paths
    """
    tracklets_data = load_tracklets(tracklets_path)
    # For each user, choose the top <gallery_size> detections
    total_gallery_size = 0
    gallery_tracklets = {}
    for user_id, track_info in tqdm(tracklets_data.items(), desc="Selecting detections"):
        gallery_tracklets[user_id] = choose_top_detections(track_info, gallery_size)
        total_gallery_size += len(gallery_tracklets[user_id])
    del(tracklets_data)  # Free up memoryx
    print(f'Gallery tracklets successfully created. Total gallery size: {total_gallery_size}')
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    gallery_images = {}
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return {}
    
    # Create base directory for gallery images
    base_dir = f'gallery_images/{video_name}'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Get video dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    for user_id, detections in tqdm(gallery_tracklets.items(), desc="Saving gallery images"):
        # Create directory for this user ID
        user_dir = os.path.join(base_dir, str(user_id))
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)
        
        gallery_images[user_id] = []
        
        for i, detection in enumerate(detections):
            frame_number = detection['frame']
            bbox = detection['bbox']
            
            # Set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            # Read the frame
            success, frame = cap.read()
            if not success:
                print(f"Error: Could not read frame {frame_number}")
                continue
            
            # Extract the bounding box region with validation
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            
            # Validate and adjust bounding box coordinates
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
            
            # Check if the bounding box is valid
            if x2 <= x1 or y2 <= y1:
                print(f"Warning: Invalid bounding box for user {user_id}, detection {i}: {bbox}")
                continue
            
            cropped_img = frame[y1:y2, x1:x2]
            
            # Check if the cropped image is empty
            if cropped_img.size == 0:
                print(f"Warning: Empty crop for user {user_id}, detection {i}: {bbox}")
                continue
            
            # Save the cropped image
            image_path = os.path.join(user_dir, f"{i:03d}_frame{frame_number}.jpg")
            cv2.imwrite(image_path, cropped_img)
            gallery_images[user_id].append(image_path)
    
    cap.release()
    print(f'Gallery images successfully saved to {base_dir}')
    return gallery_images

if __name__ == "__main__":
    # Load tracklets
    video_path = 'videos/a.mp4'
    tracklets_path = 'results/2_clear.json'
    # create_a_gallery(video_path, tracklets_path, video_name='s2/', gallery_size=32)
    create_a_gallery_pictures(video_path, tracklets_path, video_name='2/', gallery_size=32)