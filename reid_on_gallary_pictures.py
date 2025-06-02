import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
import json

def load_tracklets(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['tracklets']

def build_transforms(input_size=(256, 128)):
    """
    Build image transforms for ReID model
    """
    normalize_transform = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        normalize_transform
    ])
    return transform

def get_feature_vector(image, model, img_transforms, device):
    """
    Extract feature vector from an image
    """
    try:
        if isinstance(image, str):
            # If image is a path
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # If image is a numpy array (OpenCV format)
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
        input_tensor = img_transforms(image).unsqueeze(0).to(device)
        with torch.no_grad():
            feature = model(input_tensor).clone()
            return feature.cpu().numpy()[0]  # Return as numpy array
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def merge_traks_info(info1, info2):
    """
    Combine track information from two tracks, keeping the best detection for each frame
    
    Args:
        info1 (list): List of detections for first track
        info2 (list): List of detections for second track
        
    Returns:
        list: Merged list of detections
    """
    # Combine both track info
    all_detections = info1 + info2
    
    # Group detections by frame
    frame_detections = {}
    for detection in all_detections:
        frame = detection['frame']
        if frame not in frame_detections:
            frame_detections[frame] = detection
        else:
            # If we already have a detection for this frame, keep the one with higher confidence
            if detection['confidence'] > frame_detections[frame]['confidence']:
                frame_detections[frame] = detection
    
    # Convert back to list and sort by frame
    merged_tracklets = list(frame_detections.values())
    merged_tracklets.sort(key=lambda x: x['frame'])
    
    return merged_tracklets

def save_remapped_tracklets(remapped_tracklets, output_path):
    """
    Save remapped tracklets to a JSON file
    
    Args:
        remapped_tracklets (dict): Dictionary of remapped tracklets
        output_path (str): Path to save the JSON file
    """
    with open(output_path, 'w') as f:
        json.dump({'tracklets': remapped_tracklets}, f, indent=2)
    print(f"Remapped tracklets saved to {output_path}")

def process_existing_gallery_with_reid(gallery_base_dir, video_name, similarity_threshold=0.85, clip_model='ViT-L/14'):
    """
    Process an existing gallery of pictures created by create_a_gallery_pictures
    and perform ReID to merge similar identities using CLIP model
    
    Args:
        gallery_base_dir (str): Base directory where gallery images are stored
        video_name (str): Name of the video/subfolder where images are saved
        similarity_threshold (float, optional): Threshold for ReID similarity
        clip_model (str, optional): CLIP model variant to use
    
    Returns:
        dict: Dictionary mapping track IDs to lists of saved image paths
        dict: Dictionary mapping original track IDs to merged ReID IDs
    """
    # Path to the gallery directory for this video
    gallery_dir = os.path.join(gallery_base_dir, video_name)
    
    if not os.path.exists(gallery_dir):
        print(f"Error: Gallery directory {gallery_dir} does not exist")
        return {}, {}
    
    # Create directory for features
    features_dir = f'features/{video_name}'
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)
    
    # Import CLIP
    try:
        import clip
        print(f"Loading CLIP model: {clip_model}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load(clip_model, device=device)
        print(f"CLIP model {clip_model} loaded successfully on {device}")
    except Exception as e:
        print(f"Error loading CLIP model: {e}")
        return {}, {}
    
    # Initialize ReID database
    reid_features = []  # List of feature vectors
    reid_ids = []       # List of assigned IDs
    next_reid_id = 0    # Next ID to assign
    
    # Dictionary to map original track IDs to ReID IDs
    track_to_reid_mapping = {}
    
    # Dictionary to store gallery images
    gallery_images = {}
    
    # Dictionary to store all features by track ID
    all_features = {}
    
    # Get all track directories in the gallery
    track_dirs = [d for d in os.listdir(gallery_dir) if os.path.isdir(os.path.join(gallery_dir, d))]
    
    for track_dir in tqdm(track_dirs, desc="Processing tracks"):
        user_id = track_dir  # The directory name is the track ID
        track_path = os.path.join(gallery_dir, track_dir)
        
        # Get all images for this track
        image_files = [f for f in os.listdir(track_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        user_features = []
        user_images = []
        
        for image_file in tqdm(image_files, total=len(image_files)):
            image_path = os.path.join(track_path, image_file)
            
            try:
                # Load and preprocess image for CLIP
                image = Image.open(image_path).convert("RGB")
                image_input = preprocess(image).unsqueeze(0).to(device)
                
                # Extract features using CLIP
                with torch.no_grad():
                    image_features = model.encode_image(image_input)
                    # Normalize features
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    user_features.append(image_features.cpu().numpy()[0])
                
                # Store the image path and image
                img = cv2.imread(image_path)
                user_images.append((image_path, img))
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        
        # Store all features for this track
        if user_features:
            all_features[user_id] = np.array(user_features)
        
        # Perform ReID comparison if we have features
        if user_features:
            # Average the features for this track
            avg_feature = np.mean(np.array(user_features), axis=0)
            
            # Compare with existing ReID database
            assigned_id = -1
            
            if reid_features:
                # Calculate similarities with all existing IDs
                similarities = []
                for feat in reid_features:
                    # Compute cosine similarity
                    similarity = np.dot(avg_feature, feat) / (np.linalg.norm(avg_feature) * np.linalg.norm(feat))
                    similarities.append(similarity)
                
                # Find best match
                best_match_idx = np.argmax(similarities)
                best_similarity = similarities[best_match_idx]
                
                if best_similarity > similarity_threshold:
                    assigned_id = reid_ids[best_match_idx]
                    print(f"Track {user_id} matched with ReID {assigned_id} (similarity: {best_similarity:.4f})")
            
            # If no match found, assign new ID
            if assigned_id == -1:
                assigned_id = next_reid_id
                next_reid_id += 1
                reid_features.append(avg_feature)
                reid_ids.append(assigned_id)
                print(f"New ReID {assigned_id} assigned to track {user_id}")
            
            # Store the mapping
            track_to_reid_mapping[user_id] = assigned_id
            
            # Create directory for this ReID
            reid_dir = os.path.join(gallery_dir, f"reid_{assigned_id}")
            if not os.path.exists(reid_dir):
                os.makedirs(reid_dir)
            
            # Copy the images to the ReID directory
            for image_path, _ in user_images:
                # Update path to use ReID
                new_path = os.path.join(reid_dir, f"track{user_id}_{os.path.basename(image_path)}")
                # Copy the image (don't need to read/write, just copy the file)
                import shutil
                shutil.copy2(image_path, new_path)
                
                # Add to gallery images
                if assigned_id not in gallery_images:
                    gallery_images[assigned_id] = []
                gallery_images[assigned_id].append(new_path)
        else:
            # No ReID, just use original track ID
            track_to_reid_mapping[user_id] = user_id
            
            # Add existing images to gallery
            if user_id not in gallery_images:
                gallery_images[user_id] = []
            for image_path, _ in user_images:
                gallery_images[user_id].append(image_path)
    
    print(f'Gallery processing completed. Total unique identities after ReID: {len(set(track_to_reid_mapping.values()))}')
    
    # Save all features to disk
    print(f"Saving {len(all_features)} feature sets to {features_dir}")
    for track_id, features in all_features.items():
        feature_file = os.path.join(features_dir, f"track_{track_id}.npy")
        np.save(feature_file, features)
    
    # Save the mapping to a file
    mapping_file = os.path.join(gallery_dir, 'track_to_reid_mapping.json')
    with open(mapping_file, 'w') as f:
        json.dump(track_to_reid_mapping, f, indent=2)
    
    return gallery_images, track_to_reid_mapping

def process_extracted_features_reid(features_dir, tracklets_path, output_path, similarity_threshold=0.85):
    """
    Process already extracted features to perform ReID and merge similar identities
    
    Args:
        features_dir (str): Directory containing extracted features (.npy files)
        tracklets_path (str): Path to the tracklets JSON file
        output_path (str): Path to save the remapped tracklets
        similarity_threshold (float, optional): Threshold for ReID similarity
    
    Returns:
        dict: Dictionary mapping original track IDs to merged ReID IDs
    """
    print(f"Processing extracted features from {features_dir} for ReID...")
    
    # Check if features directory exists
    if not os.path.exists(features_dir):
        print(f"Error: Features directory {features_dir} does not exist")
        return {}
    
    # Load all feature files
    feature_files = [f for f in os.listdir(features_dir) if f.endswith('.npy')]
    if not feature_files:
        print(f"Error: No feature files found in {features_dir}")
        return {}
    
    print(f"Found {len(feature_files)} feature files")
    
    # Initialize ReID database
    reid_features = []  # List of feature vectors (mean of each track's features)
    reid_ids = []       # List of assigned IDs
    next_reid_id = 0    # Next ID to assign
    
    # Dictionary to map original track IDs to ReID IDs
    track_to_reid_mapping = {}
    
    # Load and process each feature file
    for feature_file in tqdm(feature_files, desc="Processing features"):
        # Extract track ID from filename (assuming format "track_ID.npy")
        track_id = feature_file.split('_')[1].split('.')[0]
        
        # Load features
        feature_path = os.path.join(features_dir, feature_file)
        try:
            features = np.load(feature_path)
            
            # Skip if no features
            if features.size == 0:
                print(f"Warning: Empty features for track {track_id}")
                continue
                
            # Average the features for this track
            avg_feature = np.mean(features, axis=0)
            
            # Compare with existing ReID database
            assigned_id = -1
            
            if reid_features:
                # Calculate similarities with all existing IDs
                similarities = []
                for feat in reid_features:
                    # Compute cosine similarity
                    similarity = np.dot(avg_feature, feat) / (np.linalg.norm(avg_feature) * np.linalg.norm(feat))
                    similarities.append(similarity)
                
                # Find best match
                best_match_idx = np.argmax(similarities)
                best_similarity = similarities[best_match_idx]
                
                if best_similarity > similarity_threshold:
                    assigned_id = reid_ids[best_match_idx]
                    print(f"Track {track_id} matched with ReID {assigned_id} (similarity: {best_similarity:.4f})")
            
            # If no match found, assign new ID
            if assigned_id == -1:
                assigned_id = next_reid_id
                next_reid_id += 1
                reid_features.append(avg_feature)
                reid_ids.append(assigned_id)
                print(f"New ReID {assigned_id} assigned to track {track_id}")
            
            # Store the mapping
            track_to_reid_mapping[track_id] = str(assigned_id)  # Convert to string to match tracklets format
            
        except Exception as e:
            print(f"Error processing features for track {track_id}: {e}")
    
    print(f"ReID processing completed. {len(feature_files)} tracks mapped to {len(set(track_to_reid_mapping.values()))} unique identities")
    
    # Save the mapping to a file
    mapping_file = os.path.join(os.path.dirname(features_dir), 'track_to_reid_mapping.json')
    with open(mapping_file, 'w') as f:
        json.dump(track_to_reid_mapping, f, indent=2)
    print(f"Track to ReID mapping saved to {mapping_file}")
    
    # Apply the mapping to tracklets
    if tracklets_path and output_path:
        print("\nApplying ReID mapping to tracklets...")
        
        # Load original tracklets
        original_tracklets = load_tracklets(tracklets_path)
        
        # Create remapped tracklets
        remapped_tracklets = {}
        for original_id, track_info in original_tracklets.items():
            # Get the ReID assigned to this track (or keep original if not in mapping)
            reid = track_to_reid_mapping.get(original_id, original_id)
            
            # Add to remapped tracklets, merging if needed
            if reid not in remapped_tracklets:
                remapped_tracklets[reid] = track_info
            else:
                # Merge track info for the same ReID
                remapped_tracklets[reid] = merge_traks_info(remapped_tracklets[reid], track_info)
        
        save_remapped_tracklets(remapped_tracklets, output_path)
        
        print(f"Original tracklets had {len(original_tracklets)} tracks")
        print(f"Merged tracklets have {len(remapped_tracklets)} tracks")
        print(f"Merged tracklets saved to {output_path}")
    
    return track_to_reid_mapping

if __name__ == "__main__":
    # Configuration parameters
    video_path = "videos/2.mp4"
    tracklets_path = "results/2_clear.json"
    video_name = "2"
    output_path = f"results/{video_name}_reid.json"
    match_threshold = 0.987
    use_extracted_features = False  # Set to True to use already extracted features
    clip_model = 'ViT-L/14'  # CLIP model variant to use
    
    if use_extracted_features:
        # Use already extracted features
        features_dir = f'features/{video_name}'
        print(f"Using pre-extracted features from {features_dir}")
        track_to_reid_mapping = process_extracted_features_reid(
            features_dir, 
            tracklets_path, 
            output_path, 
            similarity_threshold=match_threshold
        )
    else:
        # Extract features and perform ReID using CLIP
        gallery_base_dir = "gallery_images"
        print(f"Processing gallery images with CLIP {clip_model}")
        gallery_images, track_to_reid_mapping = process_existing_gallery_with_reid(
            gallery_base_dir, 
            video_name, 
            similarity_threshold=match_threshold, 
            clip_model=clip_model
        )
        
        # Create merged tracklets based on ReID mapping
        print("\nCreating merged tracklets based on ReID mapping...")
        
        # Load original tracklets
        original_tracklets = load_tracklets(tracklets_path)
        
        # Create remapped tracklets
        remapped_tracklets = {}
        for original_id, track_info in original_tracklets.items():
            # Get the ReID assigned to this track (or keep original if not in mapping)
            reid = track_to_reid_mapping.get(original_id, original_id)
            
            # Add to remapped tracklets, merging if needed
            if reid not in remapped_tracklets:
                remapped_tracklets[reid] = track_info
            else:
                # Merge track info for the same ReID
                remapped_tracklets[reid] = merge_traks_info(remapped_tracklets[reid], track_info)
        
        save_remapped_tracklets(remapped_tracklets, output_path)
        
        print(f"Original tracklets had {len(original_tracklets)} tracks")
        print(f"Merged tracklets have {len(remapped_tracklets)} tracks")
        print(f"Merged tracklets saved to {output_path}")