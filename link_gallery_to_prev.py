import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
import json
import sys
from utils.visualize_links import visualize_gallery_links


def load_tracklets(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['tracklets']

def load_reid_model(reid_model_name='resnet50_fc512'):
    """
    Load a pre-trained ReID model
    """
    try:
        import torchreid
        from torchreid.utils import FeatureExtractor
        
        # Check if CUDA is available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize feature extractor
        extractor = FeatureExtractor(
            model_name=reid_model_name,
            model_path='', # use default pre-trained model
            device=device, 
        )
        print("ReID model loaded successfully.")
        return extractor
    except Exception as e:
        print(f"Error loading ReID model: {e}")
        return None
    
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
    
def save_remapped_tracklets(remapped_tracklets, output_path):
    with open(output_path, 'w') as f:
        json.dump({'tracklets': remapped_tracklets}, f, indent=2)
    print(f"Remapped tracklets saved to {output_path}")

def link_gallery_force(base_gallery_path, query_gallery_path, output_path=None, similarity_threshold=0.7, 
                      use_distance=True, top_k=5, reid_model_name='resnet50_fc512'):
    """
    For each identity in the query gallery, find the best match in the base gallery.
    Uses a voting mechanism based on top-k nearest neighbors for each query image.
    
    Args:
        base_gallery_path (str): Path to the base gallery directory
        query_gallery_path (str): Path to the query gallery directory
        output_path (str, optional): Path to save the links JSON file
        similarity_threshold (float, optional): Threshold for logging high confidence matches
        use_distance (bool): Use L2 distance instead of cosine similarity
        top_k (int): Number of top neighbors to consider for voting
    
    Returns:
        dict: Dictionary mapping query IDs to base IDs
    """
    print(f"Linking galleries: {query_gallery_path} -> {base_gallery_path}")
    print(f'Loading ReID model and transforms...")')
    
    # Load ReID model and transforms
    reid_model = load_reid_model(reid_model_name=reid_model_name)
    reid_transforms = build_transforms()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if reid_model is None:
        print("Error: Could not load ReID model")
        return {}
    
    # Get all person directories in both galleries
    base_dirs = [d for d in os.listdir(base_gallery_path) 
                if os.path.isdir(os.path.join(base_gallery_path, d))]
    query_dirs = [d for d in os.listdir(query_gallery_path) 
                 if os.path.isdir(os.path.join(query_gallery_path, d))]
    
    print(f"Found {len(base_dirs)} identities in base gallery")
    print(f"Found {len(query_dirs)} identities in query gallery")
    
    # Extract features for base gallery
    base_features = {}  # Dictionary to store all base features: {base_id: [feature1, feature2, ...]}
    print("Extracting features for base gallery...")
    for person_dir in tqdm(base_dirs):
        person_path = os.path.join(base_gallery_path, person_dir)
        person_id = person_dir.replace("reid_", "")  # Handle both "reid_X" and "X" formats
        
        # Get all images for this person
        images = [os.path.join(person_path, f) for f in os.listdir(person_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not images:
            continue
            
        # Extract features for each image
        features = []
        for img_path in images:
            feature = get_feature_vector(img_path, reid_model, reid_transforms, device)
            if feature is not None:
                features.append(feature)
        
        if features:
            base_features[person_id] = features
    
    # Flatten base features for easier nearest neighbor search
    all_base_features = []
    base_id_map = []  # Maps index in all_base_features to base_id
    
    for base_id, feature_list in base_features.items():
        for feature in feature_list:
            all_base_features.append(feature)
            base_id_map.append(base_id)
    
    all_base_features = np.array(all_base_features)
    
    # Extract features for query gallery and find matches using voting
    links = {}
    print("Extracting features for query gallery and finding matches...")
    for person_dir in tqdm(query_dirs):
        person_path = os.path.join(query_gallery_path, person_dir)
        person_id = person_dir.replace("reid_", "")  # Handle both "reid_X" and "X" formats
        
        # Get all images for this person
        images = [os.path.join(person_path, f) for f in os.listdir(person_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not images:
            continue
            
        # Extract features for each image
        query_features = []
        for img_path in images:
            feature = get_feature_vector(img_path, reid_model, reid_transforms, device)
            if feature is not None:
                query_features.append(feature)
        
        if not query_features:
            continue
        
        # For each query feature, find top-k nearest neighbors
        votes = {}  # Dictionary to count votes for each base_id
        total_votes = 0
        
        for query_feature in query_features:
            # Compute distances/similarities to all base features
            if use_distance:
                # L2 distance (lower is better)
                distances = np.linalg.norm(all_base_features - query_feature, axis=1)
                # Get indices of top-k nearest neighbors (smallest distances)
                top_indices = np.argsort(distances)[:top_k]
            else:
                # Cosine similarity (higher is better)
                similarities = np.dot(all_base_features, query_feature) / (
                    np.linalg.norm(all_base_features, axis=1) * np.linalg.norm(query_feature))
                # Get indices of top-k nearest neighbors (largest similarities)
                top_indices = np.argsort(similarities)[-top_k:]
            
            # Vote for base_ids of top-k neighbors
            for idx in top_indices:
                base_id = base_id_map[idx]
                votes[base_id] = votes.get(base_id, 0) + 1
                total_votes += 1
        
        # Find the base_id with the most votes
        if votes:
            best_match_id = max(votes.items(), key=lambda x: x[1])[0]
            vote_count = votes[best_match_id]
            confidence = vote_count / total_votes  # Percentage of votes for the winning base_id
            
            # Store the match with confidence
            links[person_id] = {
                "match_id": best_match_id,
                "similarity": float(confidence),  # Use confidence as similarity
                "vote_count": vote_count,
                "total_votes": total_votes
            }
            
            # Log match information
            confidence_level = "HIGH" if confidence > similarity_threshold else "LOW"
            print(f"Linked {person_id} -> {best_match_id} (confidence: {confidence:.4f}, votes: {vote_count}/{total_votes}, level: {confidence_level})")
    
    # Save links to file if output_path is provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(links, f, indent=2)
        print(f"Links saved to {output_path}")
    
    return links

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

def apply_gallery_links(tracklets_path, links, output_path):
    """
    Apply gallery links to tracklets, remapping IDs based on the links
    
    Args:
        tracklets_path (str): Path to the tracklets JSON file
        links (dict): Dictionary mapping query IDs to base IDs
        output_path (str): Path to save the remapped tracklets
    """
    # Load tracklets
    original_tracklets = load_tracklets(tracklets_path)
    
    # Create remapped tracklets
    remapped_tracklets = {}
    for original_id, track_info in original_tracklets.items():
        # Get the linked ID (or keep original if not in links)
        if original_id in links:
            linked_id = links[original_id]["match_id"]
        else:
            linked_id = original_id
        
        # Add to remapped tracklets
        if linked_id not in remapped_tracklets:
            remapped_tracklets[linked_id] = track_info
        else:
            # Merge track info for the same linked ID
            remapped_tracklets[linked_id] = merge_traks_info(remapped_tracklets[linked_id], track_info)
    
    # Save the remapped tracklets
    save_remapped_tracklets(remapped_tracklets, output_path)
    
    print(f"Original tracklets had {len(original_tracklets)} tracks")
    print(f"Linked tracklets have {len(remapped_tracklets)} tracks")
    print(f"Linked tracklets saved to {output_path}")

def link_gallery_clip(base_gallery_path, query_gallery_path, output_path=None, similarity_threshold=0.7, 
                     use_distance=True, top_k=5, clip_model='ViT-B/16'):
    """
    For each identity in the query gallery, find the best match in the base gallery using CLIP features.
    Uses a voting mechanism based on top-k nearest neighbors for each query image.
    
    Args:
        base_gallery_path (str): Path to the base gallery directory
        query_gallery_path (str): Path to the query gallery directory
        output_path (str, optional): Path to save the links JSON file
        similarity_threshold (float, optional): Threshold for logging high confidence matches
        use_distance (bool): Use L2 distance instead of cosine similarity
        top_k (int): Number of top neighbors to consider for voting
        clip_model (str): CLIP model variant to use
    
    Returns:
        dict: Dictionary mapping query IDs to base IDs
    """
    print(f"Linking galleries using CLIP {clip_model}: {query_gallery_path} -> {base_gallery_path}")
    
    # Import CLIP
    try:
        import clip
        from PIL import Image
    except ImportError:
        print("Error: CLIP not installed. Please install with: pip install git+https://github.com/openai/CLIP.git")
        return {}
    
    # Load CLIP model
    print(f"Loading CLIP model: {clip_model}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model, preprocess = clip.load(clip_model, device=device)
        print(f"CLIP model {clip_model} loaded successfully on {device}")
    except Exception as e:
        print(f"Error loading CLIP model: {e}")
        return {}
    
    # Get all person directories in both galleries
    base_dirs = [d for d in os.listdir(base_gallery_path) 
                if os.path.isdir(os.path.join(base_gallery_path, d))]
    query_dirs = [d for d in os.listdir(query_gallery_path) 
                 if os.path.isdir(os.path.join(query_gallery_path, d))]
    
    print(f"Found {len(base_dirs)} identities in base gallery")
    print(f"Found {len(query_dirs)} identities in query gallery")
    
    # Extract features for base gallery
    base_features = {}  # Dictionary to store all base features: {base_id: [feature1, feature2, ...]}
    print("Extracting CLIP features for base gallery...")
    for person_dir in tqdm(base_dirs):
        person_path = os.path.join(base_gallery_path, person_dir)
        person_id = person_dir.replace("reid_", "")  # Handle both "reid_X" and "X" formats
        
        # Get all images for this person
        images = [os.path.join(person_path, f) for f in os.listdir(person_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not images:
            continue
            
        # Extract features for each image
        features = []
        for img_path in tqdm(images, total=len(images)):
            try:
                # Load and preprocess image
                image = Image.open(img_path).convert("RGB")
                image_input = preprocess(image).unsqueeze(0).to(device)
                
                # Extract features
                with torch.no_grad():
                    image_features = model.encode_image(image_input)
                    # Normalize features
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    features.append(image_features.cpu().numpy()[0])
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        if features:
            base_features[person_id] = features
    
    # Flatten base features for easier nearest neighbor search
    all_base_features = []
    base_id_map = []  # Maps index in all_base_features to base_id
    
    for base_id, feature_list in base_features.items():
        for feature in feature_list:
            all_base_features.append(feature)
            base_id_map.append(base_id)
    
    all_base_features = np.array(all_base_features)
    
    # Extract features for query gallery and find matches using voting
    links = {}
    print("Extracting CLIP features for query gallery and finding matches...")
    for person_dir in tqdm(query_dirs):
        person_path = os.path.join(query_gallery_path, person_dir)
        person_id = person_dir.replace("reid_", "")  # Handle both "reid_X" and "X" formats
        
        # Get all images for this person
        images = [os.path.join(person_path, f) for f in os.listdir(person_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not images:
            continue
            
        # Extract features for each image
        query_features = []
        for img_path in tqdm(images, total=len(images)):
            try:
                # Load and preprocess image
                image = Image.open(img_path).convert("RGB")
                image_input = preprocess(image).unsqueeze(0).to(device)
                
                # Extract features
                with torch.no_grad():
                    image_features = model.encode_image(image_input)
                    # Normalize features
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    query_features.append(image_features.cpu().numpy()[0])
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        if not query_features:
            continue
        
        # For each query feature, find top-k nearest neighbors
        votes = {}  # Dictionary to count votes for each base_id
        total_votes = 0
        
        for query_feature in query_features:
            # Compute distances/similarities to all base features
            if use_distance:
                # L2 distance (lower is better)
                distances = np.linalg.norm(all_base_features - query_feature, axis=1)
                # Get indices of top-k nearest neighbors (smallest distances)
                top_indices = np.argsort(distances)[:top_k]
            else:
                # Cosine similarity (higher is better)
                similarities = np.dot(all_base_features, query_feature) / (
                    np.linalg.norm(all_base_features, axis=1) * np.linalg.norm(query_feature))
                # Get indices of top-k nearest neighbors (largest similarities)
                top_indices = np.argsort(similarities)[-top_k:]
            
            # Vote for base_ids of top-k neighbors
            for idx in top_indices:
                base_id = base_id_map[idx]
                votes[base_id] = votes.get(base_id, 0) + 1
                total_votes += 1
        
        # Find the base_id with the most votes
        if votes:
            best_match_id = max(votes.items(), key=lambda x: x[1])[0]
            vote_count = votes[best_match_id]
            confidence = vote_count / total_votes  # Percentage of votes for the winning base_id
            
            # Store the match with confidence
            links[person_id] = {
                "match_id": best_match_id,
                "similarity": float(confidence),  # Use confidence as similarity
                "vote_count": vote_count,
                "total_votes": total_votes
            }
            
            # Log match information
            confidence_level = "HIGH" if confidence > similarity_threshold else "LOW"
            print(f"Linked {person_id} -> {best_match_id} (confidence: {confidence:.4f}, votes: {vote_count}/{total_votes}, level: {confidence_level})")
    
    # Save links to file if output_path is provided
    if output_path:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
            
        with open(output_path, 'w') as f:
            json.dump(links, f, indent=2)
        print(f"Links saved to {output_path}")
    
    return links

if __name__ == "__main__":
    # Parameters for gallery linking
    base_gallery_path = "gallery_images/improved_a"
    query_gallery_path = "gallery_images/improved_b"
    links_output_path = "results/gallery_links.json"
    tracklets_path = "results/bvid2.json"
    linked_tracklets_output_path = "results/b_linked_tracks.json"
    similarity_threshold = 0.6  # Threshold for confidence (percentage of votes)
    reid_model_name = 'osnet_x1_0'
    
    # Link galleries with ReID-based matching
    links = link_gallery_force(
        base_gallery_path, 
        query_gallery_path, 
        output_path=links_output_path,
        similarity_threshold=similarity_threshold,
        use_distance=True,  # Use L2 distance for nearest neighbor search
        top_k=5,           # Consider top 5 neighbors for voting
        reid_model_name=reid_model_name  # Use osnet_x1_0 model
    )
    
    # Visualize gallery links
    visualize_gallery_links( 
        links,
        base_gallery_path,
        query_gallery_path,
        output_path="results/gallery_links_visualization.jpg",
        query_samples=3,
        base_samples=3
    )
    
    # Apply links to tracklets
    apply_gallery_links(tracklets_path, links, linked_tracklets_output_path)