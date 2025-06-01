import numpy as np
import cv2
import json
import os
def load_tracklets(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['tracklets']

def load_gallary(gallary_path='features/'):
    gallary_features = {}
    for filename in os.listdir(gallary_path):
        if filename.endswith('.npy'):
            track_id = int(filename.split('.')[0].split('_')[-1])
            features = np.load(os.path.join(gallary_path, filename))
            gallary_features[track_id] = features
    return gallary_features

def compute_similarity(features1, features2, method='cosine'):
    """
    Compute similarity between two sets of features
    
    Args:
        features1 (np.ndarray): Features of the first track
        features2 (np.ndarray): Features of the second track
        method (str): Similarity method ('cosine', 'euclidean')
    
    Returns:
        float: Similarity score
    """
    # If features are 2D arrays (multiple feature vectors per track)
    if len(features1.shape) > 1 and len(features2.shape) > 1:
        # Compute all pairwise similarities
        similarities = []
        for feat1 in features1:
            for feat2 in features2:
                if method == 'cosine':
                    # Cosine similarity (1 = identical, 0 = orthogonal, -1 = opposite)
                    similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
                else:  # euclidean
                    # Convert Euclidean distance to similarity (1 = identical, 0 = very different)
                    distance = np.linalg.norm(feat1 - feat2)
                    similarity = 1 / (1 + distance)
                similarities.append(similarity)
        
        # Return the maximum similarity (best match between any pair)
        return max(similarities)
    else:
        # Single feature vector per track
        if method == 'cosine':
            return np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
        else:  # euclidean
            distance = np.linalg.norm(features1 - features2)
            return 1 / (1 + distance)

def find_matches(gallery, threshold=0.85, method='cosine'):
    """
    Find matches between all tracks in the gallery
    
    Args:
        gallery (dict): Dictionary of track features {track_id: features}
        threshold (float): Similarity threshold (0-1)
        method (str): Similarity method ('cosine', 'euclidean')
    
    Returns:
        list: List of tuples (track_id1, track_id2, similarity)
    """
    matches = []
    track_ids = list(gallery.keys())
    
    # Compare each pair of tracks
    for i in range(len(track_ids)):
        for j in range(i+1, len(track_ids)):
            id1 = track_ids[i]
            id2 = track_ids[j]
            
            # Compute similarity
            similarity = compute_similarity(gallery[id1], gallery[id2], method)
            
            # If similarity is above threshold, consider it a match
            if similarity >= threshold:
                matches.append((id1, id2, similarity))
    
    # Sort matches by similarity (descending)
    matches.sort(key=lambda x: x[2], reverse=True)
    
    return matches

def cluster_identities(gallery, threshold=0.85, method='cosine'):
    """
    Cluster track IDs into identity groups using an iterative approach
    
    Args:
        gallery (dict): Dictionary of track features {track_id: features}
        threshold (float): Similarity threshold (0-1)
        method (str): Similarity method ('cosine', 'euclidean')
    
    Returns:
        list: List of identity clusters (each cluster is a list of track IDs)
    """
    # Initialize each track as its own cluster
    clusters = {i: [track_id] for i, track_id in enumerate(gallery.keys())}
    
    # Create a mapping from track_id to cluster_id
    track_to_cluster = {track_id: i for i, track_ids in clusters.items() for track_id in track_ids}
    
    # Create merged galleries (initially just the original galleries)
    merged_galleries = {i: gallery[track_id] for i, track_id in enumerate(gallery.keys())}
    
    # Keep track of whether we found any matches in the current iteration
    matches_found = True
    iteration = 0
    
    while matches_found:
        iteration += 1
        print(f"Iteration {iteration}, current clusters: {len(clusters)}")
        
        # Find matches between current merged galleries
        matches_found = False
        cluster_ids = list(merged_galleries.keys())
        
        # Compare each pair of clusters
        for i in range(len(cluster_ids)):
            for j in range(i+1, len(cluster_ids)):
                cluster_id1 = cluster_ids[i]
                cluster_id2 = cluster_ids[j]
                
                # Skip if these clusters have already been merged
                if cluster_id1 not in merged_galleries or cluster_id2 not in merged_galleries:
                    continue
                
                # Compute similarity between merged galleries
                similarity = compute_similarity(
                    merged_galleries[cluster_id1], 
                    merged_galleries[cluster_id2], 
                    method
                )
                
                # If similarity is above threshold, merge the clusters
                if similarity >= threshold:
                    matches_found = True
                    
                    # Merge feature vectors from both clusters
                    if len(merged_galleries[cluster_id1].shape) == 1:
                        # Handle case where we have single vectors
                        merged_features = np.vstack([
                            merged_galleries[cluster_id1].reshape(1, -1),
                            merged_galleries[cluster_id2].reshape(1, -1)
                        ])
                    else:
                        # Handle case where we have multiple vectors per track
                        merged_features = np.vstack([
                            merged_galleries[cluster_id1],
                            merged_galleries[cluster_id2]
                        ])
                    
                    # Merge track IDs from both clusters
                    clusters[cluster_id1].extend(clusters[cluster_id2])
                    
                    # Update the merged gallery
                    merged_galleries[cluster_id1] = merged_features
                    
                    # Update the track to cluster mapping
                    for track_id in clusters[cluster_id2]:
                        track_to_cluster[track_id] = cluster_id1
                    
                    # Remove the merged cluster
                    del clusters[cluster_id2]
                    del merged_galleries[cluster_id2]
                    
                    # Break out of the inner loop to restart with the new merged galleries
                    break
            
            # If we found a match and merged clusters, break out of the outer loop too
            if matches_found:
                break
    
    # Return the final list of clusters
    return list(clusters.values())

def save_clusters(clusters, output_path='clusters.json'):
    with open(output_path, 'w') as f:
        json.dump({'clusters': clusters}, f)
    print(f"Clusters saved to {output_path}")

def filter_gallary(gallary, min_size=3):
    # Filter out tracks with less than min_size features
    filtered_gallary = {track_id: features for track_id, features in gallary.items() if len(features) >= min_size}
    return filtered_gallary

def merge_and_save_gallery(gallery_path, clusters, output_path=None):
    """
    Merge gallery features based on identity clusters and save the merged gallery
    
    Args:
        gallery_path (str): Path to the original gallery directory
        clusters (list): List of identity clusters (each cluster is a list of track IDs)
        output_path (str, optional): Path to save the merged gallery. If None, overwrites the original gallery.
    
    Returns:
        dict: Dictionary of merged gallery features {cluster_id: features}
    """
    # If no output path is specified, use the original gallery path
    if output_path is None:
        output_path = gallery_path
    
    # Load the original gallery
    original_gallery = {}
    for filename in os.listdir(gallery_path):
        if filename.endswith('.npy'):
            track_id = int(filename.split('.')[0].split('/')[-1])
            features = np.load(os.path.join(gallery_path, filename))
            original_gallery[track_id] = features
    
    # Create merged gallery
    merged_gallery = {}
    
    # For each cluster, merge the features
    for i, cluster in enumerate(clusters):
        cluster_id = i + 1  # Start cluster IDs from 1
        merged_features = []
        
        for track_id in cluster:
            if track_id in original_gallery:
                # Add all features from this track to the merged features
                merged_features.append(original_gallery[track_id])
        
        if merged_features:
            # Stack all features for this cluster
            merged_gallery[cluster_id] = np.vstack(merged_features)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Remove all existing .npy files in the output directory
    for filename in os.listdir(output_path):
        if filename.endswith('.npy'):
            os.remove(os.path.join(output_path, filename))
    
    # Save the merged gallery
    for cluster_id, features in merged_gallery.items():
        np.save(os.path.join(output_path, f"{cluster_id}.npy"), features)
    
    print(f"Merged gallery saved to {output_path}")
    print(f"Original gallery had {len(original_gallery)} tracks")
    print(f"Merged gallery has {len(merged_gallery)} identity clusters")
    
    return merged_gallery

if __name__ == "__main__":
    # Load tracklets
    video_path = 'videos/simple_2.mov'
    tracklets_path = 'results/rtraks2.json'
    
    # Load the gallery we just created
    gallery_path = 'gallery/s2/'
    gallery = load_gallary(gallery_path)
    
    # Filter gallery if needed
    filtered_gallary = filter_gallary(gallery, min_size=3)
    
    # Cluster identities
    print("\nClustering identities...")
    clusters = cluster_identities(filtered_gallary, threshold=0.75)
    # Print clusters
    for i, cluster in enumerate(clusters):
        if len(cluster) > 1:  # Only show clusters with multiple tracks
            print(f"Identity {i+1}: Tracks {cluster}")
    
    # Merge and save the gallery based on clusters
    print("\nMerging gallery based on identity clusters...")
    merge_and_save_gallery(gallery_path, clusters)
    
    print("\nGallery processing complete. The original gallery has been replaced with the merged gallery.")