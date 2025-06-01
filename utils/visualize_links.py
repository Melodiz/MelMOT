import os

def visualize_gallery_links(links, base_gallery_path, query_gallery_path, output_path="gallery_links_visualization.jpg", 
                           query_samples=5, base_samples=3):
    """
    Visualize gallery links by showing random samples from each matched pair.
    
    Args:
        links (dict): Dictionary mapping query IDs to base IDs
        base_gallery_path (str): Path to the base gallery directory
        query_gallery_path (str): Path to the query gallery directory
        output_path (str): Path to save the visualization image
        query_samples (int): Number of random samples to show from query gallery
        base_samples (int): Number of random samples to show from base gallery
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import random
    
    # Count number of links to determine figure size
    num_links = len(links)
    if num_links == 0:
        print("No links to visualize")
        return
    
    # Create figure
    fig = plt.figure(figsize=(15, 5 * num_links))
    gs = GridSpec(num_links, query_samples + base_samples + 1, figure=fig)
    
    # Process each link
    for i, (query_id, link_info) in enumerate(links.items()):
        base_id = link_info["match_id"]
        similarity = link_info["similarity"]
        
        # Get query person directory
        query_person_dir = os.path.join(query_gallery_path, f"reid_{query_id}")
        if not os.path.exists(query_person_dir):
            query_person_dir = os.path.join(query_gallery_path, query_id)
        
        # Get base person directory
        base_person_dir = os.path.join(base_gallery_path, f"reid_{base_id}")
        if not os.path.exists(base_person_dir):
            base_person_dir = os.path.join(base_gallery_path, base_id)
        
        # Get image files
        query_images = [f for f in os.listdir(query_person_dir) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        base_images = [f for f in os.listdir(base_person_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Sample random images
        query_samples_actual = min(query_samples, len(query_images))
        base_samples_actual = min(base_samples, len(base_images))
        
        sampled_query_images = random.sample(query_images, query_samples_actual)
        sampled_base_images = random.sample(base_images, base_samples_actual)
        
        # Add label in the middle
        ax_label = fig.add_subplot(gs[i, query_samples])
        ax_label.text(0.5, 0.5, f"→\nSim: {similarity:.2f}", 
                     ha='center', va='center', fontsize=14)
        ax_label.axis('off')
        
        # Plot query images
        for j, img_file in enumerate(sampled_query_images):
            img_path = os.path.join(query_person_dir, img_file)
            img = plt.imread(img_path)
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(img)
            ax.set_title(f"Query ID: {query_id}")
            ax.axis('off')
        
        # Plot base images
        for j, img_file in enumerate(sampled_base_images):
            img_path = os.path.join(base_person_dir, img_file)
            img = plt.imread(img_path)
            ax = fig.add_subplot(gs[i, query_samples + 1 + j])
            ax.imshow(img)
            ax.set_title(f"Base ID: {base_id}")
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Gallery links visualization saved to {output_path}")
    plt.close(fig)