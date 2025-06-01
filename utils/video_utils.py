import cv2
import time
import matplotlib.pyplot as plt

def get_video_properties(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, width, height

def save_processed_video(output_path, processed_frames, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in processed_frames:
        out.write(frame)
    
    out.release()

def play_video(video_path, speed=7):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    delay = int(1000 / (fps * speed))  # Delay between frames in milliseconds

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Tracked Video (3x speed)', frame)

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_and_play_video(frames, detections, video_path, output_path, track_objects_func):
    # Get video properties
    fps, width, height = get_video_properties(video_path)

    # Perform tracking
    all_tracks, processed_frames = track_objects_func(frames, detections)

    # Save processed video
    save_processed_video(output_path, processed_frames, fps, width, height)

    print(f"Video processing complete. Output saved as {output_path}")

    # Play the resulting video
    print("Playing the resulting video at 7x speed. Press 'q' to quit.")
    play_video(output_path, speed=3)

def get_frame(video_path, frame_number):
    """
    Get a full frame from a video
    
    Args:
        video_path (str): Path to the video file
        frame_number (int): Frame number to extract
        
    Returns:
        numpy.ndarray: Frame image or None if failed
    """
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
        
    video.set(1, frame_number)
    ret, frame = video.read()
    video.release()
    
    if not ret:
        print(f"Error: Could not read frame {frame_number} from {video_path}")
        return None

    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def plot_two_crops(track1, track2, video_path1, video_path2, query_id=None, base_id=None, save_silent=False):
    """
    Plot two full frames side by side with bounding boxes and annotations
    
    Args:
        track1: Track data for the first image
        track2: Track data for the second image
        video_path1: Path to the first video
        video_path2: Path to the second video
        query_id: Optional ID for the query track (first image)
        base_id: Optional ID for the base track (second image)
    """
    # Get full frames instead of crops
    frame1 = get_frame(video_path1, track1['frame'])
    frame2 = get_frame(video_path2, track2['frame'])
    
    if frame1 is None or frame2 is None:
        print("Error: Could not read one or both frames")
        return
    
    # Extract video names from paths (just the filename)
    video1_name = video_path1.split('/')[-1]
    video2_name = video_path2.split('/')[-1]
    
    # Format coordinates and timestamps nicely
    coords1 = f"({track1['real_coordinates'][0]:.2f}, {track1['real_coordinates'][1]:.2f})"
    coords2 = f"({track2['real_coordinates'][0]:.2f}, {track2['real_coordinates'][1]:.2f})"
    
    # Create figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    
    # Annotate and plot first frame
    frame1_rgb = frame1.copy()  # Make a copy to avoid modifying the original
    frame1_bgr = cv2.cvtColor(frame1_rgb, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV functions
    
    # Draw bounding box on first frame
    bbox1 = track1['bbox']
    x1, y1, x2, y2 = map(int, bbox1)
    
    # Generate a unique color for this track ID
    track_id1 = query_id if query_id is not None else 1
    color_hash1 = hash(str(track_id1)) % 0xFFFFFF
    color1 = ((color_hash1 >> 16) & 0xFF, (color_hash1 >> 8) & 0xFF, color_hash1 & 0xFF)  # RGB format
    color1_bgr = (color1[2], color1[1], color1[0])  # BGR format for OpenCV
    
    # Draw rectangle
    cv2.rectangle(frame1_bgr, (x1, y1), (x2, y2), color=color1_bgr, thickness=2)
    
    # Create label with ID
    label1 = f"ID: {track_id1}"
    
    # Calculate text size for better positioning
    text_size1 = cv2.getTextSize(label1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    
    # Draw background rectangle for text
    cv2.rectangle(
        frame1_bgr,
        (x1, y1 - text_size1[1] - 10),
        (x1 + text_size1[0] + 10, y1),
        color1_bgr,
        -1  # Fill the rectangle
    )
    
    # Draw ID text with white color for better visibility
    cv2.putText(
        frame1_bgr,
        label1,
        (x1 + 5, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),  # White text
        2,
        lineType=cv2.LINE_AA
    )
    
    # Convert back to RGB for matplotlib
    frame1_annotated = cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2RGB)
    
    # Plot the annotated frame
    axs[0].imshow(frame1_annotated)
    title1 = f"{video1_name}"
    if query_id is not None:
        title1 += f" - ID: {query_id}"
    axs[0].set_title(title1, fontsize=12)
    
    # Add annotations as text below the image
    axs[0].text(0.5, -0.05, f"Frame: {track1['frame']}", 
                transform=axs[0].transAxes, ha='center', fontsize=10)
    axs[0].text(0.5, -0.10, f"Coordinates: {coords1}", 
                transform=axs[0].transAxes, ha='center', fontsize=10)
    axs[0].text(0.5, -0.15, f"Time: {track1['timestamp']:.2f}s", 
                transform=axs[0].transAxes, ha='center', fontsize=10)
    
    # Annotate and plot second frame
    frame2_rgb = frame2.copy()  # Make a copy to avoid modifying the original
    frame2_bgr = cv2.cvtColor(frame2_rgb, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV functions
    
    # Draw bounding box on second frame
    bbox2 = track2['bbox']
    x1, y1, x2, y2 = map(int, bbox2)
    
    # Generate a unique color for this track ID
    track_id2 = base_id if base_id is not None else 2
    color_hash2 = hash(str(track_id2)) % 0xFFFFFF
    color2 = ((color_hash2 >> 16) & 0xFF, (color_hash2 >> 8) & 0xFF, color_hash2 & 0xFF)  # RGB format
    color2_bgr = (color2[2], color2[1], color2[0])  # BGR format for OpenCV
    
    # Draw rectangle
    cv2.rectangle(frame2_bgr, (x1, y1), (x2, y2), color=color2_bgr, thickness=2)
    
    # Create label with ID
    label2 = f"ID: {track_id2}"
    
    # Calculate text size for better positioning
    text_size2 = cv2.getTextSize(label2, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    
    # Draw background rectangle for text
    cv2.rectangle(
        frame2_bgr,
        (x1, y1 - text_size2[1] - 10),
        (x1 + text_size2[0] + 10, y1),
        color2_bgr,
        -1  # Fill the rectangle
    )
    
    # Draw ID text with white color for better visibility
    cv2.putText(
        frame2_bgr,
        label2,
        (x1 + 5, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),  # White text
        2,
        lineType=cv2.LINE_AA
    )
    
    # Convert back to RGB for matplotlib
    frame2_annotated = cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2RGB)
    
    # Plot the annotated frame
    axs[1].imshow(frame2_annotated)
    title2 = f"{video2_name}"
    if base_id is not None:
        title2 += f" - ID: {base_id}"
    axs[1].set_title(title2, fontsize=12)
    
    # Add annotations as text below the image
    axs[1].text(0.5, -0.05, f"Frame: {track2['frame']}", 
                transform=axs[1].transAxes, ha='center', fontsize=10)
    axs[1].text(0.5, -0.10, f"Coordinates: {coords2}", 
                transform=axs[1].transAxes, ha='center', fontsize=10)
    axs[1].text(0.5, -0.15, f"Time: {track2['timestamp']:.2f}s", 
                transform=axs[1].transAxes, ha='center', fontsize=10)
    
    # Remove axis ticks for cleaner visualization
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Adjust layout to make room for the text
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Check if we should save silently or show the plot
    if save_silent:
        # Create the reid directory if it doesn't exist
        import os
        os.makedirs("reid", exist_ok=True)
        
        # Save the figure to a file
        save_path = f"reid/{query_id}-{base_id}.jpg"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free memory
        print(f"Saved match visualization to {save_path}")
    else:
        plt.show()
    
# Keep the get_crop function for backward compatibility
def get_crop(video_path, bbox, frame_number):
    # Read the frame
    video = cv2.VideoCapture(video_path)
    video.set(1, frame_number)
    ret, frame = video.read()
    if not ret:
        return None

    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get the bounding box coordinates
    left, top, right, bottom = map(int, bbox)
    
    # Ensure coordinates are within frame boundaries
    height, width = frame.shape[:2]
    left = max(0, min(left, width-1))
    right = max(left+1, min(right, width))
    top = max(0, min(top, height-1))
    bottom = max(top+1, min(bottom, height))
    
    try:
        crop = frame[top:bottom, left:right]
        return crop
    except Exception as e:
        print(f"Error cropping image: {e}")
        return None