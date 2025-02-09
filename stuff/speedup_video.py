import cv2
import os

def speed_up_video(input_path, output_path, speed_factor=3):
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps * speed_factor, (width, height))
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Write every 8th frame to output video
        if frame_count % speed_factor == 0:
            out.write(frame)
        
        frame_count += 1
        
        # Print progress
        if frame_count % 100 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")
    
    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print("Video processing complete!")

# Set up input and output paths
input_video = 'results/tracklet_visualization.mp4'
output_video = 'results/visual_tracker_3x.mp4'

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_video), exist_ok=True)

# Process the video
speed_up_video(input_video, output_video)