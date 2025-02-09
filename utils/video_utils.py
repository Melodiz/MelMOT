import cv2
import time

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