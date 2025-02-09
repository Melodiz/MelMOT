import pickle
import os
import numpy as np

def load_preprocessed_data(checkpoint_dir='checkpoints', num_frames=300):
    all_frames = []
    all_detections = []
    frames_loaded = 0
    checkpoint_size = 100  # Size of each checkpoint

    while frames_loaded < num_frames:
        checkpoint_number = (frames_loaded // checkpoint_size) + 1
        frames_path = os.path.join(checkpoint_dir, f'frames_checkpoint_{checkpoint_number * checkpoint_size}.npy')
        detections_path = os.path.join(checkpoint_dir, f'detections_checkpoint_{checkpoint_number * checkpoint_size}.pkl')

        if not os.path.exists(frames_path) or not os.path.exists(detections_path):
            break

        frames = np.load(frames_path)
        with open(detections_path, 'rb') as f:
            detections = pickle.load(f)

        frames_to_add = min(len(frames), num_frames - frames_loaded)
        all_frames.extend(frames[:frames_to_add])
        all_detections.extend(detections[:frames_to_add])

        frames_loaded += frames_to_add

        if len(frames) < checkpoint_size:
            break  # This was the last (possibly partial) checkpoint

    return all_frames[:num_frames], all_detections[:num_frames]