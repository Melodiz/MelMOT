import json
import os

def save_tracklets_to_json(tracklets, output_path):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save tracklets to JSON file
    with open(output_path, "w") as f:
        json.dump({"tracklets": tracklets}, f, indent=2)
    
    # print(f"Tracklets saved to {output_path}")