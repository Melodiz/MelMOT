# MelMOT: Multi-Object Tracking for Retail Spaces

A research project implementing innovative approaches to multi-user tracking in retail environments, focusing on robust single-camera tracking and effective cross-camera re-identification (Re-ID).

**Repository**: [https://github.com/Melodiz/MelMOT](https://github.com/Melodiz/MelMOT)

## Abstract

This research details the development of an innovative system for continuous multi-user tracking within retail environments, focusing on robust single-camera tracking and effective cross-camera re-identification (Re-ID). A key contribution for single-camera tracking is the application of tailored post-processing heuristics addressing phantom tracks and manikin misclassifications to a state-of-the-art pipeline (YOLOv12, BoT-SORT, OSNet). The primary innovation lies in the cross-camera Re-ID methodology, which overcomes the limitations of appearance-only approaches in challenging retail settings. This is achieved through a robust spatiotemporal matching strategy utilizing homography to establish a common ground plane for coordinate-based comparisons. A novel exponential loss function is introduced for similarity scoring, significantly improving matching accuracy, particularly in scenarios with partial occlusions or non-ideal detections. The system architecture proposes a multi-stage matching process, where spatiotemporal constraints primarily guide Re-ID, with aggregated appearance features from multiple confirmed views providing a fallback mechanism, ensuring high accuracy and stability in generating comprehensive visitor trajectories.

## Project Overview

This system constructs continuous trajectories for every visitor using a network of static, ceiling-mounted surveillance cameras. The data can be utilized for analyzing customer movement within shopping malls, including:

- **Conversion tracking**
- **Behavioral pattern analysis**
- **High-interest area identification**
- **Business intelligence statistics**

## Architecture

The system consists of two main components:

### 1. Single-Camera Multiple Object Tracking (MOT)
- **Detection**: YOLOv12 (state-of-the-art object detection)
- **Tracking**: BoT-SORT (robust online tracking)
- **Re-ID**: OSNet (appearance feature extraction)
- **Post-processing**: Heuristics for phantom track removal and manikin filtering

### 2. Cross-Camera Person Re-identification
- **Spatiotemporal matching**: Homography-based coordinate alignment
- **Appearance matching**: Aggregated feature comparison
- **Multi-stage process**: Spatiotemporal constraints with appearance fallback

## Features

- **Real-time processing** at 30 FPS
- **Robust tracking** in crowded retail environments
- **Cross-camera linking** using geometric and appearance cues
- **Post-processing heuristics** for improved accuracy
- **Comprehensive trajectory generation** across entire mall visits

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- Time-synchronized surveillance cameras

## Installation

1. **Clone the repository**
    ```bash
    git clone https://github.com/Melodiz/MelMOT.git
    cd MelMOT
    ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download models** (optional - will download automatically on first run)
   ```bash
   python -c "from ultralytics import YOLO; YOLO('yolov12n.pt')"
   ```

## Usage

### Basic Single-Camera Tracking

```python
from melmot.tracking import SingleCameraTracker

# Initialize tracker
tracker = SingleCameraTracker(
    model_path="yolov12n.pt",
    max_age=100,
    min_hits=3
)

# Process video
results = tracker.track_video("path/to/video.mp4")
```

### Cross-Camera Re-identification

```python
from melmot.reid import CrossCameraReID

# Initialize Re-ID system
reid_system = CrossCameraReID(
    homography_matrices=homography_data,
    appearance_threshold=0.8
)

# Link tracklets across cameras
global_trajectories = reid_system.link_tracklets(camera_tracklets)
```

### Command Line Interface

```bash
# Single camera tracking
python -m melmot.cli track --video videos/input.mp4 --output results/output.mp4

# Cross-camera Re-ID
python -m melmot.cli reid --tracklets results/tracklets.json --output results/global_trajectories.json

# Full pipeline
python -m melmot.cli pipeline --config config/retail_mall.yaml
```

## Configuration

Create a configuration file `config/retail_mall.yaml`:

```yaml
# Camera configuration
cameras:
  - id: "cam_001"
    position: [0, 0, 3.5]  # x, y, height in meters
    homography: "homography/cam_001.json"
  
  - id: "cam_002"
    position: [10, 0, 3.5]
    homography: "homography/cam_002.json"

# Tracking parameters
tracking:
  max_age: 100
  min_hits: 3
  iou_threshold: 0.3
  static_threshold: 2.0

# Re-ID parameters
reid:
  appearance_threshold: 0.8
  spatial_tolerance: 0.5  # meters
  temporal_window: 5.0    # seconds
```

## Output Format

### Tracklets
```json
{
  "camera_001": {
    "track_001": [
      {
        "frame": 100,
        "bbox": [x, y, w, h],
        "confidence": 0.95,
        "features": [0.1, 0.2, ...]
      }
    ]
  }
}
```

### Global Trajectories
```json
{
  "person_001": {
    "global_id": "person_001",
    "trajectories": {
      "camera_001": {"start_frame": 100, "end_frame": 200},
      "camera_002": {"start_frame": 250, "end_frame": 350}
    },
    "total_duration": 15.0
  }
}
```

## Research Contributions

This project introduces several novel approaches:

1. **Tailored post-processing heuristics** for phantom track removal and manikin misclassification
2. **Robust spatiotemporal matching** using homography for coordinate-based comparisons
3. **Novel exponential loss function** for similarity scoring in challenging scenarios
4. **Multi-stage matching process** with spatiotemporal constraints and appearance fallback

## Performance

- **MOTA**: Competitive with state-of-the-art methods
- **ID Switches**: Significantly reduced through post-processing
- **Processing Speed**: Real-time capable (30 FPS)
- **Accuracy**: Robust in crowded retail environments

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_tracking.py
pytest tests/test_reid.py
pytest tests/test_utils.py

# Run with coverage
pytest --cov=melmot tests/
```

## Development

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Write comprehensive docstrings
- Maintain 90%+ test coverage

### Project Structure
```
melmot/
├── core/           # Core tracking algorithms
├── detection/      # Object detection models
├── reid/          # Re-identification modules
├── utils/         # Utility functions
├── config/        # Configuration files
├── tests/         # Test suite
└── examples/      # Usage examples
```

## References

- **YOLOv12**: State-of-the-art object detection
- **BoT-SORT**: Robust online tracking
- **OSNet**: Omni-scale network for Re-ID
- **Homography-based matching**: Geometric constraint utilization

## Research Paper

The complete research paper is available in this repository:
- **Source**: [main.tex](latex/main.tex) (LaTeX source)
- **PDF**: [Download Research Paper](https://drive.google.com/file/d/1r1hOHQpZdUl5fumM93CqdrHqX0_uswO0/view)
- **Topic**: Innovative Approaches to Multi-User Tracking in Retail Spaces
- **Focus**: Single-camera MOT and cross-camera Re-ID methodologies

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This is a research project for coursework. The implementation focuses on demonstrating novel approaches rather than production deployment.