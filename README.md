# 🛣️ Advanced Curved Lane Detection System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research Grade](https://img.shields.io/badge/Research-Grade-orange.svg)](#)

> **A Research-Grade Computer Vision System for Robust Lane Detection in Autonomous Driving Applications**

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/zahramh99/advanced-lane-detection.git
cd advanced-lane-detection

# Create virtual environment
python -m venv lane_env
source lane_env/bin/activate  # On Windows: lane_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Camera Calibration (First-time setup)
python calibrate_camera.py

from advanced_lane_detection import AdvancedLaneDetector, LaneDetectionConfig

# Initialize detector with custom configuration
config = LaneDetectionConfig(
    s_thresh=(120, 255),
    sx_thresh=(20, 255),
    nwindows=12
)

detector = AdvancedLaneDetector(config)

# Process single image
result = detector.process_image(your_image)

# Process video stream
detector.process_video('input_video.mp4', 'output_video.mp4')
# 1. Standard pipeline
result = detector.process_image(image)

# 2. With debug visualization
result, debug_info = detector.process_image_debug(image)

# 3. Batch processing for datasets
results = detector.process_batch(image_list)
config = LaneDetectionConfig(
    # Threshold parameters
    s_thresh=(100, 255),
    sx_thresh=(15, 255),
    l_thresh=(120, 255),
    
    # Perspective transform
    src_points=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)]),
    
    # Detection parameters
    nwindows=9,
    margin=100,
    minpix=50
)

## Citation
@software{gharehmahmoodlee2024lanedection,
  title = {Advanced Curved Lane Detection System},
  author = {Gharehmahmoodlee, Zahra},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/zahramh99/advanced-lane-detection}}
}