# Players Tracking System

A **real-time** football player detection and tracking system using **YOLOv11** and **DeepSORT** for identifying and maintaining consistent player IDs throughout video footage.

## Overview

This system combines YOLOv11 object detection with DeepSORT tracking to provide robust player identification and re-identification in football video. The implementation addresses challenges like identical jersey colors, player occlusions, and temporary disappearances from the frame.

## Features

- **Real-time player detection** using a fine-tuned YOLOv11 model  
- **Multi-object tracking** with DeepSORT for consistent player IDs  
- **Re-identification** capabilities for players re-entering the frame  
- **False-positive filtering** to reduce grass and non-player detections  
- Configurable tracking parameters optimized for football scenarios

## Software Requirements

- **Python** 3.8 or higher  
- **CUDA** 11.8+ (CPU-only mode is supported but much slower)  
- **Operating System:** Windows 10+

## Installation

1. **Clone the project**  
   ```bash
   git clone https://github.com/your-username/football-tracking.git
   cd football-tracking

  Create a new directory and save the following files:
  - YOLOv11_Detector.py
  - tracker.py
  - football_player_tracking.py
  - Fine tunned model (in .pt format)

 * Set up Python Environment
   - Create virtual environment (recommended)
    python -m venv football_tracking_env

  - Activate environment
  - Windows:
      football_tracking_env\Scripts\activate
  - macOS/Linux:
      source football_tracking_env/bin/activate

## Install Dependencies
pip install \
  ultralytics>=8.0.0 \
  opencv-python>=4.5.0 \
  deep-sort-realtime>=1.2.1 \
  numpy>=1.21.0 \
  torch>=1.9.0 \
  torchvision>=0.10.0

## Usage
* Place your fine-tuned YOLOv11 model and input video in the project directory.
* Update the paths in football_player_tracking.py as needed.
* Run the main script

## 📚 References

This work is inspired by several research and open-source projects:

1. **Identification and Tracking of Players in Sport Videos**  
   - ResearchGate: [Identification and tracking of players in sport videos](https://www.researchgate.net/publication/262257764_Identification_and_tracking_of_players_in_sport_videos)

2. **YOLOv10 Detection & Tracking**  
   - GitHub: [thehummingbird/computer_vision – yolov10_detection_and_tracking](https://github.com/thehummingbird/computer_vision/blob/main/yolov10_detection_and_tracking)

3. **Recent Tracking Advances**  
   - ArXiv preprint: [2307.14591](https://arxiv.org/pdf/2307.14591)

4. **IEEE Conference Paper: Player Tracking**  
   - IEEE Xplore: [arnumber=9739737](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9739737)

These sources provided foundational approaches in detection, appearance modeling, and multi-object tracking mechanisms.



