# Vehicle Detection and Tracking using YOLOv8 and ByteTrack

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction
This project aims to develop a system for vehicle detection and tracking using the YOLOv8 object detection model and ByteTrack tracking algorithm. The system is designed to identify and track vehicles in real-time from video feeds, which can be utilized in various applications such as traffic monitoring, autonomous driving, and smart city initiatives.

## Features
- Real-time vehicle detection using YOLOv8
- Accurate and efficient vehicle tracking with ByteTrack
- Support for various video formats
- Visualization of detection and tracking results

## Requirements
- Python 3.7+
- CUDA-enabled GPU (for faster processing)
- The following Python libraries:
  - numpy
  - opencv-python
  - torch
  - torchvision
  - yolov8 (custom package or repository)
  - bytetrack (custom package or repository)

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/vehicle-detection-tracking.git
    cd vehicle-detection-tracking
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Install YOLOv8 and ByteTrack packages:
    ```bash
    pip install yolov8
    pip install bytetrack
    ```

## Usage
1. Prepare your video file or camera feed.
2. Run the detection and tracking script:
    ```bash
    python detect_and_track.py --input path_to_video.mp4 --output path_to_output.mp4
    ```

### Script Options
- `--input`: Path to the input video file or camera feed.
- `--output`: Path to save the output video with detection and tracking annotations.

## Results
The output will be a video file with detected and tracked vehicles highlighted. Each vehicle will have a unique ID for tracking across frames.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements
- [YOLOv8](https://github.com/ultralytics/yolov8)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
