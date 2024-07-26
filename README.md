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

![COCO_Result](https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/yolo-comparison-plots.png)

## Features
- Real-time vehicle detection using YOLOv8
- Accurate and efficient vehicle tracking with ByteTrack
- Support for various video formats
- Visualization of detection and tracking results

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/KevinElevenVN/Vehicle-Detection-and-Tracking-using-YOLOv8.git
    cd Vehicle-Detection-and-Tracking-using-YOLOv8
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Prepare your video file or camera feed.
2. Run the detection and tracking script:
    ```bash
    python detect_and_track.py --input path_to_video.mp4 --output path_to_output.mp4
    ```

## Results
The output will be a video file with detected and tracked vehicles highlighted. Each vehicle will have a unique ID for tracking across frames.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements
- [YOLOv8](https://github.com/ultralytics/yolov8)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
