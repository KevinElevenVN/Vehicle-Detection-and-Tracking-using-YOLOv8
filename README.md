# Vehicle Detection and Tracking using YOLOv8 and ByteTrack

## <div align='center'>Introduction</div>
<div align='center'>
    This project aims to develop a system for vehicle detection and tracking using the YOLOv8 object detection model and ByteTrack tracking algorithm. The system is designed to identify and track vehicles in real-time from video feeds, which can be utilized in various applications such as traffic monitoring, autonomous driving, and smart city initiatives.
</div>

![COCO_Result](https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/yolo-comparison-plots.png)

## <div align='center'>Features</div>
- Real-time vehicle detection using YOLOv8
- Accurate and efficient vehicle tracking with ByteTrack
- Support for various video formats
- Visualization of detection and tracking results

<details>
<summary>Installation</summary>

1. Clone the repository:
    ```bash
    git clone https://github.com/KevinElevenVN/Vehicle-Detection-and-Tracking-using-YOLOv8.git
    cd Vehicle-Detection-and-Tracking-using-YOLOv8
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
</details>
    
<details open>
<summary>Usage</summary>

### CLI
YOLOv8 may be used directly in the Command Line Interface (CLI) with a `yolo` command:

```bash
yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'
```

`yolo` can be used for a variety of tasks and modes and accepts additional arguments, i.e. `imgsz=640`. See the YOLOv8 [CLI Docs](https://docs.ultralytics.com/usage/cli) for examples.

### Python
YOLOv8 may also be used directly in a Python environment, and accepts the same [arguments](https://docs.ultralytics.com/usage/cfg/) as in the CLI example above:

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="coco8.yaml", epochs=3)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format
```

See YOLOv8 [Python Docs](https://docs.ultralytics.com/usage/python) for more examples.

</details>

## Results
The output will be a video file with detected and tracked vehicles highlighted. Each vehicle will have a unique ID for tracking across frames.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements
- [YOLOv8](https://github.com/ultralytics/yolov8)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
