# RFDETR Object Detection

Real-time object detection using RFDETR (Roboflow Detection Transformer) model.

## Features

- Real-time object detection from webcam
- FPS monitoring
- Detection count display
- Bounding boxes and confidence scores
- Support for multiple RFDETR models (Base, Nano, Small, Medium, Large)

## Requirements

```bash
pip install rfdetr supervision opencv-python torch torchvision
```

## Usage

```bash
python3 denem_rfdetr.py
```

## Models Available

- `RFDETRNano()` - Fastest, smallest model
- `RFDETRBase()` - Balanced performance
- `RFDETRSmall()` - Small model
- `RFDETRMedium()` - Medium model  
- `RFDETRLarge()` - Largest, most accurate model

## Controls

- Press 'q' to quit the application

## Performance

The FPS (Frames Per Second) is displayed in real-time on the video feed. Performance varies based on:
- Selected model size
- Hardware capabilities
- Number of detected objects

## License

MIT License
