# RagbaarNet - Video Processing with Real-time Segmentation

A modular video processing system that performs real-time semantic segmentation on video frames with synchronized display.

## Master's Thesis Project

**Title**: Real-time Music Generation from Driver Perception and In-Vehicle Context in Smart Vehicles

## 🎯 Features

- **Real-time Video Processing**: Processes video frames from multiple sources (camera, file, stream)
- **Semantic Segmentation**: Uses YOLO and Segformer models for object segmentation
- **Optimized Performance**: Processes segmentation every 5 frames for faster computation
- **Synchronized Display**: Shows original video and segmentation overlay in real-time
- **Interactive ROI**: Drag and adjust Region of Interest with curved boundaries
- **Web-based Interface**: Modern HTML5 interface with mobile support

## 🏗️ System Architecture

```
UI.html (Frontend)
    ↓ (HTTP/WebSocket)
processor.py (Backend)
    ↓ (Frame Processing)
Segmentor.py (AI Models)
    ↓ (Inference)
YOLO/Segformer Models
```

```
UI.html (Frontend)
    ↓ (HTTP/WebSocket)
processor.py (Backend)
    ↓ (Frame Processing)
Segmentor.py (AI Models)
    ↓ (Inference)
YOLO/Segformer Models
```

## 📋 Prerequisites

- Python 3.8+
- Webcam or video files for input
- GPU recommended for faster segmentation (CUDA support)

## 🚀 Quick Start

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the System**

   ```bash
   python run_system.py
   ```

3. **Choose Option 3** to start both processor and UI

4. **Open UI** in your web browser (auto-opens or use the provided file path)

## 📦 Installation Details

### Dependencies

```bash
# Core dependencies
pip install flask flask-socketio opencv-python numpy

# AI/ML dependencies  
pip install torch torchvision ultralytics transformers

# Additional utilities
pip install python-socketio eventlet Pillow
```

### Pre-trained Models

The system automatically downloads required models:

- **YOLO**: `yolov8m-seg.pt` (Medium YOLO segmentation model)
- **Segformer**: `segformer-b0-finetuned-cityscapes-1024-1024` (from Hugging Face)

## 🎮 Usage

### 1. Start the Video Processor

```bash
cd modules/Platform
python processor.py --host 127.0.0.1 --port 5000 --interval 5
```

**Parameters:**

- `--host`: Server host (default: 127.0.0.1)
- `--port`: Server port (default: 5000)  
- `--interval`: Process every N frames (default: 5)

### 2. Open the UI

Open `modules/Platform/UI.html` in a web browser

### 3. Select Input Source

- 📱 **Phone Camera**: Use device camera
- 🎥 **Video File**: Upload video file
- 📺 **Screen Record**: Record screen (fallback to camera)
- 🌐 **Network Stream**: RTMP/HTTP video stream

### 4. Adjust ROI (Region of Interest)

- Drag **green points** to adjust corners
- Drag **cyan points** to control curve shapes
- Use **🎛️ Toggle Curves** to show/hide curve controls
- Use **🔄 Reset ROI** to restore default

### 5. Monitor Segmentation

- **Segmentation Preview**: Real-time overlay in top-right corner
- **Frame Counter**: Shows current processing frame
- **Status Indicator**: Connection status to processor
- **Performance Info**: Frames since last segmentation

## 🎛️ Controls

| Button | Function |
|--------|----------|
| 📂 Change Source | Switch input source |
| 🎛️ Toggle Curves | Show/hide curve controls |
| 🔄 Reset ROI | Reset region of interest |
| ⏸️ Pause/Resume | Pause video processing |
| 📸 Screenshot | Save current frame |
| 🔍 Segmentation | Toggle frame processing |

## 🔧 Configuration

### Processing Interval

Adjust segmentation frequency for performance:

```bash
# Process every 3 frames (faster segmentation, higher CPU usage)
python processor.py --interval 3

# Process every 10 frames (slower segmentation, lower CPU usage)  
python processor.py --interval 10
```

### Network Configuration

For remote access or multiple devices:

```bash
# Allow external connections
python processor.py --host 0.0.0.0 --port 5000
```

## 📊 API Endpoints

The processor exposes REST API endpoints:

- `POST /api/process_frame` - Send frame for processing
- `GET /api/get_display` - Get synchronized display data
- `GET /api/status` - Get processor status
- `WebSocket /` - Real-time updates

## 🔍 Segmentation Models

### YOLO Segmentation

- **Model**: YOLOv8 Medium Segmentation
- **Classes**: 80 COCO classes (person, car, bike, etc.)
- **Performance**: ~30-60 FPS on GPU
- **Use Case**: Real-time object detection and segmentation

### Segformer

- **Model**: Segformer-B0 fine-tuned on Cityscapes
- **Classes**: 19 urban scene classes (road, building, car, person, etc.)
- **Performance**: ~10-20 FPS on GPU
- **Use Case**: Detailed urban scene understanding

## 🎯 Performance Optimization

### For Real-time Processing

1. **Use GPU**: Install CUDA-compatible PyTorch
2. **Adjust Interval**: Increase frame interval (5-10 frames)
3. **Reduce Resolution**: Lower input video resolution
4. **Use YOLO**: Faster than Segformer for real-time

### For High Accuracy

1. **Use Segformer**: Better segmentation quality
2. **Process Every Frame**: Set interval to 1
3. **Higher Resolution**: Use full-resolution input
4. **Post-processing**: Enable confidence filtering

## 🐛 Troubleshooting

### Common Issues

#### 1. "Import flask could not be resolved"

```bash
pip install flask flask-socketio
```

#### 2. "Processor not available"

- Start processor first: `python modules/Platform/processor.py`
- Check port 5000 is not in use
- Verify firewall settings

#### 3. "No segmentation available"**

- Check processor logs for errors
- Ensure YOLO model downloads successfully
- Verify sufficient GPU/CPU resources

#### 4. "Camera not accessible"**

- Allow browser camera permissions
- Check if camera is used by other apps
- Try different browsers (Chrome recommended)

### Performance Issues

**Slow Processing:**

- Increase processing interval (`--interval 10`)
- Use smaller YOLO model (`yolov8n-seg.pt`)
- Reduce video resolution
- Close other GPU-intensive applications

**High CPU Usage:**

- Enable GPU acceleration
- Increase processing interval
- Use CPU-optimized models

## 📂 Project Structure

```
RagbaarNet/
├── modules/
│   ├── Platform/
│   │   ├── processor.py      # Main processing server
│   │   └── UI.html          # Web interface
│   └── Segmentation/
│       ├── Segmentor.py     # Segmentation framework
│       ├── example_usage.py # Example usage
│       └── Pre-trained Models/
├── requirements.txt         # Dependencies
├── run_system.py           # Startup script
└── README.md              # This file
```

## 🚀 Advanced Usage

### Custom Models

Add new segmentation models by extending `BaseSegmentor` in `Segmentor.py`:

```python
from modules.Segmentation.Segmentor import BaseSegmentor

class CustomSegmentor(BaseSegmentor):
    def load_model(self):
        # Load your custom model
        pass
    
    def predict(self, image):
        # Implement prediction logic
        pass
```

### Integration

Use the processor in your own applications:

```python
from modules.Platform.processor import VideoProcessor

processor = VideoProcessor()
result = processor.add_frame(frame)
display_data = processor.get_synchronized_display()
```

## 📄 License

This project is part of a Master's Thesis research. Please refer to the academic institution's guidelines for usage and distribution.

## 🤝 Contributing

This is a research project. For questions or collaborations, please contact the research team.

## 📚 References

- [YOLOv8 Segmentation](https://docs.ultralytics.com/)
- [Segformer](https://huggingface.co/docs/transformers/model_doc/segformer)
- [Flask-SocketIO](https://flask-socketio.readthedocs.io/)
- [OpenCV Python](https://docs.opencv.org/)
