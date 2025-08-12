# RagbaarNet - AI Music Generation Platform

**Master's Thesis Project**: Real-time Music Generation from Driver Perception and In-Vehicle Context in Smart Vehicles

> ⚠️ **Project Status**: Work in Progress - Core components under development

## 🎯 Overview

RagbaarNet is a research project developing an AI system that generates music based on driver perception and in-vehicle context. The platform uses real-time video processing and semantic segmentation to understand driving scenarios.

## 🏗️ Current Architecture

```text
UI.html (Frontend)
    ↓ (WebSocket)
processor.py (Backend)
    ↓ (Frame Processing)
Segmentor.py (AI Models)
    ↓ (YOLO/Segformer)
Pre-trained Models
```

## 📋 Prerequisites

- Python 3.8+
- Webcam or video input
- GPU recommended (CUDA support)

## 🚀 Quick Start

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Processor**

   ```bash
   cd modules/Platform
   python processor.py
   ```

3. **Open Web Interface**
   
   Open `modules/Platform/UI.html` in your browser

4. **Select Input Source**
   - � Camera
   - 🎥 Video File  
   - � Screen Record
   - 🌐 Network Stream

## 🔧 Components

### Video Processing (`processor.py`)

- Flask-SocketIO backend server
- Real-time frame processing
- WebSocket communication with frontend

### Segmentation Framework (`Segmentor.py`)

- Modular segmentation architecture
- YOLO and Segformer model support
- Extensible base classes for custom models

### Web Interface (`UI.html`)

- Input source selection
- Real-time video display
- ROI (Region of Interest) controls

## 📊 Supported Models

- **YOLO**: YOLOv8 segmentation models
- **Segformer**: Hugging Face transformers

<!-- ## 📂 Project Structure

```bash
RagbaarNet/
├── modules/
│   ├── Platform/
│   │   ├── processor.py      # Backend server
│   │   ├── UI.html          # Web interface
│   │   ├── styles.css       # UI styling
│   │   └── script.js        # Frontend logic
│   ├── Segmentation/
│   │   ├── Segmentor.py     # Segmentation framework
│   │   └── Pre-trained Models/
│   └── utils/
│       └── logging_setup.py # Logging utilities
├── Dataset/                  # Training datasets
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## � Development Status

- ✅ Core segmentation framework
- ✅ Web-based video processing interface  
- ✅ Real-time frame processing pipeline
- 🚧 Music generation algorithms
- 🚧 Driver perception analysis
- ⏳ In-vehicle context integration

## 📄 License

This project is part of Master's Thesis research. Academic use only. -->

## 📚 Key Dependencies

- Flask & Flask-SocketIO
- OpenCV
- PyTorch & Ultralytics
- Transformers (Hugging Face)
