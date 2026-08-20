"""
RagbaarNet Central Configuration
================================

This module centralizes all hard-coded variables for the RagbaarNet platform.
"""

import os
from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# --- Server Settings ---
SERVER_HOST = os.environ.get('RAGBAARNET_HOST', '0.0.0.0')
PROCESSOR_PORT = int(os.environ.get('RAGBAARNET_PORT', 5000))
TELEMETRY_PORT = int(os.environ.get('RAGBAARNET_TELEMETRY_PORT', 5500))
CORS_ALLOWED_ORIGINS = "*"
SECRET_KEY = 'video_processing_secret'

# --- Pipeline Settings ---
INPUT_QUEUE_MAXSIZE = 16
MUSIC_QUEUE_MAXSIZE = 16
DEBUG_INTERVAL = 10.0       # Seconds between debug logs
JPEG_QUALITY = 75
PROCESSING_MAX_SIDE = int(os.environ.get('RAGBAARNET_PROCESSING_MAX_SIDE', 0)) or None

# --- Detection Parameters ---
MAX_MISSING_FRAMES = 8
MAX_OBJECT_DISTANCE = 200   # Pixels
ROI_THICKNESS = 3           # Pixels
ROI_SAMPLES_PER_EDGE = 20

# --- Music Generation Defaults ---
DEFAULT_TEMPO = 120
DEFAULT_KEY_SIGNATURE = "C_major"
DEFAULT_TIME_SIGNATURE = (4, 4)
DEFAULT_MUSICIAN_TYPE = 'lstm-onessen-orchestral'
DEFAULT_AUDIO_BACKEND = os.environ.get('RAGBAARNET_AUDIO_BACKEND', 'tone').strip().lower()

# --- MIDI Settings ---
DEFAULT_MIDI_PORT = os.environ.get('RAGBAARNET_MIDI_PORT', 'RagbaarNetMIDI Port 1')
INSTRUMENT_MIDI_CHANNELS = {
    'piano': 0,
    'electric_piano': 1,
    'acoustic_guitar': 2,
    'electric_guitar': 3,
    'strings': 4,
    'pad': 5,
    'bass': 6,
    'synth': 7,
    'drums': 9,
}

# --- Model Paths ---
MODELS_DIR = PROJECT_ROOT / 'modules' / 'Models'

# Segmentation
YOLO_MODEL_PATH = os.environ.get(
    'RAGBAARNET_SEGMENTATION_MODEL_PATH',
    str(MODELS_DIR / 'Segmentation' / 'Pre-trained Models' / 'yolo26' / 'yolo26n-seg.pt')
)
SEGFORMER_MODEL_PATH = os.environ.get(
    'RAGBAARNET_SEGFORMER_PATH',
    str(MODELS_DIR / 'Segmentation' / 'Pre-trained Models' / 'segformer-b2-finetuned-cityscapes-1024-1024')
)

# Music
LSTM_MODEL_DIR = MODELS_DIR / 'Music' / 'LSTM_OnEssen'
LSTM_MODEL_PATH = LSTM_MODEL_DIR / 'LSTM_OnEssen.pt'
LSTM_MAPPING_PATH = LSTM_MODEL_DIR / 'mapping.json'
LSTM_SINGLE_FILE_DATASET = LSTM_MODEL_DIR / 'single_file_dataset'

# Datasets
KERN_DATASET_PATH = PROJECT_ROOT / "Dataset/KernScores/essen/europa/deutschl"
LMD_DATASET_PATH = PROJECT_ROOT / "Dataset/Lakh MIDI"  # Placeholder

# --- Mappings ---
RULE_BASED_CLASS_MAPPING = {
    #                   MIDI, velocity, instrument
    "car":              (60, 100, 'piano'),
    "truck":            (48, 120, 'piano'),
    "bus":              (48, 90, 'piano'),
    "train":            (55, 110, 'electric_piano'),
    "plane":            (72, 100, 'electric_piano'),
    "bicycle":          (64, 90, 'acoustic_guitar'),
    "person":           (72, 110, 'acoustic_guitar'),
    "motorcycle":       (70, 100, 'electric_guitar'),
    "traffic light":    (67, 70, 'strings'),
    "traffic sign":     (67, 70, 'strings'),
    "stop sign":        (69, 80, 'strings'),
}

TYPING_KEY_CLASS_MAP = {chr(c): "typing_letter" for c in range(ord('a'), ord('z') + 1)}
TYPING_KEY_CLASS_MAP.update({str(d): "typing_digit" for d in range(10)})
TYPING_KEY_CLASS_MAP.update({
    "backspace": "typing_delete", "enter": "typing_newline",
    "tab": "typing_indent", "space": "typing_space",
    "scroll": "scroll",
    "mousemove": "mousemove",
})

SEGMENTATION_PALETTE = {
    # Cityscapes Semantic Classes
    "road":            [128,  64, 128],   # Viola Purple
    "sidewalk":        [244,  35, 232],   # Bright Magenta
    "building":        [ 70,  70,  70],   # Dark Gray
    "wall":            [102, 102, 156],   # Slate Blue
    "fence":           [190, 153, 153],   # Dusty Pink
    "pole":            [153, 153, 153],   # Light Gray
    "traffic light":   [250, 170,  30],   # Amber
    "traffic sign":    [220, 220,   0],   # Lemon Yellow
    "vegetation":      [107, 142,  35],   # Olive Green
    "terrain":         [152, 251, 152],   # Pale Green
    "sky":             [ 70, 130, 180],   # Steel Blue
    "person":          [220,  20,  60],   # Crimson
    "rider":           [255,   0,   0],   # Pure Red
    "car":             [  0,   0, 142],   # Navy Blue
    "truck":           [  0,   0,  70],   # Midnight Blue
    "bus":             [  0,  60, 100],   # Deep Teal Blue
    "train":           [  0,  80, 100],   # Dark Cyan
    "motorcycle":      [  0,   0, 230],   # Royal Blue
    "bicycle":         [119,  11,  32],   # Burgundy
    # Extended Cityscapes Labels
    "parking":         [160, 160, 160],   # Cool Gray
    "rail track":      [230, 150, 140],   # Salmon Pink
    "guard rail":      [180, 165, 180],   # Silver Lilac
    "bridge":          [150, 100, 100],   # Warm Brown
    "tunnel":          [150, 120,  90],   # Earth Brown
    "caravan":         [  0,   0,  90],   # Dark Navy
    "trailer":         [  0,   0, 110],   # Indigo Blue
    # COCO Road Objects
    "stop sign":       [255,   0,   0],   # Stop Sign Red
    "fire hydrant":    [178,  34,  34],   # Firebrick
    "bench":           [160,  82,  45],   # Saddle Brown
    "parking meter":   [112, 128, 144],   # Slate Gray
    # Animals (Road Relevant)
    "bird":            [135, 206, 235],   # Sky Blue
    "dog":             [139,  69,  19],   # Saddle Brown
    "cat":             [205, 133,  63],   # Peru
    "horse":           [160,  82,  45],   # Sienna
    "sheep":           [245, 245, 220],   # Beige
    "cow":             [110,  70,  30],   # Dark Brown
    "elephant":        [105, 105, 105],   # Dim Gray
    "bear":            [ 92,  64,  51],   # Coffee Brown
    "zebra":           [240, 240, 240],   # Light Gray
    "giraffe":         [218, 165,  32],   # Goldenrod
    # Temporary Road Objects
    "cone":            [255, 140,   0],   # Dark Orange
    "traffic cone":    [255, 140,   0],   # Dark Orange
    "barrier":         [255, 215,   0],   # Gold
    "bollard":         [255, 255, 255],   # White
}