"""
This module defines the input channels for the RagbaarNet Processor.
Each channel processes raw input data and converts it into a normalized observation format for the Detector.
"""

import os
import cv2
import numpy as np
import zlib
import hashlib
import base64
import colorsys
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from modules.utils.logging_setup import setup_logging, set_level
logger = setup_logging("INFO", name="Platform.channels")


class BaseChannel(ABC):
    """
    One Channel = one input modality Processor can run with.

    A Channel's to_observation() takes one raw queued item and returns:
        (detector_input, display_payload)
    """

    name: str = "base"
    expected_kind: str = "frame"   # 'frame' (via add_frame) or 'event' (via add_event)
    detector_strategy: Optional[str] = None

    @abstractmethod
    def to_observation(self, item: Dict[str, Any]) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
        pass
    
    def set_debug_mode(self, enable: bool) -> None:
        """Update this channel's debug flag and its logger level at runtime."""
        
        self.debug_mode = enable
        set_level(logger, "DEBUG" if enable else "INFO")

class DrivingPipeline(BaseChannel):
    """
    Video frames -> Driving Model -> (result, roi) for Detector().
    """

    name = "driving"
    expected_kind = "frame"
    detector_strategy = "roi-events"
    alternate_strategy = "image-captioning"

    def __init__(self):
        """
        Initialize the driving channel.
        """
        from modules.Models.Segmentation.Segmentor import Segmentor

        self.processing_interval = int(os.environ.get('RAGBAARNET_DRIVING_INTERVAL', '1'))
        self._tick = 0

        max_side_raw = os.environ.get('RAGBAARNET_PROCESSING_MAX_SIDE', '').strip()
        self.processing_max_side = int(max_side_raw) if max_side_raw.isdigit() else None

        self.debug_mode = False

        self.encode_params = [cv2.IMWRITE_JPEG_QUALITY, 75]
        self._last_overlay_b64 = None
        self._last_overlay_hash = None
        self._color_mapping_cache: Dict[tuple, np.ndarray] = {}

        # self.sensing_mode = None    # Mode (1-Segmentation 2-Captioning)

        logger.info("🔄 Initializing segmentation model...")
        try:
            model_type = os.environ.get('RAGBAARNET_SEGMENTATION_MODEL', 'yolo').strip().lower()
            model_path = os.environ.get('RAGBAARNET_SEGMENTATION_MODEL_PATH', '').strip()

            if model_type == 'yolo':
                if not model_path:
                    model_path = os.path.join(
                        os.path.dirname(__file__), '..', 'Models', 'Segmentation',
                        'Pre-trained Models', 'yolo26', 'yolo26n-seg.pt',
                    )
                self.segmentor = Segmentor('yolo', model_path=model_path)
                logger.info("✅ YOLO Segmentor initialized successfully")
            elif model_type == 'segformer':
                if not model_path:
                    model_path = os.environ.get(
                        'RAGBAARNET_SEGFORMER_PATH',
                        os.path.abspath(os.path.join(
                            os.path.dirname(__file__), '..', 'Models', 'Segmentation',
                            'Pre-trained Models', 'segformer-b2-finetuned-cityscapes-1024-1024',
                        ))
                    )
                self.segmentor = Segmentor('segformer', model_path=model_path)
                logger.info("✅ SegFormer Segmentor initialized successfully")
            else:
                self.segmentor = None
        except Exception as e:
            logger.exception("❌ Error initializing segmentor: %s", e)
            self.segmentor = None

        logger.info("✅ Driving channel initialized successfully")

    def _create_consistent_color_map(self, class_labels=None):
        """
        Create a deterministic color map for segmentation labels.
        """

        labels = []
        for label in (class_labels or []):
            label = str(label).strip().lower().replace("_", " ").replace("-", " ")
            labels.append(label)

        palette = {
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

        def hashed_color(label: str):
            digest = hashlib.md5(label.encode("utf-8")).digest()
            hue = digest[0] / 255.0
            saturation = 0.65 + (digest[1] / 255.0) * 0.30
            value = 0.75 + (digest[2] / 255.0) * 0.20
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            return [int(r * 255), int(g * 255), int(b * 255)]

        color_map = {}
        for class_id, label in enumerate(labels):
            color_map[class_id] = palette.get(label) or hashed_color(label)
        color_map[255] = [0, 0, 0]

        if self.debug_mode and labels:
            logger.debug("🎨 Generated deterministic color map for %d classes.", len(labels))

        return color_map

    def _get_color_mapping_array(self, class_labels=None):
        """
        Return a cached lookup table for the current label set.
        """

        key = tuple(str(label) for label in (class_labels or []))
        if key in self._color_mapping_cache:
            return self._color_mapping_cache[key]

        color_map = self._create_consistent_color_map(class_labels)
        mapping = np.zeros((256, 3), dtype=np.uint8)
        for class_id, color in color_map.items():
            if color is not None:
                mapping[class_id] = color

        self._color_mapping_cache[key] = mapping
        return mapping

    def _derive_detected_classes(self, segmentation_map, class_labels=None):
        """
        Build a stable list of class names from a segmentation map and model labels.
        """

        labels = list(class_labels or [])
        if not labels or segmentation_map is None:
            return []
        
        try:
            unique_ids = np.unique(np.asarray(segmentation_map))
        except Exception:
            return []
        detected = []
        for class_id in unique_ids:
            class_id_int = int(class_id)
            if 0 <= class_id_int < len(labels) and labels[class_id_int]:
                detected.append(labels[class_id_int])

        return sorted(set(detected))

    def _validate_segmentation_map(self, seg_map):
        """
        Normalize and validate segmentation map into a 2D uint8 index array.
        """

        arr = np.asarray(seg_map)
        if arr.ndim == 3:
            arr = arr.squeeze(2) if arr.shape[2] == 1 else arr[..., 0]
        arr = np.rint(arr).astype(np.int32) if np.issubdtype(arr.dtype, np.floating) else arr.astype(np.int32)
        if arr.size == 0:
            return np.zeros((0, 0), dtype=np.uint8)
        return np.clip(arr, 0, 255).astype(np.uint8)

    def _create_segmentation_overlay(self, frame, result):
        """
        Create an optimized visualization overlay for the segmentation result.
        """

        try:
            segmentation_map = getattr(result, 'segmentation_map', None)
            if segmentation_map is None:
                logger.warning("⚠️ No segmentation_map present in result; returning original frame")
                return frame
            
            try:
                segmentation_map = self._validate_segmentation_map(segmentation_map)
            except Exception:
                segmentation_map = np.clip(np.asarray(segmentation_map, dtype=np.int32), 0, 255).astype(np.uint8)

            class_labels = list(getattr(result, 'class_labels', None) or [])
            if not class_labels and self.segmentor is not None:
                try:
                    class_labels = self.segmentor.get_class_labels()
                except Exception:
                    class_labels = []

            color_mapping_array = self._get_color_mapping_array(class_labels)
            overlay = color_mapping_array[segmentation_map]

            if overlay.shape[:2] != frame.shape[:2]:
                overlay = cv2.resize(overlay, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

            blended = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)
            blended = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
            return blended
        
        except Exception as e:
            logger.exception("❌ Error creating segmentation overlay: %s", e)
            return frame

    def to_observation(self, item):
        if self.segmentor is None:
            return None, None

        self._tick += 1
        if (self._tick % max(1, self.processing_interval)) != 0:
            return None, None  # skip this frame for performance; queue was still drained

        frame = item['frame']
        roi_points = item.get('roi_points')
        roi_controls = item.get('roi_controls')

        try:
            seg_frame = frame
            orig_h, orig_w = frame.shape[:2]

            if self.processing_max_side is not None:
                max_side = max(orig_h, orig_w)
                if max_side > self.processing_max_side:
                    scale = self.processing_max_side / float(max_side)
                    new_w, new_h = max(1, int(orig_w * scale)), max(1, int(orig_h * scale))
                    seg_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            result = self.segmentor(seg_frame)

            if seg_frame is not frame:
                try:
                    result.segmentation_map = cv2.resize(
                        result.segmentation_map, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                    if result.confidence_map is not None:
                        result.confidence_map = cv2.resize(
                            result.confidence_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

                    scale_x, scale_y = orig_w / float(seg_frame.shape[1]), orig_h / float(seg_frame.shape[0])
                    for detected_object in result.bounding_boxes:
                        x1, y1, x2, y2 = detected_object['bbox']
                        detected_object['bbox'] = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
                        if 'centroid' in detected_object:
                            cx, cy = detected_object['centroid']
                            detected_object['centroid'] = (cx * scale_x, cy * scale_y)

                    if isinstance(result.masks, dict):
                        result.masks = {
                            key: cv2.resize(np.asarray(mask, dtype=np.uint8), (orig_w, orig_h),
                                             interpolation=cv2.INTER_NEAREST).astype(bool)
                            for key, mask in result.masks.items()
                        }

                    result.segmentation_map = self._validate_segmentation_map(result.segmentation_map)

                    if self.debug_mode:
                        logger.debug("↔️ Resized segmentation outputs from %dx%d to original frame size (%dx%d)", 
                                     seg_frame.shape[1], seg_frame.shape[0], orig_w, orig_h)

                except Exception as resize_err:
                    logger.exception("❌ Failed to resize segmentation outputs: %s", resize_err)

            class_labels = list(getattr(result, 'class_labels', None) or [])
            if not class_labels and self.segmentor is not None:
                class_labels = self.segmentor.get_class_labels()
            detected_classes = self._derive_detected_classes(result.segmentation_map, class_labels)

            overlay = self._create_segmentation_overlay(frame, result)
            try:
                overlay_hash = zlib.crc32(overlay.tobytes())
            except Exception:
                overlay_hash = None
            if overlay_hash is None or overlay_hash != self._last_overlay_hash or self._last_overlay_b64 is None:
                _, buffer = cv2.imencode('.jpg', overlay, self.encode_params)
                self._last_overlay_b64 = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
                self._last_overlay_hash = overlay_hash

            display_payload = {
                'overlay_b64': self._last_overlay_b64,
                'class_labels': result.class_labels,
                'detected_classes': detected_classes,
                'model_type': (result.metadata or {}).get('model_type'),
            }
            # detector_input shaped exactly how ROIEventsDetector unpacks it: "result, roi = input"
            detector_input = (result, {'corners': roi_points, 'controls': roi_controls})

            return detector_input, display_payload

        except Exception as e:
            logger.exception("❌ Error in segmentation channel: %s", e)
            return None, None

class TypingPipeline(BaseChannel):
    """
    Keystrokes -> normalized observation -> Detector('key-events') -> NOTE_ON/NOTE_OFF.
    """

    name = "typing"
    expected_kind = "event"
    detector_strategy = "key-events"
    alternate_strategy = None

    _KEY_CLASS_MAP: Dict[str, str] = {chr(c): "typing_letter" for c in range(ord('a'), ord('z') + 1)}
    _KEY_CLASS_MAP.update({str(d): "typing_digit" for d in range(10)})
    _KEY_CLASS_MAP.update({
        "backspace": "typing_delete", "enter": "typing_newline",
        "tab": "typing_indent", "space": "typing_space",
        "scroll": "scroll",
        "mousemove": "mousemove",
    })

    def __init__(self):
        self.debug_mode = False
        logger.info("✅ Typing channel initialized successfully")

    def to_observation(self, item):
        payload = item.get('raw_payload') or {}
        kind = payload.get('type')

        if kind == 'keydown':
            key = str(payload.get('key', '')).lower()
            return {
                'kind': 'onset',
                'key': key,
                'class_name': self._KEY_CLASS_MAP.get(key, 'typing_other'),
                'intensity': 1.0,
            }, None

        if kind == 'keyup':
            key = str(payload.get('key', '')).lower()
            return {
                'kind': 'release',
                'key': key,
                'class_name': self._KEY_CLASS_MAP.get(key, 'typing_other'),
                'intensity': 1.0,
            }, None

        logger.warning("⚠️ Unrecognized typing-channel payload type: %s", kind)
        return None, None


AVAILABLE_CHANNELS: Dict[int, type] = {
    1: DrivingPipeline,
    2: TypingPipeline,
}
