"""
Modular Processing Framework for Turning Sequential Input into Music
======================================================================

This module implements the Processor class, which orchestrates the flow of data from input sources 
through a selected processing channel and into a music generation pipeline. 
The architecture is designed to be modular, allowing for different input modalities to be processed 
by different channels, each with its own detection strategy.
"""

import os
import sys
import cv2
import numpy as np
import time
import threading
import traceback
import base64
import zlib
import hashlib
import colorsys
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
from queue import Queue, Empty

from modules.Detection.Detector import Detector
from modules.Music_Generator.Musician import Musician
from modules.Platform.midi_output import MidiOutput
from modules.utils.logging_setup import setup_logging, set_level

logger = setup_logging("INFO", name="Platform.Processor")


# ============================================================================
# Channels
# ============================================================================

class BaseChannel(ABC):
    """
    One Channel = one input modality Processor can run with. Exactly one
    Channel is active per run (see architecture note above).

    A Channel's to_observation() takes one raw queued item and returns:
        (detector_input, display_payload)
    """

    name: str = "base"
    expected_kind: str = "frame"   # 'frame' (via add_frame) or 'event' (via add_event)
    detector_strategy: Optional[str] = None

    @abstractmethod
    def to_observation(self, item: Dict[str, Any]) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
        raise NotImplementedError

class DrivingPipeline(BaseChannel):
    """
    Video frames -> Driving Model -> (result, roi) for Detector().
    """

    name = "driving"
    expected_kind = "frame"
    detector_strategy = "roi-events"

    def __init__(self):
        """
        Initialize the driving channel.
        """
        from modules.Models.Segmentation.Segmentor import Segmentor

        # How many queued frames to skip between actual driving model runs.
        self.processing_interval = int(os.environ.get('RAGBAARNET_DRIVING_INTERVAL', '2'))
        self._tick = 0

        # Mode (1-Segmentation 2-Captioning)
        self.processing_mode = None

        max_side_raw = os.environ.get('RAGBAARNET_PROCESSING_MAX_SIDE', '').strip()
        self.processing_max_side = int(max_side_raw) if max_side_raw.isdigit() else None

        self.debug_mode = False
        self.encode_params = [cv2.IMWRITE_JPEG_QUALITY, 75]
        self._last_overlay_b64 = None
        self._last_overlay_hash = None
        self._color_mapping_cache: Dict[tuple, np.ndarray] = {}

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

    def set_debug_mode(self, enabled: bool) -> None:
        self.debug_mode = enabled

    # --- colormap helpers --------------------------------------------------

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
            logger.debug(
                "🎨 Generated deterministic color map for %d classes.",
                len(labels)
            )

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
                if self.debug_mode:
                    logger.debug("⚠️ No segmentation_map present in result; returning original frame")
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

    # --- main entry point ----------------------------------------------------

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
                except Exception as resize_err:
                    if self.debug_mode:
                        logger.warning("❌ Failed to resize segmentation outputs: %s", resize_err)

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

            # detector_input shaped exactly how ROIEventsDetector.__call__ unpacks it:
            #   result, roi = input
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

    def set_debug_mode(self, enabled: bool) -> None:
        self.debug_mode = enabled

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

        if self.debug_mode:
            logger.debug("⚠️ Unrecognized typing-channel payload type: %s", kind)
        return None, None


AVAILABLE_CHANNELS: Dict[int, type] = {
    1: DrivingPipeline,
    2: TypingPipeline,
}

# ============================================================================
# Processor
# ============================================================================

class Processor:
    """
    Orchestrates turning queued input into music via exactly one active Channel
    and one shared Detector. See the module-level architecture note above.
    """

    def __init__(self, socketio_instance=None):
        """
        Initialize the Processor, including selecting the active channel and starting the processing loop.
        """

        self.socketio = socketio_instance
        self.frame_counter = 0

        # Single queue for whichever kind of item the active channel expects.
        self.input_queue = Queue(maxsize=10)

        self.current_frame = None
        self.current_display = None
        self.is_processing = False

        self.debug_mode = False
        self.last_debug_time = 0
        self.debug_interval = 10.0

        self.main_ui_connected = False
        self.status_page_clients = set()

        self._shutdown_lock = threading.Lock()
        self._is_shutdown = False

        logger.info("🔄 Initializing music generation platfom...")
        try:
            self.musician = Musician('lstm-onessen-orchestral', tempo=120, key_signature="C_major", time_signature=(4,4))
            self.music_queue = Queue(maxsize=5)
            self.current_music = None
            self.music_enabled = True
            logger.info("✅ Music Generator initialized successfully")
        except Exception as e:
            logger.exception("❌ Error initializing musician: %s", e)
            self.musician = None
            self.music_enabled = False

        self.audio_backend = os.environ.get('RAGBAARNET_AUDIO_BACKEND', 'tone').strip().lower()
        if self.audio_backend not in ('tone', 'midi', 'both'):
            logger.warning("⚠️ Invalid audio backend '%s' - defaulting to 'tone'.", self.audio_backend)
            self.audio_backend = 'tone'
        self.midi_output = None
        if self.audio_backend in ('midi', 'both'):
            try:
                logger.info("🔄 Initializing MIDI output backend...")
                self.midi_output = MidiOutput()
            except Exception as e:
                logger.exception("❌ Error initializing MIDI output: %s", e)
                self.midi_output = None

        # Pick exactly one channel for this run and the single Detector strategy that goes with it.
        self.channel = self._select_channel()
        self.detector = Detector(self.channel.detector_strategy)

        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()

    def _select_channel(self) -> BaseChannel:
        """
        Prompt once at startup for which input modality this run will use.
        """

        print("CHOOSE A PROCESSING CHANNEL (this cannot be changed without restarting):")
        for key, cls in AVAILABLE_CHANNELS.items():
            print(f"  {key}. {cls.name}")

        choice_raw = input("Enter your choice (1-2) [default: 1 - driving]: ").strip()
        try:
            choice = int(choice_raw) if choice_raw else 1
        except ValueError:
            choice = None

        channel_cls = AVAILABLE_CHANNELS.get(choice)
        if channel_cls is None:
            logger.warning("⚠️ Invalid choice '%s' - defaulting to 'driving'.", choice_raw)
            channel_cls = DrivingPipeline

        logger.info("🔄 Initializing '%s' channel...", channel_cls.name)
        return channel_cls()

    # --- ingress ---------------------------------------------------------------

    def add_frame(self, frame, frame_id=None, timestamp=None, roi_points=None, roi_controls=None):
        """
        Queue a video frame. Only consumed if the active channel expects 'frame'.
        """
        
        self._enqueue({
            'kind': 'frame',
            'frame': frame,
            'frame_id': frame_id or f"frame_{self.frame_counter}",
            'timestamp': timestamp if timestamp is not None else time.time(),
            'roi_points': roi_points,
            'roi_controls': roi_controls,
        })

    def add_event(self, source_name: str, raw_payload: dict, timestamp=None):
        """
        Queue a generic event (keystroke, scroll, ...). Only consumed if the active channel expects 'event'.
        """

        self._enqueue({
            'kind': 'event',
            'source_name': source_name,
            'raw_payload': raw_payload,
            'timestamp': timestamp if timestamp is not None else time.time(),
        })

    def _enqueue(self, item: Dict[str, Any]):
        if item['kind'] != self.channel.expected_kind:
            logger.warning(
                "⚠️ Ignoring '%s' input - active channel is '%s', which expects '%s' input. "
                "Restart with a different channel choice to use this input instead.",
                item['kind'], self.channel.name, self.channel.expected_kind,
            )
            return

        if self.input_queue.full():
            try:
                self.input_queue.get_nowait()
            except Empty:
                pass

        self.input_queue.put(item)

    # --- processing loop ---------------------------------------------------------

    def _processing_loop(self):
        """
        Main loop that continuously processes items from the input queue, applies the selected channel's
        logic, and updates the display and music generation accordingly.
        """
        logger.info("🚀 Processing loop started (channel: %s)", self.channel.name)

        while True:
            try:
                item = self.input_queue.get(timeout=1.0)
                if item is None:  # Shutdown signal
                    break

                if item['kind'] == 'frame':
                    self.current_frame = item['frame']

                self.channel.set_debug_mode(self.debug_mode)

                try:
                    detector_input, display_payload = self.channel.to_observation(item)
                except Exception as e:
                    logger.exception("❌ Error in channel '%s': %s", self.channel.name, e)
                    detector_input, display_payload = None, None

                item_id = item.get('frame_id') if item['kind'] == 'frame' else item.get('source_name')

                if display_payload is not None:
                    display_payload = {
                        **display_payload,
                        'frame_id': item_id,
                        'timestamp': item['timestamp'],
                        'frame_counter': self.frame_counter,
                    }
                    self.current_display = display_payload
                    self._broadcast_display_update()

                if detector_input is not None:
                    scene_events = []
                    try:
                        scene_events = self.detector(input=detector_input, frame_id=self.frame_counter)
                    except Exception as detector_err:
                        logger.error("❌ Detector error: %s", detector_err)
                        logger.error("Traceback:\n%s", traceback.format_exc())

                    if scene_events:
                        self._generate_and_broadcast_music(
                            scene_events,
                            frame_id=item_id,
                            timestamp=item['timestamp'],
                            state=getattr(self.detector.detector, 'state', None),
                        )

                self.frame_counter += 1

            except Empty:
                continue
            except Exception as e:
                logger.exception("❌ Error in processing loop: %s", e)

    # --- shared musician plumbing -------------------------------------------------

    def _generate_and_broadcast_music(self, scene_events, frame_id, timestamp, state):
        """
        Generate music based on scene events and broadcast the update.
        """

        if not (self.music_enabled and self.musician is not None):
            return

        try:
            music_frame = self.musician(results=scene_events, frame_id=self.frame_counter, state=state)

            if self.midi_output is not None:
                self.midi_output.send_music_frame(music_frame)

            music_data = {
                'frame_id': frame_id,
                'timestamp': timestamp,
                'frame_counter': self.frame_counter,
                'music_frame': music_frame,
                'events_count': len(music_frame.events),
                'tempo': self.musician.tempo,
                'key_signature': self.musician.key_signature,
                'time_signature': self.musician.time_signature,
            }

            if self.music_queue.full():
                try:
                    self.music_queue.get_nowait()
                except Empty:
                    pass

            self.music_queue.put(music_data)
            self.current_music = music_data
            self._broadcast_music_update(music_data)

            if self.debug_mode and (time.time() - self.last_debug_time) > self.debug_interval:
                logger.debug("🎵 Generated %s music events for frame %s", len(music_frame.events), self.frame_counter)

        except Exception as music_err:
            logger.error("❌ Error generating music: %s", music_err)
            logger.error("Traceback:\n%s", traceback.format_exc())

    def _broadcast_display_update(self):
        """
        Immediately broadcast the current display state to connected clients.
        """

        try:
            if self.main_ui_connected and self.socketio:
                display_data = self.get_synchronized_display(for_main_ui=True)
                state = self.get_current_state()
                response_data = {**display_data, 'queue_size': state['queue_size']}
                self.socketio.emit('frame_update', response_data)

                if self.debug_mode and (time.time() - self.last_debug_time) > self.debug_interval:
                    logger.debug("📡 Broadcasted display update for frame %s", self.frame_counter)
        except Exception as e:
            if self.debug_mode:
                logger.warning("❌ Error broadcasting display update: %s", e)

    def _broadcast_music_update(self, music_data):
        """
        Broadcast music events to connected WebSocket clients.
        """

        try:
            if self.main_ui_connected and self.socketio:
                music_frame = music_data['music_frame']
                events_data = []

                for event in music_frame.events:
                    instrument_name = event.instrument
                    if instrument_name in ('unknown', None, ''):
                        logger.error("❌ Event has unknown instrument: %s", event)

                    events_data.append({
                        'event_type': event.event_type,
                        'note': event.note,
                        'channel': event.channel,
                        'velocity': event.velocity,
                        'instrument': instrument_name,
                        'timestamp': event.timestamp,
                    })

                music_response = {
                    'frame_id': music_data['frame_id'],
                    'frame_counter': music_data['frame_counter'],
                    'events': events_data,
                    'events_count': music_data['events_count'],
                    'tempo': music_data['tempo'],
                    'key_signature': music_data['key_signature'],
                    'time_signature': music_data['time_signature'],
                    'timestamp': music_data['timestamp'],
                    'audio_backend': self.audio_backend,
                }

                self.socketio.emit('music_update', music_response)

                if self.debug_mode and (time.time() - self.last_debug_time) > self.debug_interval:
                    logger.debug("🎵 Broadcasted music update: %s events for frame %s",
                                 len(events_data), music_data['frame_counter'])
        except Exception as e:
            if self.debug_mode:
                logger.warning("❌ Error broadcasting music update: %s", e)

    # --- state / display queries used by main.py's routes --------------------------

    def get_current_state(self):
        """
        Return a dictionary representing the current state of the processor.
        """

        return {
            'frame_counter': self.frame_counter,
            'current_frame_available': self.current_frame is not None,
            'current_display_available': self.current_display is not None,
            'current_music_available': self.current_music is not None if hasattr(self, 'current_music') else False,
            'music_enabled': self.music_enabled if hasattr(self, 'music_enabled') else False,
            'active_channel': getattr(self.channel, 'name', None),
            'queue_size': self.input_queue.qsize(),
            'music_queue_size': self.music_queue.qsize() if hasattr(self, 'music_queue') else 0,
        }

    def get_synchronized_display(self, for_main_ui=True):
        """
        Get synchronized frame and segmentation data for display.
        """

        display_data = {
            'original_frame': None,
            'segmentation_overlay': None,
            'segmentation_info': None,
            'music_info': None,
            'frame_counter': self.frame_counter,
            'timestamp': time.time(),
        }

        if self.current_display is not None:
            payload = self.current_display
            frame_diff = self.frame_counter - payload['frame_counter']

            if frame_diff <= 10:
                if for_main_ui:
                    display_data['segmentation_overlay'] = payload.get('overlay_b64')

                display_data['segmentation_info'] = {
                    'frame_id': payload['frame_id'],
                    'timestamp': payload['timestamp'],
                    'frame_counter': payload['frame_counter'],
                    'frames_since_segmentation': frame_diff,
                    'class_labels': payload.get('detected_classes') or [],
                    'model_type': payload.get('model_type'),
                }

        if hasattr(self, 'current_music') and self.current_music is not None:
            music_data = self.current_music
            frame_diff = self.frame_counter - music_data['frame_counter']

            if frame_diff <= 10:
                display_data['music_info'] = {
                    'frame_id': music_data['frame_id'],
                    'frame_counter': music_data['frame_counter'],
                    'events_count': music_data['events_count'],
                    'tempo': music_data['tempo'],
                    'key_signature': music_data['key_signature'],
                    'time_signature': music_data['time_signature'],
                    'frames_since_music': frame_diff,
                    'timestamp': music_data['timestamp'],
                }

        return display_data

    # --- music controls -------------------------------------------------------------

    def toggle_music_generation(self, enable: bool = None):
        """Enable or disable music generation"""

        if hasattr(self, 'music_enabled'):
            self.music_enabled = (not self.music_enabled) if enable is None else enable
            logger.info(f"🎵 Music generation {'enabled' if self.music_enabled else 'disabled'}")
            return self.music_enabled
        
        return False

    def set_music_tempo(self, tempo: int):
        """Set music tempo (BPM)"""

        if hasattr(self, 'musician') and self.musician is not None:
            self.musician.set_tempo(tempo)
            logger.info(f"🎵 Music tempo set to {tempo} BPM")
            return True
        return False

    def set_music_key(self, key_signature: str):
        """Set music key signature"""

        if hasattr(self, 'musician') and self.musician is not None:
            self.musician.key_signature = key_signature
            logger.info(f"🎵 Music key signature set to {key_signature}")
            return True
        return False

    def set_music_time(self, time_signature: tuple):
        """Set music time signature"""

        if hasattr(self, 'musician') and self.musician is not None:
            self.musician.time_signature = time_signature
            logger.info(f"🎵 Music time signature set to {time_signature}")
            return True
        return False

    def get_music_status(self):
        """Get current music generation status"""

        if hasattr(self, 'musician') and self.musician is not None:
            return {
                'enabled': getattr(self, 'music_enabled', False),
                'tempo': self.musician.tempo,
                'key_signature': self.musician.key_signature,
                'time_signature': self.musician.time_signature,
                'musician_type': self.musician.musician_type,
                'instrument': self.musician.instrument,
                'queue_size': self.music_queue.qsize() if hasattr(self, 'music_queue') else 0,
            }
        return {'enabled': False, 'musician_available': False}

    def get_available_musicians(self):
        """Get the list of musician types the UI can offer, plus the current selection"""

        try:
            musicians = Musician.list_available_musicians()
        except Exception as e:
            logger.exception("❌ Error listing available musicians: %s", e)
            musicians = []

        current = None
        instrument = 'piano'
        if hasattr(self, 'musician') and self.musician is not None:
            current = self.musician.musician_type
            instrument = self.musician.instrument

        return {'musicians': musicians, 'current': current, 'instrument': instrument}

    def apply_music_settings(self, musician_type: str, tempo: int, instrument: str):
        """Apply musician, tempo, and LSTM instrument settings together."""

        if not hasattr(self, 'musician') or self.musician is None:
            return {'success': False, 'error': 'Musician system not initialized'}

        try:
            tempo = int(tempo)
            if not 60 <= tempo <= 300:
                raise ValueError('Tempo must be between 60 and 300 BPM')

            if musician_type != self.musician.musician_type:
                self.musician.switch_musician(musician_type, tempo=tempo, instrument=instrument)
            else:
                self.musician.set_tempo(tempo)
                if musician_type == 'lstm-onessen':
                    self.musician.set_instrument(instrument)

            return {
                'success': True,
                'musician_type': self.musician.musician_type,
                'tempo': self.musician.tempo,
                'instrument': self.musician.instrument,
            }
        except Exception as e:
            logger.error(f"❌ Error applying music settings: {e}")
            return {'success': False, 'error': str(e)}

    def switch_musician(self, musician_type: str):
        """Switch to a different music generation model (keeps current tempo/key/timesign)"""

        if not hasattr(self, 'musician') or self.musician is None:
            return {'success': False, 'error': 'Musician system not initialized'}

        try:
            self.musician.switch_musician(musician_type)
            return {'success': True, 'musician_type': self.musician.musician_type}
        except Exception as e:
            logger.error(f"❌ Error switching musician: {e}")
            return {'success': False, 'error': str(e)}

    # --- lifecycle / misc -------------------------------------------------------------

    def shutdown(self):
        """Shutdown the processor safely and only once."""

        with self._shutdown_lock:

            if self._is_shutdown:
                logger.debug("Processor shutdown already completed.")
                return
            
            self.midi_output.close()

            self._is_shutdown = True

            logger.info("🎼 Saving generated music...")
            self.musician.save_generated_melody()

            logger.info("🛑 Shutting down Main processor...")
            # Send shutdown signal to the processing loop
            try:
                self.input_queue.put(None)
            except Exception:
                logger.exception("❌ Failed to send shutdown signal to input queue.")

        # Wait for processing thread outside the lock
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)

            if self.processing_thread.is_alive():
                logger.warning("⚠️ Processing thread did not stop within the timeout.")
            else:
                logger.info("✅ Processing thread stopped.")

        logger.info("✅ Processor shutdown complete.")

    def enable_debug_mode(self, enable=True):
        """Enable or disable debug mode for verbose logging"""

        self.debug_mode = enable
        set_level(logger, "DEBUG" if enable else "INFO")
        logger.info("🐛 Debug mode %s", "enabled - verbose logging activated" if enable else "disabled - minimal logging activated")

    def set_main_ui_connected(self, connected=True):
        """Mark main UI as connected/disconnected to prioritize it over status page"""

        if self.main_ui_connected != connected:
            self.main_ui_connected = connected
            logger.info("🎯 Main UI connected - prioritizing data for main interface" if connected else "📄 Main UI disconnected")
        else:
            self.main_ui_connected = connected
