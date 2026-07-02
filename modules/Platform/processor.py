"""
Modular Processing Framework for Receiving Data and Performing Segmentation and Generating Music
=================================================

This module receives data from UI.html and processes it using the Segmentor class.
Then sends the processed data back to UI.html for Generating Music.
"""

import cv2
import numpy as np
import base64
import time
import threading
import argparse
import colorsys
import zlib
import os
import sys
from queue import Queue, Empty
from flask import Flask, request, jsonify, send_from_directory, redirect
from flask_socketio import SocketIO, emit
from flask_cors import CORS

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Segmentation.Segmentor import Segmentor
from Music_Generator.Musician import Musician
from utils.logging_setup import setup_logging, set_level

logger = setup_logging("INFO", name="Platform.Processor")

class Processor:
    """
    Video Processing Class that handles frame reception, segmentation, and synchronization
    """

    def __init__(self, socketio_instance=None):
        """Initialize the video processor with segmentation models"""

        self.socketio = socketio_instance  # Store socketio instance for broadcasting
        self.frame_counter = 0
        # Process segmentation every N frames (higher => higher FPS, lower segmentation refresh rate)
        self.segmentation_interval = int(os.environ.get('RAGBAARNET_SEGMENTATION_INTERVAL', '2'))

        # Optional downscale for segmentation input to improve FPS (keeps output resized back to original).
        # Example: set RAGBAARNET_SEGMENTATION_MAX_SIDE=512
        max_side_raw = os.environ.get('RAGBAARNET_SEGMENTATION_MAX_SIDE', '').strip()
        self.segmentation_max_side = int(max_side_raw) if max_side_raw.isdigit() else None
        self.frame_queue = Queue(maxsize=10)
        self.segmentation_queue = Queue(maxsize=5)
        self.current_frame = None
        self.current_segmentation = None
        self.is_processing = False

        # Cache for last encoded overlay to avoid re-encoding on every websocket tick
        self._last_overlay_b64 = None
        self._last_overlay_counter = -1
        self._last_overlay_hash = None

        # Performance optimization flags
        self.debug_mode = False
        self.last_debug_time = 0
        self.debug_interval = 5.0

        # Connection management to avoid dual streaming conflicts
        self.main_ui_connected = False
        self.status_page_clients = set()

        # Pre-compute color mapping arrays for faster lookup
        self.color_mapping_array = None
        self._color_mapping_cache = {}
        self.color_map = self._create_consistent_color_map()
        self.color_mapping_array = self._get_color_mapping_array()

        # Cache for image encoding to avoid repeated allocations
        self.encode_params = [cv2.IMWRITE_JPEG_QUALITY, 75]

        # Initialize segmentation models
        logger.info("🔄 Initializing segmentation models...")
        try:
            model_type = os.environ.get('RAGBAARNET_SEGMENTATION_MODEL', 'segformer').strip().lower() or 'segformer'
            model_path = os.environ.get('RAGBAARNET_SEGMENTATION_MODEL_PATH', '').strip()

            if model_type == 'yolo':
                if not model_path:
                    model_path = os.path.join(
                        os.path.dirname(__file__),
                        '..',
                        'Segmentation',
                        'Pre-trained Models',
                        'yolo11',
                        'yolo11s-seg.pt',
                    )
                self.segmentor = Segmentor('yolo', model_path=model_path)
                logger.info("✅ YOLO Segmentor initialized successfully")
            else:
                if not model_path:
                    model_path = os.environ.get(
                        'RAGBAARNET_SEGFORMER_PATH',
                        os.path.abspath(
                            os.path.join(
                                os.path.dirname(__file__),
                                '..',
                                'Segmentation',
                                'Pre-trained Models',
                                'segformer-b2-finetuned-cityscapes-1024-1024',
                            )
                        )
                    )
                self.segmentor = Segmentor('segformer', model_path=model_path)
                logger.info("✅ SegFormer Segmentor initialized successfully")
        except Exception as e:
            logger.exception("❌ Error initializing segmentor: %s", e)
            self.segmentor = None

        # Initialize music generation
        logger.info("🔄 Initializing music generation...")
        try:
            self.musician = Musician('lstm-onEssen', tempo=120, key_signature="C_major")
            self.music_queue = Queue(maxsize=5)
            self.current_music = None
            self.music_enabled = True
            logger.info("✅ Music Generator initialized successfully")
        except Exception as e:
            logger.exception("❌ Error initializing musician: %s", e)
            self.musician = None
            self.music_enabled = False

        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()

    def _create_consistent_color_map(self, class_labels=None):
        """Create a deterministic color mapping for any segmentation label set."""

        labels = [str(label).strip() for label in (class_labels or []) if str(label).strip()]
        normalized_labels = [label.lower() for label in labels]

        # Preserve the familiar Cityscapes palette for the common semantic labels.
        palette = {
            "road": [128, 64, 128],
            "sidewalk": [244, 35, 232],
            "building": [70, 70, 70],
            "wall": [102, 102, 156],
            "fence": [190, 153, 153],
            "pole": [153, 153, 153],
            "traffic light": [250, 170, 30],
            "traffic sign": [220, 220, 0],
            "vegetation": [107, 142, 35],
            "terrain": [152, 251, 152],
            "sky": [70, 130, 180],
            "person": [220, 20, 60],
            "rider": [255, 0, 0],
            "car": [0, 0, 142],
            "truck": [0, 0, 70],
            "bus": [0, 60, 100],
            "train": [0, 80, 100],
            "motorcycle": [0, 0, 230],
            "bicycle": [119, 11, 32],
            "parking": [160, 160, 160],
            "rail track": [230, 150, 140],
            "on rails": [128, 128, 128],
            "caravan": [0, 0, 90],
            "trailer": [0, 0, 110],
            "guard rail": [180, 165, 180],
            "bridge": [150, 100, 100],
            "tunnel": [150, 120, 90],
            "pole group": [153, 153, 153],
            "ground": [81, 0, 81],
            "dynamic": [111, 74, 0],
            "static": [81, 81, 81],
        }

        color_map = {}
        for class_id, label in enumerate(normalized_labels):
            color_map[class_id] = palette.get(label, None)

        for class_id in range(len(normalized_labels), 255):
            hue = (class_id * 137.5) % 360
            saturation = 70 + (class_id % 3) * 15
            value = 180 + (class_id % 4) * 20
            r, g, b = colorsys.hsv_to_rgb(hue / 360.0, saturation / 100.0, value / 255.0)
            color_map[class_id] = [int(r * 255), int(g * 255), int(b * 255)]

        for class_id, label in enumerate(normalized_labels):
            if color_map[class_id] is None:
                hue = (class_id * 137.5) % 360
                saturation = 80
                value = 220
                r, g, b = colorsys.hsv_to_rgb(hue / 360.0, saturation / 100.0, value / 255.0)
                color_map[class_id] = [int(r * 255), int(g * 255), int(b * 255)]

        color_map[255] = [0, 0, 0]

        if self.debug_mode and labels:
            logger.debug("🎨 Color mapping generated for %s labels", len(labels))

        return color_map

    def _get_color_mapping_array(self, class_labels=None):
        """Return a cached lookup table for the current label set."""

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
        """Build a stable list of class names from a segmentation map and model labels."""

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
            if 0 <= class_id_int < len(labels):
                label = labels[class_id_int]
                if label:
                    detected.append(label)

        return sorted(set(detected))

    def _validate_segmentation_map(self, seg_map):
        """Normalize and validate segmentation map into a 2D uint8 index array.

        - Ensures 2D shape
        - Clips values to [0,255]
        - Converts floats to nearest integers
        """

        arr = np.asarray(seg_map)

        # Reduce channel dim if present (e.g., HxWx1)
        if arr.ndim == 3:
            if arr.shape[2] == 1:
                arr = arr.squeeze(2)
            else:
                if self.debug_mode:
                    logger.warning("⚠️ segmentation_map has %s channels; using first channel", arr.shape[2])
                arr = arr[..., 0]

        # Ensure numeric integer type
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.rint(arr).astype(np.int32)
        else:
            arr = arr.astype(np.int32)

        if arr.size == 0:
            return np.zeros((0, 0), dtype=np.uint8)

        minv = int(arr.min())
        maxv = int(arr.max())
        if (minv < 0) or (maxv > 255):
            if self.debug_mode:
                logger.warning("⚠️ segmentation_map values out of range: min=%s max=%s — clamping to [0,255]", minv, maxv)

        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    def _create_generic_color_mapping_array(self):
        """Create a generic per-class color mapping array for non-Cityscapes models (e.g., YOLO/COCO).

        Index 255 is the background/no-detection sentinel (black). All COCO class IDs (0-79)
        get deterministic HSV colors via the golden-angle hue spacing.
        """

        mapping = np.zeros((256, 3), dtype=np.uint8)
        mapping[255] = [0, 0, 0]  # 255 = background/no-detection sentinel → black

        for class_id in range(0, 255):  # start at 0 so COCO class 0 (person) gets a colour
            hue = (class_id * 137.5) % 360  # golden angle
            saturation = 80
            value = 220
            r, g, b = colorsys.hsv_to_rgb(hue / 360.0, saturation / 100.0, value / 255.0)
            mapping[class_id] = [int(r * 255), int(g * 255), int(b * 255)]

        return mapping

    def _processing_loop(self):
        """Main processing loop that runs in a separate thread"""
        logger.info("🚀 Processing loop started")

        while True:
            try:
                # Get frame from queue (timeout prevents blocking)
                frame_data = self.frame_queue.get(timeout=1.0)

                if frame_data is None:  # Shutdown signal
                    break

                frame = frame_data['frame']
                frame_id = frame_data['frame_id']
                timestamp = frame_data['timestamp']

                self.current_frame = frame

                # Process segmentation every N frames
                if self.frame_counter % self.segmentation_interval == 0 and self.segmentor is not None:
                    # Reduced logging for performance
                    if self.debug_mode and (time.time() - self.last_debug_time) > self.debug_interval:
                        logger.debug("🔍 Processing segmentation for frame %s", self.frame_counter)
                        self.last_debug_time = time.time()

                    try:
                        # Perform segmentation
                        seg_frame = frame
                        orig_h, orig_w = frame.shape[:2]

                        if self.segmentation_max_side is not None:
                            max_side = max(orig_h, orig_w)
                            if max_side > self.segmentation_max_side:
                                scale = self.segmentation_max_side / float(max_side)
                                new_w = max(1, int(orig_w * scale))
                                new_h = max(1, int(orig_h * scale))
                                seg_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                        result = self.segmentor(seg_frame)

                        # Derive a small, UI-friendly list of detected class names from the segmentation output.
                        detected_classes = []
                        try:
                            class_labels = list(getattr(result, 'class_labels', None) or [])
                            if not class_labels and getattr(self, 'segmentor', None) is not None:
                                class_labels = self.segmentor.get_class_labels()
                            detected_classes = self._derive_detected_classes(result.segmentation_map, class_labels)

                            if getattr(result, 'bounding_boxes', None):
                                detected_from_boxes = sorted({b.get('class_name') for b in result.bounding_boxes if b.get('class_name')})
                                detected_classes = sorted(set(detected_classes) | set(detected_from_boxes))
                        except Exception as cls_err:
                            if self.debug_mode:
                                logger.debug("Failed to derive detected classes: %s", cls_err)

                        # Resize outputs back to original frame size for consistent downstream processing.
                        if seg_frame is not frame:
                            try:
                                result.segmentation_map = cv2.resize(
                                    result.segmentation_map,
                                    (orig_w, orig_h),
                                    interpolation=cv2.INTER_NEAREST,
                                )
                                if result.confidence_map is not None:
                                    result.confidence_map = cv2.resize(
                                        result.confidence_map,
                                        (orig_w, orig_h),
                                        interpolation=cv2.INTER_LINEAR,
                                    )
                                # Validate and normalize segmentation map to safe uint8 indices
                                try:
                                    result.segmentation_map = self._validate_segmentation_map(result.segmentation_map)
                                except Exception as _v:
                                    if self.debug_mode:
                                        logger.warning("⚠️ Failed to validate segmentation_map: %s", _v)
                                    result.segmentation_map = np.clip(np.asarray(result.segmentation_map, dtype=np.int32), 0, 255).astype(np.uint8)
                            except Exception as resize_err:
                                if self.debug_mode:
                                    logger.warning("❌ Failed to resize segmentation outputs: %s", resize_err)

                        # After resizing/validation, update detected classes from the normalized segmentation map.
                        try:
                            class_labels = list(getattr(result, 'class_labels', None) or [])
                            if not class_labels and getattr(self, 'segmentor', None) is not None:
                                class_labels = self.segmentor.get_class_labels()
                            detected_classes = sorted(set(detected_classes) | set(self._derive_detected_classes(result.segmentation_map, class_labels)))
                        except Exception as cls_err:
                            if self.debug_mode:
                                logger.debug("Failed to derive detected classes from segmentation: %s", cls_err)

                        # Create segmentation visualization (optimized)
                        segmentation_overlay = self._create_segmentation_overlay_optimized(frame, result)
                        # Compute a small hash for the overlay to avoid re-encoding identical images
                        try:
                            overlay_hash = zlib.crc32(segmentation_overlay.tobytes())
                        except Exception:
                            overlay_hash = None

                        try:
                            if overlay_hash is None or overlay_hash != self._last_overlay_hash or self._last_overlay_b64 is None:
                                _, buffer = cv2.imencode('.jpg', segmentation_overlay, self.encode_params)
                                overlay_b64 = base64.b64encode(buffer).decode('utf-8')
                                self._last_overlay_b64 = f"data:image/jpeg;base64,{overlay_b64}"
                                self._last_overlay_counter = self.frame_counter
                                self._last_overlay_hash = overlay_hash
                            else:
                                # Reuse cached overlay
                                if self.debug_mode:
                                    logger.debug("♻️ Reusing cached overlay (frame %s)", self.frame_counter)
                        except Exception as enc_err:
                            if self.debug_mode:
                                logger.warning("❌ JPEG encode failed: %s", enc_err)
                            self._last_overlay_b64 = None
                            self._last_overlay_counter = -1

                        # Store result
                        segmentation_data = {
                            'frame_id': frame_id,
                            'timestamp': timestamp,
                            'frame_counter': self.frame_counter,
                            'segmentation_map': result.segmentation_map,
                            'overlay': segmentation_overlay,
                            'overlay_b64': self._last_overlay_b64,
                            'class_labels': result.class_labels,
                            'detected_classes': detected_classes,
                            'model_type': (result.metadata or {}).get('model_type'),
                            'metadata': result.metadata,
                        }

                        # Add to segmentation queue (remove old ones if full)
                        if self.segmentation_queue.full():
                            try:
                                self.segmentation_queue.get_nowait()
                            except Empty:
                                pass

                        self.segmentation_queue.put(segmentation_data)
                        self.current_segmentation = segmentation_data

                        # Immediately broadcast to connected WebSocket clients for smooth display
                        self._broadcast_segmentation_update()

                        # Generate music based on segmentation data
                        if self.music_enabled and self.musician is not None:
                            try:
                                music_frame = self.musician(
                                    result.segmentation_map,
                                    frame_id=self.frame_counter,
                                    class_labels=getattr(result, 'class_labels', None),
                                    metadata=getattr(result, 'metadata', None),
                                )

                                # Store music data
                                music_data = {
                                    'frame_id': frame_id,
                                    'timestamp': timestamp,
                                    'frame_counter': self.frame_counter,
                                    'music_frame': music_frame,
                                    'events_count': len(music_frame.events),
                                    'tempo': music_frame.tempo,
                                    'key_signature': music_frame.key_signature
                                }

                                # Add to music queue (remove old ones if full)
                                if self.music_queue.full():
                                    try:
                                        self.music_queue.get_nowait()
                                    except Empty:
                                        pass

                                self.music_queue.put(music_data)
                                self.current_music = music_data

                                # Broadcast music events to connected clients
                                self._broadcast_music_update(music_data)

                                if self.debug_mode and (time.time() - self.last_debug_time) > self.debug_interval:
                                    logger.debug("🎵 Generated %s music events for frame %s", len(music_frame.events), self.frame_counter)

                            except Exception as music_err:
                                logger.warning("❌ Error generating music: %s", music_err)

                        if self.debug_mode and (time.time() - self.last_debug_time) > self.debug_interval:
                            logger.debug("✅ Segmentation completed for frame %s", self.frame_counter)

                    except Exception as e:
                        logger.exception("❌ Error processing segmentation: %s", e)

                self.frame_counter += 1

            except Empty:
                # No frame available, continue loop
                continue
            except Exception as e:
                logger.exception("❌ Error in processing loop: %s", e)

    def _broadcast_segmentation_update(self):
        """Immediately broadcast segmentation update to connected WebSocket clients"""

        try:
            # Only broadcast to main UI for smooth display
            if self.main_ui_connected and self.socketio:
                display_data = self.get_synchronized_display(for_main_ui=True)
                state = self.get_current_state()
                response_data = {**display_data, 'queue_size': state['queue_size']}

                # Use socketio to broadcast to all connected clients
                self.socketio.emit('frame_update', response_data)

                if self.debug_mode and (time.time() - self.last_debug_time) > self.debug_interval:
                    logger.debug("📡 Broadcasted segmentation update for frame %s", self.frame_counter)
        except Exception as e:
            if self.debug_mode:
                logger.warning("❌ Error broadcasting update: %s", e)

    def _broadcast_music_update(self, music_data):
        """Broadcast music events to connected WebSocket clients"""
        try:
            if self.main_ui_connected and self.socketio:
                # Prepare music events data for transmission
                music_frame = music_data['music_frame']
                events_data = []

                for event in music_frame.events:
                    event_data = {
                        'note': event.note,
                        'velocity': event.velocity,
                        'duration': event.duration,
                        'channel': event.channel,
                        'timestamp': event.timestamp,
                        'class_name': event.metadata.get('class_name', 'unknown'),
                        'instrument': event.metadata.get('instrument', 'unknown'),
                        'presence_ratio': event.metadata.get('presence_ratio', 0.0)
                    }
                    events_data.append(event_data)

                music_response = {
                    'frame_id': music_data['frame_id'],
                    'frame_counter': music_data['frame_counter'],
                    'events': events_data,
                    'events_count': music_data['events_count'],
                    'tempo': music_data['tempo'],
                    'key_signature': music_data['key_signature'],
                    'timestamp': music_data['timestamp']
                }

                # Emit music events to connected clients
                self.socketio.emit('music_update', music_response)

                if self.debug_mode and (time.time() - self.last_debug_time) > self.debug_interval:
                    logger.debug("🎵 Broadcasted music update: %s events for frame %s",
                               len(events_data), music_data['frame_counter'])
        except Exception as e:
            if self.debug_mode:
                logger.warning("❌ Error broadcasting music update: %s", e)

    def _create_segmentation_overlay_optimized(self, frame, result):
        """Create an optimized visualization overlay for the segmentation result"""

        try:
            segmentation_map = getattr(result, 'segmentation_map', None)
            if segmentation_map is None:
                if self.debug_mode:
                    logger.debug("⚠️ No segmentation_map present in result; returning original frame")
                return frame

            # Validate and normalize segmentation map
            try:
                segmentation_map = self._validate_segmentation_map(segmentation_map)
            except Exception as _v:
                if self.debug_mode:
                    logger.warning("⚠️ segmentation_map validation failed in overlay: %s", _v)
                segmentation_map = np.clip(np.asarray(segmentation_map, dtype=np.int32), 0, 255).astype(np.uint8)

            # Occasional debug info (not every frame)
            if self.debug_mode and (time.time() - self.last_debug_time) > self.debug_interval:
                unique_classes = np.unique(segmentation_map)
                logger.debug("🔍 Classes: %s, Shape: %s", unique_classes, segmentation_map.shape)

                # Quick road detection check
                road_pixels = np.sum(segmentation_map == 0)
                road_percentage = (road_pixels / segmentation_map.size) * 100
                logger.debug("🛣️ Road: %.1f%% of image", road_percentage)

            # Vectorized color mapping using a lookup table derived from the model labels.
            class_labels = list(getattr(result, 'class_labels', None) or [])
            if not class_labels and getattr(self, 'segmentor', None) is not None:
                try:
                    class_labels = self.segmentor.get_class_labels()
                except Exception:
                    class_labels = []

            color_mapping_array = self._get_color_mapping_array(class_labels)
            overlay = color_mapping_array[segmentation_map]

            # Resize overlay to match original frame size if needed
            if overlay.shape[:2] != frame.shape[:2]:
                overlay = cv2.resize(overlay, (frame.shape[1], frame.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)  # Use nearest neighbor for segmentation

            # Optimized blending (reduced alpha for better performance)
            blended = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)

            # Convert back to BGR for encoding
            blended = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

            return blended

        except Exception as e:
            logger.exception("❌ Error creating segmentation overlay: %s", e)
            return frame

    def add_frame(self, frame, frame_id=None, timestamp=None):
        """Add a frame to the processing queue"""

        if timestamp is None:
            timestamp = time.time()

        if frame_id is None:
            frame_id = f"frame_{self.frame_counter}"

        frame_data = {
            'frame': frame,
            'frame_id': frame_id,
            'timestamp': timestamp
        }

        # Add to queue (remove old frame if full)
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                pass

        self.frame_queue.put(frame_data)

    def get_current_state(self):
        """Get current processing state for display"""

        return {
            'frame_counter': self.frame_counter,
            'current_frame_available': self.current_frame is not None,
            'current_segmentation_available': self.current_segmentation is not None,
            'current_music_available': self.current_music is not None if hasattr(self, 'current_music') else False,
            'music_enabled': self.music_enabled if hasattr(self, 'music_enabled') else False,
            'processing_interval': self.segmentation_interval,
            'queue_size': self.frame_queue.qsize(),
            'music_queue_size': self.music_queue.qsize() if hasattr(self, 'music_queue') else 0
        }

    def get_synchronized_display(self, for_main_ui=True):
        """Get synchronized frame and segmentation data for display"""

        display_data = {
            'original_frame': None,
            'segmentation_overlay': None,
            'segmentation_info': None,
            'music_info': None,
            'frame_counter': self.frame_counter,
            'timestamp': time.time()
        }

        # Only provide segmentation overlay to main UI to avoid conflicts
        if self.current_segmentation is not None and for_main_ui:
            seg_data = self.current_segmentation

            # Check if this segmentation is recent enough (within last 10 frames)
            frame_diff = self.frame_counter - seg_data['frame_counter']

            if frame_diff <= 10:  # Only send if recent
                # Use cached encoded overlay when available to avoid re-encoding
                if seg_data.get('overlay_b64'):
                    display_data['segmentation_overlay'] = seg_data['overlay_b64']
                else:
                    # Fallback to encoding if cache unavailable
                    _, buffer = cv2.imencode('.jpg', seg_data['overlay'], self.encode_params)
                    overlay_b64 = base64.b64encode(buffer).decode('utf-8')
                    display_data['segmentation_overlay'] = f"data:image/jpeg;base64,{overlay_b64}"

                # Minimal segmentation info
                display_data['segmentation_info'] = {
                    'frame_id': seg_data['frame_id'],
                    'timestamp': seg_data['timestamp'],
                    'frame_counter': seg_data['frame_counter'],
                    'frames_since_segmentation': frame_diff,
                    'class_labels': seg_data.get('detected_classes') or [],
                    'model_type': seg_data.get('model_type')
                }

        # For status page, provide basic info without heavy data
        elif not for_main_ui and self.current_segmentation is not None:
            seg_data = self.current_segmentation
            display_data['segmentation_info'] = {
                'frame_id': seg_data['frame_id'],
                'frame_counter': seg_data['frame_counter'],
                'frames_since_segmentation': self.frame_counter - seg_data['frame_counter'],
                'class_labels': seg_data.get('detected_classes') or [],
                'model_type': seg_data.get('model_type')
            }

        # Add music information if available
        if hasattr(self, 'current_music') and self.current_music is not None:
            music_data = self.current_music
            frame_diff = self.frame_counter - music_data['frame_counter']

            if frame_diff <= 10:  # Only include recent music data
                display_data['music_info'] = {
                    'frame_id': music_data['frame_id'],
                    'frame_counter': music_data['frame_counter'],
                    'events_count': music_data['events_count'],
                    'tempo': music_data['tempo'],
                    'key_signature': music_data['key_signature'],
                    'frames_since_music': frame_diff,
                    'timestamp': music_data['timestamp']
                }

        return display_data

    def toggle_music_generation(self, enable: bool = None):
        """Enable or disable music generation"""

        if hasattr(self, 'music_enabled'):
            if enable is None:
                self.music_enabled = not self.music_enabled
            else:
                self.music_enabled = enable

            status = "enabled" if self.music_enabled else "disabled"
            logger.info(f"🎵 Music generation {status}")
            return self.music_enabled
        return False

    def set_music_tempo(self, tempo: int):
        """Set music tempo (BPM)"""

        if hasattr(self, 'musician') and self.musician is not None:
            self.musician.tempo = tempo
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

    def get_music_status(self):
        """Get current music generation status"""

        if hasattr(self, 'musician') and self.musician is not None:
            return {
                'enabled': getattr(self, 'music_enabled', False),
                'tempo': self.musician.tempo,
                'key_signature': self.musician.key_signature,
                'musician_type': self.musician.musician_type,
                'queue_size': self.music_queue.qsize() if hasattr(self, 'music_queue') else 0
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
        if hasattr(self, 'musician') and self.musician is not None:
            current = self.musician.musician_type

        return {'musicians': musicians, 'current': current}

    def switch_musician(self, musician_type: str):
        """Switch to a different music generation model (keeps current tempo/key)"""

        if not hasattr(self, 'musician') or self.musician is None:
            return {'success': False, 'error': 'Musician system not initialized'}

        try:
            self.musician.switch_musician(musician_type)
            logger.info(f"🔄 Musician switched to: {self.musician.musician_type}")
            return {'success': True, 'musician_type': self.musician.musician_type}
        except Exception as e:
            logger.error(f"❌ Error switching musician: {e}")
            return {'success': False, 'error': str(e)}

    def shutdown(self):
        """Shutdown the processor"""

        logger.info("🛑 Shutting down Main processor...")
        self.frame_queue.put(None)  # Shutdown signal
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)

    def enable_debug_mode(self, enable=True):
        """Enable or disable debug mode for verbose logging"""

        self.debug_mode = enable
        if enable:
            set_level(logger, "DEBUG")
            logger.info("🐛 Debug mode enabled - verbose logging activated")
        else:
            set_level(logger, "INFO")
            logger.info("🔇 Debug mode disabled - minimal logging activated")

    def set_main_ui_connected(self, connected=True):
        """Mark main UI as connected/disconnected to prioritize it over status page"""

        if self.main_ui_connected != connected:
            self.main_ui_connected = connected
            
            if connected:
                logger.info("🎯 Main UI connected - prioritizing segmentation data for main interface")
            else:
                logger.info("📄 Main UI disconnected - status page can receive data")
        else:
            self.main_ui_connected = connected

# Initialize Flask app and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'video_processing_secret'
CORS(app)  # Enable CORS for all routes


class ClientDisconnectSafeMiddleware:
    """Gracefully ignore client disconnects that happen during streaming or polling."""

    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        try:
            return self.app(environ, start_response)
        except Exception as exc:
            if isinstance(exc, (BrokenPipeError, ConnectionResetError, OSError, AssertionError, RuntimeError)):
                logger.debug("Client disconnected while serving a response: %s", exc)
                return []
            raise


app.wsgi_app = ClientDisconnectSafeMiddleware(app.wsgi_app)

# Reduce Socket.IO/engineio log noise in production
socketio = SocketIO(app, cors_allowed_origins="*", logger=False, engineio_logger=False)

# Paths for serving the existing web UI (so mobile devices can load it from the laptop)
PLATFORM_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(PLATFORM_DIR, '..', '..'))
ASSETS_DIR = os.path.join(PROJECT_ROOT, 'assets')

# Additional CORS headers for all routes
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Global processor instance - pass socketio for real-time broadcasting
processor = Processor(socketio_instance=socketio)

@app.route('/')
def index():
    """Redirect the root URL to the main UI so the processor server is usable directly."""
    return redirect('/ui/', code=302)

@app.route('/ui')
def ui_redirect():
    """Redirect /ui to /ui/ so static assets resolve correctly."""
    return redirect('/ui/', code=302)

@app.route('/ui/')
def ui_index():
    """Serve the main Platform UI entrypoint (UI.html).

    Keeping UI.html as-is means all existing responsive behavior and JS logic stays identical;
    relative links (styles.css/script.js) resolve under /ui/ automatically.
    """
    return send_from_directory(PLATFORM_DIR, 'UI.html')

@app.route('/ui/<path:filename>')
def ui_static(filename: str):
    """Serve Platform UI static files (script.js, styles.css, etc.)."""

    return send_from_directory(PLATFORM_DIR, filename)

@app.route('/assets/<path:filename>')
def serve_assets(filename: str):
    """Serve shared project assets (icons, etc.) referenced by UI.html."""

    return send_from_directory(ASSETS_DIR, filename)

@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    """Receive frame data from UI and add to processing queue"""

    try:
        data = request.get_json()

        if 'frame' not in data:
            return jsonify({'error': 'No frame data provided'}), 400

        # Decode base64 frame
        frame_data = data['frame']
        if frame_data.startswith('data:image'):
            # Remove data URL prefix
            frame_data = frame_data.split(',')[1]

        # Decode image
        img_buffer = base64.b64decode(frame_data)
        img_array = np.frombuffer(img_buffer, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Invalid frame data'}), 400

        # Convert BGR to RGB for proper processing
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Add frame to processor
        frame_id = data.get('frame_id', f"frame_{int(time.time() * 1000)}")
        timestamp = data.get('timestamp', time.time())

        processor.add_frame(frame, frame_id, timestamp)

        # Get current state
        state = processor.get_current_state()

        return jsonify({
            'success': True,
            'frame_counter': state['frame_counter'],
            'queue_size': state['queue_size'],
            'message': 'Frame processed successfully'
        })

    except Exception as e:
        logger.exception("❌ Error processing frame: %s", e)
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_display', methods=['GET'])
def get_display():
    """Get synchronized display data - prioritized for main UI"""

    try:
        # Mark main UI as connected when it requests data
        processor.set_main_ui_connected(True)
        display_data = processor.get_synchronized_display(for_main_ui=True)
        return jsonify(display_data)
    except Exception as e:
        logger.exception("❌ Error getting display data: %s", e)
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get processor status"""

    try:
        state = processor.get_current_state()
        return jsonify(state)
    except Exception as e:
        logger.exception("❌ Error getting status: %s", e)
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug/<action>', methods=['POST'])
def toggle_debug(action):
    """Toggle debug mode for performance monitoring"""

    try:
        if action == 'enable':
            processor.enable_debug_mode(True)
            return jsonify({'success': True, 'debug_mode': True})
        elif action == 'disable':
            processor.enable_debug_mode(False)
            return jsonify({'success': True, 'debug_mode': False})
        else:
            return jsonify({'error': 'Invalid action. Use enable or disable'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@socketio.on('request_update')
def handle_update_request():
    """Handle real-time update requests via WebSocket - PRIORITIZED FOR MAIN UI"""

    try:
        # Check if this is from main UI or status page
        is_main_ui = request.sid not in processor.status_page_clients

        if is_main_ui:
            # Mark main UI as connected and get full data
            processor.set_main_ui_connected(True)
            display_data = processor.get_synchronized_display(for_main_ui=True)
        else:
            # Status page gets limited data to avoid conflicts
            display_data = processor.get_synchronized_display(for_main_ui=False)

        state = processor.get_current_state()

        # Combine display data with state
        response_data = {**display_data, 'queue_size': state['queue_size']}

        # Always emit, even if no new segmentation data - client decides what to display
        try:
            emit('frame_update', response_data)
        except Exception as emit_err:
            if isinstance(emit_err, (BrokenPipeError, ConnectionResetError, OSError, RuntimeError)):
                logger.debug("Client disconnected while emitting frame update: %s", emit_err)
            else:
                logger.exception("❌ Error emitting frame update: %s", emit_err)

        # Debug logging (only when enabled)
        if processor.debug_mode:
            has_overlay = 'segmentation_overlay' in response_data and response_data['segmentation_overlay'] is not None
            client_type = "Main UI" if is_main_ui else "Status Page"
            logger.debug("📡 Update sent to %s - Frame: %s, Has overlay: %s, Queue: %s",
                         client_type, response_data.get('frame_counter', 0), has_overlay, response_data.get('queue_size', 0))

    except Exception as e:
        logger.exception("❌ Error handling update request: %s", e)
        emit('error', {'message': str(e)})

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""

    # Determine if this is status page or main UI based on referrer
    referrer = (request.headers.get('Referer', '') or '').lower()

    # If the client came from /ui/, treat as Main UI; otherwise, treat as status page.
    # When Referer is missing (e.g., some WebViews), default to Main UI.
    is_main_ui = (not referrer) or ('/ui/' in referrer) or (referrer.endswith('/ui'))

    if is_main_ui:
        logger.info("🎯 Main UI connected: %s", request.sid)
        return

    processor.status_page_clients.add(request.sid)
    logger.info("📄 Status page connected: %s", request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""

    if request.sid in processor.status_page_clients:
        processor.status_page_clients.remove(request.sid)
        logger.info("📄 Status page disconnected: %s", request.sid)
    else:
        # Check if any main UI clients are still connected
        # If not, mark main UI as disconnected
        logger.info("🎯 Main UI disconnected: %s", request.sid)
        # In a simple case, assume main UI is disconnected
        processor.set_main_ui_connected(False)

@socketio.on('toggle_music')
def handle_toggle_music(data):
    """Handle music generation toggle from client"""

    try:
        enabled = data.get('enabled', True)
        result = processor.toggle_music_generation(enabled)
        emit('music_status', {'enabled': result, 'success': True})
        logger.info("🎵 Music generation toggled: %s", enabled)
    except Exception as e:
        emit('music_status', {'error': str(e), 'success': False})
        logger.error("❌ Error toggling music: %s", e)

@socketio.on('set_music_tempo')
def handle_set_music_tempo(data):
    """Handle music tempo change from client"""

    try:
        tempo = data.get('tempo', 120)
        result = processor.set_music_tempo(tempo)
        emit('music_status', {'tempo': tempo, 'success': result})
        logger.info("🎵 Music tempo set to: %s BPM", tempo)
    except Exception as e:
        emit('music_status', {'error': str(e), 'success': False})
        logger.error("❌ Error setting music tempo: %s", e)

@socketio.on('set_music_key')
def handle_set_music_key(data):
    """Handle music key change from client"""

    try:
        key_signature = data.get('key_signature', 'C_major')
        result = processor.set_music_key(key_signature)
        emit('music_status', {'key_signature': key_signature, 'success': result})
        logger.info("🎵 Music key set to: %s", key_signature)
    except Exception as e:
        emit('music_status', {'error': str(e), 'success': False})
        logger.error("❌ Error setting music key: %s", e)

@socketio.on('get_music_status')
def handle_get_music_status():
    """Get current music generation status"""

    try:
        status = processor.get_music_status()
        emit('music_status', status)
    except Exception as e:
        emit('music_status', {'error': str(e), 'success': False})
        logger.error("❌ Error getting music status: %s", e)

@socketio.on('get_available_musicians')
def handle_get_available_musicians():
    """Send the list of available musicians (for the "Change Musician" picker) to the client"""

    try:
        data = processor.get_available_musicians()
        emit('musicians_list', data)
    except Exception as e:
        emit('musicians_list', {'error': str(e), 'musicians': [], 'current': None})
        logger.error("❌ Error getting available musicians: %s", e)

@socketio.on('switch_musician')
def handle_switch_musician(data):
    """Handle musician switch request from client"""

    try:
        musician_type = (data or {}).get('musician_type')
        if not musician_type:
            emit('musician_switched', {'success': False, 'error': 'musician_type is required'})
            return

        result = processor.switch_musician(musician_type)
        emit('musician_switched', result)
        if result.get('success'):
            logger.info("🎭 Musician switched to: %s", result.get('musician_type'))
    except Exception as e:
        emit('musician_switched', {'success': False, 'error': str(e)})
        logger.error("❌ Error switching musician: %s", e)

def run_processor_server(host='0.0.0.0', port=5000, debug=False):
    """Run the processor server"""

    logger.info("🚀 Starting Video Processor Server on %s:%s", host, port)
    logger.info("📊 Processing every %s frames for optimal performance", processor.segmentation_interval)
    logger.info("🌐 Web interface available at:")
    logger.info("   - Status: http://%s:%s/", host, port)
    logger.info("   - UI:     http://%s:%s/ui/", host, port)
    logger.info("📡 API endpoints:")
    logger.info("   - POST /api/process_frame - Send frame data")
    logger.info("   - GET  /api/get_display  - Get synchronized display")
    logger.info("   - GET  /api/status       - Get processor status")
    logger.info("   - POST /api/debug/enable - Enable verbose debug logging")
    logger.info("   - POST /api/debug/disable - Disable debug logging for performance")
    logger.info("🚀 Performance Mode: Debug logging %s", "ON" if processor.debug_mode else "OFF")
    logger.info("⚡ Optimizations: Reduced queues, vectorized color mapping, throttled updates")

    try:
        socketio.run(app, host=host, port=port, debug=debug)
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutting down server...")
        processor.shutdown()
    except Exception as e:
        logger.exception("❌ Server error: %s", e)
        processor.shutdown()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Main Processing Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (use 0.0.0.0 for LAN/mobile access)')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--interval', type=int, default=5, help='Segmentation processing interval (frames)')

    args = parser.parse_args()

    # Update processing interval if specified
    if args.interval != 5:
        processor.segmentation_interval = args.interval
        logger.info("🔄 Updated segmentation interval to %s frames", args.interval)

    # Set debug mode based on argument
    if args.debug:
        processor.enable_debug_mode(True)
        logger.info("🐛 Debug mode enabled via command line")

    run_processor_server(args.host, args.port, args.debug)
