"""
Modular Processing Framework for Receiving Data and Performing Segmentation and Generating Music
=================================================

This module receives data from UI.html and processes it using the Segmentor and Detector classes.
Then sends the processed data back to UI.html for Generating Music.
"""

import numpy as np
import cv2
import base64
import time
import threading
import argparse
import hashlib
import colorsys
import zlib
import traceback
import os
import sys
from typing import List, Tuple, Dict, Any
from queue import Queue, Empty
from flask import Flask, request, jsonify, send_from_directory, redirect
from flask_socketio import SocketIO, emit
from flask_cors import CORS

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Segmentation.Segmentor import Segmentor, SegmentationResult
from Music_Generator.Musician import Musician
from utils.logging_setup import setup_logging, set_level

logger = setup_logging("INFO", name="Platform.Processor")


class ROI:
    """
    ROI defined by 4 corner points + 4 bezier control points
    """

    def __init__(
        self,
        corners: List[Tuple[float, float]],
        controls: List[Tuple[float, float]],
        frame_size: Tuple[int, int] = (1280, 720),
    ):
        """
        Args:
            corners: List of 4 corner points (x, y)
            controls: List of 4 bezier control points (x, y)
            frame_size: (width, height) of the frame these masks must align with.
                Must match the actual segmentation_map / mask resolution used at
                collision-check time, or intersects_mask() will silently misfire
                (wrong shape -> wrong/empty results).
        """

        if len(corners) != 4 or len(controls) != 4:
            raise ValueError("ROI must have exactly 4 corners and 4 control points")

        self.corners = corners
        self.controls = controls
        self.frame_width, self.frame_height = frame_size

        self.polygon = self._build_polygon()
        self.edges = self._build_edges()

        self.boundary_mask = self._build_boundary_mask(width=self.frame_width, height=self.frame_height)
        self.edge_masks = self._build_edge_masks(width=self.frame_width, height=self.frame_height)

    def _build_boundary_mask(self, width, height, thickness=3):

        mask = np.zeros((height, width), dtype=np.uint8)

        pts = np.array(self.polygon, dtype=np.int32)

        cv2.polylines(
            mask,
            [pts],
            isClosed=True,
            color=255,
            thickness=thickness
        )

        return mask.astype(bool)

    def _build_edge_masks(self, width, height, thickness=3):

        edge_masks = []

        samples_per_edge = len(self.polygon) // 4

        for i in range(4):

            mask = np.zeros((height, width), dtype=np.uint8)

            start = i * samples_per_edge
            end = (i + 1) * samples_per_edge

            pts = np.array(
                self.polygon[start:end],
                dtype=np.int32
            )

            cv2.polylines(
                mask,
                [pts],
                isClosed=False,
                color=255,
                thickness=thickness
            )

            edge_masks.append(mask.astype(bool))

        return edge_masks

    def _quad_bezier(self, p0, p1, p2, t):

        return (
            (1 - t) ** 2 * np.array(p0)
            + 2 * (1 - t) * t * np.array(p1)
            + t ** 2 * np.array(p2)
        )

    def _build_polygon(self):

        poly = []

        n = len(self.corners)

        for i in range(n):

            p0 = self.corners[i]
            p2 = self.corners[(i + 1) % n]
            p1 = self.controls[i]

            for t in np.linspace(0, 1, 20):
                pt = self._quad_bezier(p0, p1, p2, t)
                poly.append((pt[0], pt[1]))

        return poly

    def _build_edges(self):

        edges = []

        for i in range(len(self.polygon)):

            a = self.polygon[i]
            b = self.polygon[(i + 1) % len(self.polygon)]

            edges.append((a, b))

        return edges

    def calculate_intersection_area(self, mask):

        intersection = np.logical_and(mask, self.boundary_mask)
        area = np.sum(intersection)

        return area

    def intersects_mask(self, mask, return_edges=False):

        touching = np.logical_and(mask, self.boundary_mask).any()

        if not return_edges:
            return touching

        edge_names = ["top", "right", "bottom", "left"]
        edges = []

        for name, edge_mask in zip(edge_names, self.edge_masks):
            if np.logical_and(mask, edge_mask).any():
                edges.append(name)

        erea = self.calculate_intersection_area(mask)

        return {
            "touching": touching,
            "edges": edges,
            "area": erea
        }

class Detector:
    """
    Scene Event Detector that tracks objects and detects events based on ROI interactions.
    """

    def __init__(self):

        self.roi = None  # Will be set per frame if provided
        self.prev_roi_payload = None  # Tracks ROI coordinates and frame dimensions

        # state: keeps track of objects
        self.state = {
            "objects": {},          # object_id -> object info
            "next_object_id": 0
        }
        self.frame_counter = 0

    def __call__(self,
            input: SegmentationResult,
            frame_id: int = 0,
            roi: Dict[str, Any] = None
        ):

        if not isinstance(input, SegmentationResult):
            raise ValueError("Input must be a SegmentationResult instance")
        
        frame_height, frame_width = input.segmentation_map.shape[:2]
        self._set_roi(roi, frame_size=(frame_width, frame_height))
        self.frame_counter = frame_id

        detected = self.detect_scene_events(input.bounding_boxes, input.masks)

        return detected

    def _set_roi(self, roi_payload, frame_size):
        
        if not roi_payload:
            self.roi = None
            self.prev_roi_payload = None
            return

        roi_state = (roi_payload, frame_size)
        if self.prev_roi_payload != roi_state:
            self.prev_roi_payload = roi_state
            self.roi = ROI(
                corners=roi_payload.get("corners", []),
                controls=roi_payload.get("controls", []),
                frame_size=frame_size,
            )
            
            logger.info(f"💢 ROI updated for frame {self.frame_counter}.")

    def assign_object_ids(self, objects, max_distance=100):
        """
        Assign unique IDs to detected objects based on their bounding boxes and class names. 
        The rule is to match objects across frames based on IoU proximity and class similarity, 
        while also considering the maximum allowed distance for matching.
        """

        updated_objects = {}
        used_tracks = set()

        for obj in objects:

            bbox = obj["bbox"]
            cls = obj["class_name"]

            x1, y1, x2, y2 = bbox
            if "centroid" in obj.keys():
                centroid = obj["centroid"]
            else:
                centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            matched_id = None
            best_score = float("-inf")

            # Search previous objects
            for object_id, previous in self.state["objects"].items():

                penalty = 0

                # Class name mismatch penalty
                if previous["class_name"] != cls:
                    penalty += -10

                # Already used in this frame penalty
                if object_id in used_tracks: 
                    penalty += -1000

                # Distance penalty
                pcx, pcy = previous["centroid"]
                cx, cy = centroid
                distance = ((cx-pcx)**2 + (cy-pcy)**2)**0.5
                if distance > max_distance:
                    penalty += -((distance / max_distance) * 100)

                IoU = None
                px1, py1, px2, py2 = previous["bbox"]
                ix1, iy1, ix2, iy2 = max(x1, px1), max(y1, py1), min(x2, px2), min(y2, py2)
                if ix1 >= ix2 or iy1 >= iy2:
                    IoU = 0.0
                else:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    area = (x2 - x1) * (y2 - y1)
                    areap = (px2 - px1) * (py2 - py1)
                    union = area + areap - inter
                    IoU = inter / union if union > 0 else 0.0

                score = IoU * 1000 + penalty
                if score > best_score:
                    best_score = score
                    matched_id = object_id

            # Existing object
            if best_score > 0:
                obj_id = matched_id
                used_tracks.add(obj_id)
                previous = self.state["objects"][obj_id]
                is_touching = previous["touching"]

            # New object
            else:
                obj_id = self.state["next_object_id"]
                if self.state["next_object_id"] > 5000:
                    self.state["next_object_id"] = 0
                    logger.warning("ID counter exceeded 5000, resetting to 0. This may cause ID collisions.")
                self.state["next_object_id"] += 1
                is_touching = False

            updated_objects[obj_id] = {
                "class_name": cls,
                "centroid": centroid,
                "bbox": bbox,
                "touching": is_touching,
                "missing_frames": 0,
                "last_seen_frame": self.frame_counter
            }

            obj["object_id"] = obj_id

        for object_id, previous in self.state["objects"].items():

            if object_id in updated_objects:
                continue

            previous["missing_frames"] += 1
            updated_objects[object_id] = previous

        # Replace old objects
        self.state["objects"] = updated_objects

    def detect_scene_events(self, bounding_boxes=None, masks=None):
        """
        Detect scene events and return a list of events.
        Here, event is defined as an object touching or releasing the ROI boundary.
        """

        events = []

        if not bounding_boxes or not masks:
            self.assign_object_ids(bounding_boxes or [])
            return events

        if self.roi is None:
            logger.warning("No ROI provided for scene event detection.")
            self.assign_object_ids(bounding_boxes)
            return events
        
        self.assign_object_ids(bounding_boxes)

        for obj in bounding_boxes:

            obj_id = obj["object_id"]
            obj_class = obj["class_name"]
            obj_mask = masks.get(obj_class, None)

            if obj_mask is None:
                logger.warning(f"No mask found for object class '{obj_class}'. Skipping event detection.")
                continue

            collision = self.roi.intersects_mask(
                mask=obj_mask,
                return_edges=True
            )
            touching = collision["touching"]
            edges = collision["edges"]
            erea = collision["area"]

            track = self.state["objects"].get(obj_id, {})
            prev = track.get("touching", False)

            event_type = None
            if touching and not prev:
                event_type = "ROI_TOUCH"
                self.state["objects"][obj_id]["touching"] = True
            elif not touching and prev:
                event_type = "ROI_RELEASE"
                self.state["objects"][obj_id]["touching"] = False

            events.append({
                "type": event_type,
                "object_id": obj_id,
                "class": obj_class,
                "edges": edges,
                "area": erea
            })

        logger.info(f"Detected {len(events)} scene events")
        return events

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
        self.current_detection = None
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
            model_type = os.environ.get('RAGBAARNET_SEGMENTATION_MODEL', 'yolo').strip().lower()
            model_path = os.environ.get('RAGBAARNET_SEGMENTATION_MODEL_PATH', '').strip()

            if model_type == 'yolo':
                if not model_path:
                    model_path = os.path.join(
                        os.path.dirname(__file__),
                        '..',
                        'Segmentation',
                        'Pre-trained Models',
                        'yolo26',
                        'yolo26l-seg.pt',
                    )
                self.segmentor = Segmentor('yolo', model_path=model_path)
                logger.info("✅ YOLO Segmentor initialized successfully")
            elif model_type == 'segformer':
                if not model_path:
                    model_path = os.environ.get(
                        'RAGBAARNET_SEGFORMER_PATH',
                        os.path.abspath(
                            os.path.join(
                                os.path.dirname(__file__),
                                '..',
                                'Segmentation',
                                'Pre-trained Models',
                                'segformer-b4-finetuned-cityscapes-1024-1024',
                            )
                        )
                    )
                self.segmentor = Segmentor('segformer', model_path=model_path)
                logger.info("✅ SegFormer Segmentor initialized successfully")
                
        except Exception as e:
            logger.exception("❌ Error initializing segmentor: %s", e)
            self.segmentor = None

        # Initialize detector
        logger.info("🔄 Initializing scene event detector...")
        try:
            self.detector = Detector()
            logger.info("✅ Scene Event Detector initialized successfully")
        except Exception as e:
            logger.exception("❌ Error initializing scene event detector: %s", e)
            self.detector = None

        # Initialize music generation
        logger.info("🔄 Initializing music generation...")
        try:
            self.musician = Musician('lstm-onessen', tempo=120, key_signature="C_major")
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
        """
        Create a deterministic color map for segmentation labels.

        Args:
            class_labels (List[str]): List of class names.

        Returns:
            Dict[int, List[int]]: class_id -> RGB color
        """

        labels = []

        for label in (class_labels or []):
            label = (
                str(label)
                .strip()
                .lower()
                .replace("_", " ")
                .replace("-", " ")
            )
            labels.append(label)

        # Standard palette (Cityscapes + useful COCO road objects)

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
            """
            Deterministically generate a pleasant RGB color from a label.
            """

            digest = hashlib.md5(label.encode("utf-8")).digest()

            hue = digest[0] / 255.0

            saturation = 0.65 + (digest[1] / 255.0) * 0.30
            value = 0.75 + (digest[2] / 255.0) * 0.20

            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)

            return [
                int(r * 255),
                int(g * 255),
                int(b * 255)
            ]

        # Build color map
        color_map = {}

        for class_id, label in enumerate(labels):

            if label in palette:
                color_map[class_id] = palette[label]
            else:
                color_map[class_id] = hashed_color(label)

        # Optional ignore label (Cityscapes convention)
        color_map[255] = [0, 0, 0]

        if self.debug_mode and labels:
            logger.debug(
                "🎨 Generated deterministic color map for %d classes.",
                len(labels)
            )

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
                roi_points = frame_data['roi_points']
                roi_controls = frame_data['roi_controls']

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

                                scale_x = orig_w / float(seg_frame.shape[1])
                                scale_y = orig_h / float(seg_frame.shape[0])
                                for detected_object in result.bounding_boxes:
                                    x1, y1, x2, y2 = detected_object['bbox']
                                    detected_object['bbox'] = [
                                        x1 * scale_x,
                                        y1 * scale_y,
                                        x2 * scale_x,
                                        y2 * scale_y,
                                    ]
                                    if 'centroid' in detected_object:
                                        cx, cy = detected_object['centroid']
                                        detected_object['centroid'] = (cx * scale_x, cy * scale_y)

                                if isinstance(result.masks, dict):
                                    result.masks = {
                                        key: cv2.resize(
                                            np.asarray(mask, dtype=np.uint8),
                                            (orig_w, orig_h),
                                            interpolation=cv2.INTER_NEAREST,
                                        ).astype(bool)
                                        for key, mask in result.masks.items()
                                    }

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

                        # After resizing/validation, derive a small, UI-friendly list of detected class names from the segmentation output.
                        detected_classes = []
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

                        #  Detect scene events based on ROI and detected objects
                        scene_events = []
                        if self.detector is not None:
                            try:
                                scene_events = self.detector(
                                    input=result,
                                    frame_id=self.frame_counter,
                                    roi={
                                        'corners': roi_points,
                                        'controls': roi_controls
                                    }
                                )
                                
                                if self.debug_mode and (time.time() - self.last_debug_time) > self.debug_interval:
                                    logger.debug("🎯 Detected %s scene events for frame %s", len(scene_events), self.frame_counter)
                            except Exception as event_err:
                                logger.error("❌ Error detecting scene events: %s", event_err)
                                logger.error("Traceback:\n%s", traceback.format_exc())

                        detections = []
                        if self.detector is not None:
                            for object_id, tracked_object in self.detector.state["objects"].items():
                                if (
                                    tracked_object.get("last_seen_frame") == self.frame_counter
                                    and tracked_object.get("touching", False)
                                ):
                                    detections.append({
                                        "object_id": int(object_id),
                                        "class_name": tracked_object["class_name"],
                                        "bbox": [float(value) for value in tracked_object["bbox"]],
                                    })

                        self.current_detection = {
                            "frame_id": frame_id,
                            "timestamp": timestamp,
                            "frame_counter": self.frame_counter,
                            "frame_width": orig_w,
                            "frame_height": orig_h,
                            "detections": detections,
                        }

                        # Comment out this call whenever collision boxes are not needed in the UI.
                        self._broadcast_detection_update()

                        # Generate music based on segmentation data
                        if self.music_enabled and self.musician is not None:
                            try:
                                music_frame = self.musician(
                                    results=scene_events,
                                    frame_id=self.frame_counter,
                                    state=self.detector.state
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
                                logger.error("❌ Error generating music: %s", music_err)
                                logger.error("Traceback:\n%s", traceback.format_exc())

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

    def _broadcast_detection_update(self):
        """Broadcast boxes for currently tracked objects intersecting the ROI."""
        try:
            if self.main_ui_connected and self.socketio and self.current_detection is not None:
                self.socketio.emit('detection_update', self.current_detection)

                if self.debug_mode and (time.time() - self.last_debug_time) > self.debug_interval:
                    logger.debug(
                        "📦 Broadcasted %s intersecting detections for frame %s",
                        len(self.current_detection['detections']),
                        self.current_detection['frame_counter'],
                    )
        except Exception as e:
            if self.debug_mode:
                logger.warning("❌ Error broadcasting detection update: %s", e)

    def _broadcast_music_update(self, music_data):
        """Broadcast music events to connected WebSocket clients"""
        try:
            if self.main_ui_connected and self.socketio:
                # Prepare music events data for transmission
                music_frame = music_data['music_frame']
                events_data = []

                for event in music_frame.events:
                    instrument_name = event.instrument
                    if instrument_name in ('unknown', None, ''):
                        logger.error("❌ Event has unknown instrument: %s", event)

                    event_data = {
                        'event_type': event.event_type,
                        'note': event.note,
                        'channel': event.channel,
                        'velocity': event.velocity,
                        'instrument': instrument_name,
                        'timestamp': event.timestamp,
                        # 'class_name': event.metadata.get('class_name', 'unknown'),
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

    def add_frame(self, frame, frame_id=None, timestamp=None, roi_points=None, roi_controls=None):
        """Add a frame to the processing queue"""

        if timestamp is None:
            timestamp = time.time()

        if frame_id is None:
            frame_id = f"frame_{self.frame_counter}"

        frame_data = {
            'frame': frame,
            'frame_id': frame_id,
            'timestamp': timestamp,
            'roi_points': roi_points,
            'roi_controls': roi_controls
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

    def get_music_status(self):
        """Get current music generation status"""

        if hasattr(self, 'musician') and self.musician is not None:
            return {
                'enabled': getattr(self, 'music_enabled', False),
                'tempo': self.musician.tempo,
                'key_signature': self.musician.key_signature,
                'musician_type': self.musician.musician_type,
                'instrument': self.musician.instrument,
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
            if not 60 <= tempo <= 180:
                raise ValueError('Tempo must be between 60 and 180 BPM')

            if musician_type != self.musician.musician_type:
                self.musician.switch_musician(
                    musician_type,
                    tempo=tempo,
                    instrument=instrument
                )
            else:
                self.musician.set_tempo(tempo)
                if musician_type == 'lstm-onessen':
                    self.musician.set_instrument(instrument)

            return {
                'success': True,
                'musician_type': self.musician.musician_type,
                'tempo': self.musician.tempo,
                'instrument': self.musician.instrument
            }
        except Exception as e:
            logger.error(f"❌ Error applying music settings: {e}")
            return {'success': False, 'error': str(e)}

    def switch_musician(self, musician_type: str):
        """Switch to a different music generation model (keeps current tempo/key)"""

        if not hasattr(self, 'musician') or self.musician is None:
            return {'success': False, 'error': 'Musician system not initialized'}

        try:
            self.musician.switch_musician(musician_type)
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
                logger.info("📄 Main UI disconnected")
        else:
            self.main_ui_connected = connected

# Initialize Flask app and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'video_processing_secret'
CORS(app)  # Enable CORS for all routes

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

        roi_points = data.get("roi_points", [])
        roi_controls = data.get("roi_controls", [])

        processor.add_frame(
            frame,
            frame_id,
            timestamp,
            roi_points=roi_points,
            roi_controls=roi_controls
        )

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
        processor.set_main_ui_connected(True)
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
    """Send the available musicians and current music settings to the client."""

    try:
        data = processor.get_available_musicians()
        emit('musicians_list', data)
    except Exception as e:
        emit('musicians_list', {'error': str(e), 'musicians': [], 'current': None})
        logger.error("❌ Error getting available musicians: %s", e)

@socketio.on('set_music_settings')
def handle_set_music_settings(data):
    """Apply the combined music settings from the platform UI."""

    try:
        settings = data or {}
        musician_type = settings.get('musician_type')
        if not musician_type:
            emit('music_settings_updated', {'success': False, 'error': 'musician_type is required'})
            return

        result = processor.apply_music_settings(
            musician_type=musician_type,
            tempo=settings.get('tempo', 120),
            instrument=settings.get('instrument', 'piano')
        )
        emit('music_settings_updated', result)
        if result.get('success'):
            logger.info(
                "🎵 Music settings updated: musician=%s, instrument=%s, tempo=%s",
                result.get('musician_type'), result.get('instrument'), result.get('tempo')
            )
    except Exception as e:
        emit('music_settings_updated', {'success': False, 'error': str(e)})
        logger.error("❌ Error applying music settings: %s", e)

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
    parser.add_argument('--interval', type=int, default=2, help='Segmentation processing interval (frames)')

    args = parser.parse_args()

    # Update processing interval if specified
    if args.interval != 2:
        processor.segmentation_interval = args.interval
        logger.info("🔄 Updated segmentation interval to %s frames", args.interval)

    # Set debug mode based on argument
    if args.debug:
        processor.enable_debug_mode(True)
        logger.info("🐛 Debug mode enabled via command line")

    run_processor_server(args.host, args.port, args.debug)
