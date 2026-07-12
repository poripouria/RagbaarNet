"""
Modular Music Generation Framework for Real-Time Visual-to-Audio Mapping
========================================================================

This module provides an extensible framework for generating music based on visual data,
particularly segmentation maps from computer vision models. It supports various music
generation strategies with easy integration for additional models.
"""

import os
import sys
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.logging_setup import setup_logging
from Segmentation.Segmentor import SegmentationResult

logger = setup_logging("INFO", name="Music_Generator.Musician")


@dataclass
class MusicEvent:
    """
    Core atomic event in the music system.

    This represents a single musical action (NOT only MIDI note).
    Designed to be extendable for future MIDI CC, pitch bend, etc.

    Attributes:
        event_type: Type of the event (e.g., "note_on", "note_off", "control_change")
        note: MIDI note number (0-127), optional depending on event_type
        channel: MIDI channel (0-15)
        velocity: Note velocity (0-127), optional depending on event_type
        instrument: Name of the instrument (e.g., "piano", "violin"), optional
        timestamp: Time at which the event occurs
        metadata: Additional event-specific information
    """

    event_type: str     # e.g. "note_on", "note_off"
    note: Optional[int] = None
    channel: int = 0
    velocity: Optional[int] = None
    instrument: Optional[str] = None
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MusicFrame:
    """
    Data class to store music generation results for a frame.
    Represents generated musical content at a single timestep.

    Attributes:
        events: List of music events for this frame
        frame_id: Identifier for the corresponding video frame
        timestamp: Generation timestamp
        tempo: Current tempo (BPM)
        key_signature: Current key signature
        metadata: Additional frame-specific information
    """

    events: List[MusicEvent]
    frame_id: int = 0
    timestamp: float = 0.0
    tempo: int = 120
    key_signature: str = "C_major"
    metadata: Dict[str, Any] = field(default_factory=dict)

class ROI:
    """
    ROI defined by 4 corner points + 4 bezier control points
    """

    def __init__(self, corners: List[Tuple[float, float]], controls: List[Tuple[float, float]]):
        """
        Args:
            corners: List of 4 corner points (x, y)
            controls: List of 4 bezier control points (x, y)
        """

        if len(corners) != 4 or len(controls) != 4:
            raise ValueError("ROI must have exactly 4 corners and 4 control points")
        
        self.corners = corners
        self.controls = controls

        self.polygon = self._build_polygon()
        self.edges = self._build_edges()

    def _quad_bezier(self, p0, p1, p2, t):

        return (
            (1 - t)**2 * np.array(p0)
            + 2 * (1 - t) * t * np.array(p1)
            + t**2 * np.array(p2)
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

class ROIGrid:
    """
    Converts ROI polygon into spatial occupancy grid for fast lookup
    """

    def __init__(self, polygon, grid_size=(64, 64), width=1280, height=720):
        self.polygon = polygon
        self.grid_size = grid_size
        self.width = width
        self.height = height

        self.grid = self._build_grid()

    # point in polygon (cheap)
    def _point_in_poly(self, x, y):
        poly = self.polygon
        inside = False

        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]
            xj, yj = poly[j]

            intersect = ((yi > y) != (yj > y)) and \
                        (x < (xj - xi) * (y - yi) / (yj - yi + 1e-6) + xi)

            if intersect:
                inside = not inside

            j = i

        return inside

    # build grid mask once (IMPORTANT)
    def _build_grid(self):

        gw, gh = self.grid_size
        grid = np.zeros((gw, gh), dtype=np.bool_)

        for i in range(gw):
            for j in range(gh):

                x = int(i * self.width / gw)
                y = int(j * self.height / gh)

                if self._point_in_poly(x, y):
                    grid[i, j] = True

        return grid

    # fast bbox check
    def intersects_bbox(self, bbox):
        x1, y1, x2, y2 = bbox

        gw, gh = self.grid_size

        gx1 = int(x1 / self.width * gw)
        gx2 = int(x2 / self.width * gw)
        gy1 = int(y1 / self.height * gh)
        gy2 = int(y2 / self.height * gh)

        gx1 = max(0, min(gw - 1, gx1))
        gx2 = max(0, min(gw - 1, gx2))
        gy1 = max(0, min(gh - 1, gy1))
        gy2 = max(0, min(gh - 1, gy2))

        # check only small region
        return np.any(self.grid[gx1:gx2+1, gy1:gy2+1])

    # fast mask check
    def intersects_mask(self, mask):
        return np.logical_and(mask, self.grid).any()

class BaseMusician(ABC):
    """
    Abstract base class for all music generation models.
    This class defines the interface that all music generation models must implement,
    ensuring consistency and extensibility across different generation strategies.
    """

    def __init__(self, tempo: int = 120, key_signature: str = "C_major"):
        """
        Initialize the base musician.

        Args:
            tempo: Music tempo in BPM
            key_signature: Key signature for music generation
        """

        self.tempo = tempo
        self.key_signature = key_signature

        self.frame_counter = 0

    def __call__(self,
        input: SegmentationResult,
        frame_id: int = 0,
        roi: Dict[str, Any] = None
    ):
        
        if not isinstance(input, SegmentationResult):
            raise ValueError("Input must be a SegmentationResult instance")

        return self.generate_music(input, frame_id, roi)

    @abstractmethod
    def generate_music(self,
        input: SegmentationResult,
        frame_id: int = 0,
        roi: Dict[str, Any] = None
    ):
        """
        Convenience method to call generate_music directly.

        Args:
            input: Segmentation result instance
            frame_id: Frame identifier for tracking
            roi: Region of interest for music generation

        Returns:
            MusicFrame containing generated music events
        """
        pass

    def extract_features(self, segmentation_data: np.ndarray) -> Dict[str, Any]:
        """
        Default feature extractor (can be overridden).

        Args:
            segmentation_data: Segmentation map as numpy array
        """

        return {
            "raw_shape": segmentation_data.shape,
            "unique_classes": np.unique(segmentation_data).tolist(),
        }

    def update_state(self, features: Dict[str, Any]):
        """
        Update temporal memory.
        Override if needed.

        Args:
            features: Dictionary of extracted features from the current frame
        """
        self.state.memory["last_features"] = features

class RuleBasedMusician(BaseMusician):
    """
    Rule-based musician that maps scene events to music events.
    This musician uses simple rules to generate music based on detected scene events,
    particularly focusing on objects interacting with a defined Region of Interest (ROI).
    """

    def __init__(self, tempo=120, key_signature="C_major"):
        """
        Args:
            tempo: Music tempo in BPM
            key_signature: Key signature for music generation
            roi: Optional ROI definition (corners + controls)
        """
        super().__init__(tempo, key_signature)
        
        # state: keeps track of objects currently touching ROI boundary
        self.state = {
            "touching": {}  # object_id -> bool
        }
        self.roi_grid = None

        logger.info(f"🎵 RuleBasedMusician initialized with tempo={tempo}, key_signature={key_signature}")

    def _set_roi(self, roi_payload, input_shape):
        
        if not roi_payload:
            return

        self.roi = ROI(corners=roi_payload.get("corners", []), 
                       controls=roi_payload.get("controls", []))

        if self.roi_grid == None:
            self.roi_grid = ROIGrid(
                polygon=self.roi.polygon,
                grid_size=(64, 64),
                width=input_shape[1] if input_shape is not None else 1280,
                height=input_shape[0] if input_shape is not None else 720
            )

    def _intersects_roi(self, bbox=None, mask=None):
        """
        Supports both:
        - YOLO: bbox-based
        - SegFormer: mask-based
        """

        # CASE 1: MASK (SegFormer)
        if mask is not None:
            return self.roi_grid.intersects_mask(mask)

        # CASE 2: BBOX (YOLO)
        if bbox is not None:
            return self.roi_grid.intersects_bbox(bbox)

        return False
    
    def extract_features(self, segmentation_result):
        return segmentation_result.metadata

    # SCENE EVENT DETECTION (ONLY ROI boundary)
    def detect_scene_events(self, bounding_boxes=None, masks=None):

        logger.info(f"Detecting scene events for {len(bounding_boxes) if bounding_boxes else 0} bounding boxes")

        events = []
            
        if masks is not None:

            for i, mask in enumerate(masks):

                obj_id = i
                obj_class = "unknown"  # Could be inferred from metadata if available

                touching = self._intersects_roi(mask=mask)
                prev = self.state["touching"].get(obj_id, False)

                logger.info(f"Object {obj_id} ({obj_class}) touching ROI: {touching} (mask), previously: {prev}")

                if touching and not prev:
                    events.append({
                        "type": "ROI_TOUCH",
                        "object_id": obj_id,
                        "class": obj_class
                    })
                    self.state["touching"][obj_id] = True

                elif not touching and prev:
                    events.append({
                        "type": "ROI_RELEASE",
                        "object_id": obj_id,
                        "class": obj_class
                    })
                    self.state["touching"][obj_id] = False

        elif bounding_boxes is not None:

            for i, obj in enumerate(bounding_boxes):

                obj_id = obj.get(i, "class_id")
                obj_class = obj.get("class_name", "unknown")
                bbox = obj["bbox"]

                touching = self._intersects_roi(bbox)
                prev = self.state["touching"].get(obj_id, False)

                logger.info(f"Object {obj_id} ({obj_class}) touching ROI: {touching} (bbox: {bbox}), previously: {prev}")

                if touching and not prev:
                    events.append({
                        "type": "ROI_TOUCH",
                        "object_id": obj_id,
                        "class": obj_class
                    })
                    self.state["touching"][obj_id] = True

                elif not touching and prev:
                    events.append({
                        "type": "ROI_RELEASE",
                        "object_id": obj_id,
                        "class": obj_class
                    })
                    self.state["touching"][obj_id] = False
        
        else:
            logger.warning("No bounding boxes or masks provided for scene event detection.")

        logger.info(f"Detected {len(events)} scene events: {events}")

        return events

    # MUSIC MAPPING (rule-based)
    def decide_music(self, scene_events, frame_id):

        music_events = []

        for e in scene_events:

            obj_class = e["class"]
            note = self._map_class_to_note(obj_class)
            velocity = self._velocity_from_class(obj_class)

            music_events.append(
                MusicEvent(
                    event_type="note_on" if e["type"] == "ROI_TOUCH" else "note_off",
                    note=note,
                    channel=0,
                    velocity=velocity if e["type"] == "ROI_TOUCH" else 0,
                    instrument="piano",  # Default instrument
                    timestamp=self.frame_counter,
                    metadata=e
                )
            )
            
            logger.info(f"Mapped scene event {e} to music event: note={note}, velocity={velocity}")

        # Log occasionally for debugging
        if self.frame_counter % 20 == 0:  # Every 20 frames
            logger.info(
                f"🎵 Generated {len(music_events)} music events for frame {frame_id}"
            )

        return music_events

    # MAIN PIPELINE
    def generate_music(self, result, frame_id, roi):
        """
        Generate music based on the input scene data.
        """

        logger.info(f"🎵 Generating music for frame {frame_id}")
        print(f"Segmentation map shape: {result.segmentation_map.shape}",
              f"Num Class labels: {len(result.class_labels) if result.class_labels else 0}",
              f"Num Bounding boxes: {len(result.bounding_boxes) if result.bounding_boxes else 0}",
              f"Num Masks: {len(result.masks) if result.masks else 0}",
              )

        self.frame_counter = frame_id
        
        self._set_roi(roi, result.segmentation_map.shape)

        scene_events = self.detect_scene_events(result.bounding_boxes, result.masks)

        music_events = self.decide_music(scene_events, frame_id)

        return MusicFrame(
            events=music_events,
            frame_id=frame_id,
            tempo=self.tempo,
            key_signature=self.key_signature,
            metadata={
                "scene_events": scene_events,
                "extra": result.metadata or {}
            }
        )
    
    # SIMPLE MAPPING
    def _map_class_to_note(self, obj_class):

        mapping = {
            "car": 60,      # Middle C
            "truck": 48,    # C2
            "bicycle": 64,  # E4
            "person": 72,   # C5
            "road": 36      # C1
        }

        return mapping.get(obj_class, 60)
    
    def _velocity_from_class(self, obj_class: str):

        base = {
            "car": 100,
            "truck": 80,
            "bicycle": 90,
            "person": 110,
            "road": 50
        }

        return base.get(obj_class, 70)


class Musician:
    """
    Main Musician class that provides a unified interface for different music generation models.

    This class acts as a factory and manager for different music generation models,
    allowing easy switching between models and unified result handling.
    """

    MUSICIAN_REGISTRY = {
        "rule-based": {
            "class": RuleBasedMusician,
            "label": "Rule-Based Musician",
            "description": "Rule-based multi-instrument demo mapping (drums, bass, strings, etc.).",
        },
        # "continuous_pianist": {
        #     "class": ContinuousPianistMusician,
        #     "label": "Continuous Pianist",
        #     "description": "Piano musician with sustained/continuous note playback.",
        # },
        # "lstm-onessen": {
        #     "class": LSTMMusician,
        #     "label": "LSTM (Essen Folk Song)",
        #     "description": "Neural LSTM model trained on the Essen folk song collection.",
        # },
    }

    def __init__(self, musician_type: str = "rule-based", tempo: int = 120, key_signature: str = "C_major"):
        """
        Initialize the main Musician.

        Args:
            musician_type: Type of musician, see Musician.MUSICIAN_REGISTRY for supported values.
            tempo: Music tempo in BPM
            key_signature: Key signature for music generation
        """

        self.musician_type = musician_type.lower()
        self.tempo = tempo
        self.key_signature = key_signature
        self.musician = self._create_musician(musician_type, tempo, key_signature)

        logger.info(f"🎵 Musician initialized: {musician_type}")

    def _create_musician(self, musician_type: str, tempo: int, key_signature: str) -> BaseMusician:
        """Create the appropriate musician based on type."""

        entry = self.MUSICIAN_REGISTRY.get(musician_type.lower())
        if entry is None:
            available = ", ".join(sorted(self.MUSICIAN_REGISTRY.keys()))
            raise ValueError(f"Unsupported musician type: {musician_type}. Supported types: {available}")
        
        return entry["class"](tempo, key_signature)

    def __call__(self, 
                 input,  # result
                 frame_id: int = 0,
                 roi: Dict[str, Any] = None,
                 ) -> MusicFrame:
        """
        Generate music based on segmentation data.

        Args:
            input: Segmentation result
            frame_id: Frame identifier for tracking
            roi: Region of interest data

        Returns:
            MusicFrame containing generated music events
        """

        if not isinstance(input, SegmentationResult):
            raise ValueError("Input must be a SegmentationResult instance")

        return self.musician(input, frame_id, roi)

    def switch_musician(self, musician_type: str, tempo: int = None, key_signature: str = None) -> None:
        """
        Switch to a different music generation model.

        Args:
            musician_type: New musician type
            tempo: New tempo (keeps current if None)
            key_signature: New key signature (keeps current if None)
        """

        self.musician_type = musician_type.lower()
        if tempo is not None:
            self.tempo = tempo
        if key_signature is not None:
            self.key_signature = key_signature

        self.musician = self._create_musician(musician_type, self.tempo, self.key_signature)

        logger.info(f"🔄 Switched to {musician_type} musician")

    @classmethod
    def list_available_musicians(cls) -> List[dict]:
        """
        Return metadata for every musician type that can be selected/switched to.

        Used by the Platform UI to populate the "Change Musician" picker without
        duplicating the list of supported types.
        """

        return [
            {"id": musician_id, "label": info["label"], "description": info["description"]}
            for musician_id, info in cls.MUSICIAN_REGISTRY.items()
        ]
