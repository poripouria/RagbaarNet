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

logger = setup_logging("INFO", name="Music_Generator.Musician")


@dataclass
class MusicEvent:
    """
    Core atomic event in the music system.

    This represents a single musical action (NOT only MIDI note).
    Designed to be extendable for future MIDI CC, pitch bend, etc.

    Attributed:
        event_type: Type of the event (e.g., "note_on", "note_off", "control_change")
        timestamp: Time at which the event occurs
        channel: MIDI channel (0-15)
        note: MIDI note number (0-127), optional depending on event_type
        velocity: Note velocity (0-127), optional depending on event_type
        metadata: Additional event-specific information
    """

    event_type: str  # e.g. "note_on", "note_off"
    timestamp: float = 0.0
    channel: int = 0
    # Note-related fields (optional depending on event_type)
    note: Optional[int] = None
    velocity: Optional[int] = None
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

@dataclass
class MusicianState:
    """
    Persistent state across frames for temporal coherence.
    Used to preserve musical continuity in real-time generation.

    Attributes:
        last_notes: List of last played notes
        current_beat: Current beat position in the bar
        current_bar: Current bar number
        memory: Arbitrary state storage for musician-specific data
    """

    last_notes: List[int] = field(default_factory=list)
    current_beat: float = 0.0
    current_bar: int = 0
    memory: Dict[str, Any] = field(default_factory=dict)
    # last_velocity: int = 64
    # active_chords: List[int] = field(default_factory=list)
    # tension: float = 0.0

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
        self.state = MusicianState()

    def __call__(self,
        segmentation_data: np.ndarray,
        frame_id: int = 0,
        class_labels: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> MusicFrame:
        
        if not isinstance(segmentation_data, np.ndarray):
            raise ValueError("segmentation_data must be a numpy array")

        return self.generate_music(segmentation_data, frame_id, class_labels, metadata)

    @abstractmethod
    def generate_music(self,
        segmentation_data: np.ndarray,
        frame_id: int = 0,
        class_labels: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MusicFrame:
        """
        Convenience method to call generate_music directly.

        Args:
            segmentation_data: Segmentation map as numpy array
            frame_id: Frame identifier for tracking
            class_labels: Optional list of class labels
            metadata: Optional dictionary of metadata

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

class RuleBasedMusician(BaseMusician):
    """
    Rule-based musician that maps scene events to music events.
    This musician uses simple rules to generate music based on detected scene events,
    particularly focusing on objects interacting with a defined Region of Interest (ROI).
    """

    def __init__(self, tempo=120, key_signature="C_major", roi=None):
        """
        Args:
            tempo: Music tempo in BPM
            key_signature: Key signature for music generation
            roi: Optional ROI definition (corners + controls)
        """
        super().__init__(tempo, key_signature)
        
        self.roi = None
        self.roi_grid = None

        # state: keeps track of objects currently touching ROI boundary
        self.state = {
            "touching": {}  # object_id -> bool
        }

        if roi:
            self._set_roi(roi)

    def _set_roi(self, roi_payload):
        
        if not roi_payload:
            return

        corners = roi_payload.get("corners", [])
        controls = roi_payload.get("controls", [])

        if len(corners) != len(controls):
            return

        self.roi = ROI(corners=corners, controls=controls)

    # ROI boundary check (bbox-based)
    def _bbox_edges(self, bbox):
        x1, y1, x2, y2 = bbox

        return [
            ((x1, y1), (x2, y1)),   # top edge
            ((x2, y1), (x2, y2)),   # right edge
            ((x2, y2), (x1, y2)),   # bottom edge
            ((x1, y2), (x1, y1)),   # left edge
        ]

    def _segments_intersect(self, a, b):
        """
        Check if two line segments intersect.
        Each segment is defined by two endpoints: a = (A, B), b = (C, D)
        """

        def ccw(A, B, C):
            """
            Check if the points A, B, C are listed in counter-clockwise order.
            """
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

        A, B = a
        C, D = b

        return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))

    def _intersects_roi(self, bbox=None, mask=None):
        """
        Supports both:
        - YOLO: bbox-based
        - SegFormer: mask-based
        """

        # CASE 1: MASK (SegFormer)
        if mask is not None:
            return self._mask_intersects_roi(mask)

        # CASE 2: BBOX (YOLO)
        if bbox is not None:
            return self._bbox_intersects_roi(bbox)

        return False

    def _mask_intersects_roi(self, mask: np.ndarray):

        roi_grid = self.roi_grid.grid  # precomputed ROI mask (same resolution)

        return np.logical_and(mask, roi_grid).any()

    def _bbox_intersects_roi(self, bbox):
        return self.roi_grid.intersects_bbox(bbox)

    # FEATURE (optional lightweight)
    def extract_features(self, segmentation_result):
        return segmentation_result.metadata

    # SCENE EVENT DETECTION (ONLY ROI boundary)
    def detect_scene_events(self, bounding_boxes, class_labels):

        events = []

        if bounding_boxes is None:
            return events

        for obj in bounding_boxes:

            obj_id = obj["object_id"]
            class_id = obj.get("class_id", None)

            obj_class = (
                class_labels[class_id]
                if class_labels and class_id is not None
                else "unknown"
            )

            bbox = obj["bbox"]

            touching = self._intersects_roi(bbox)
            prev = self.state["touching"].get(obj_id, False)

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

        return events

    # MUSIC MAPPING (rule-based)
    def decide_music(self, scene_events, class_labels=None):

        music_events = []

        for e in scene_events:

            obj_class = e["class"]

            note = self._map_class_to_note(obj_class)

            velocity = self._velocity_from_class(obj_class)

            if e["type"] == "ROI_TOUCH":

                music_events.append(
                    MusicEvent(
                        note=note,
                        velocity=velocity,
                        channel=0,
                        timestamp=self.frame_counter,
                        metadata=e
                    )
                )

            elif e["type"] == "ROI_RELEASE":

                music_events.append(
                    MusicEvent(
                        note=note,
                        velocity=0,
                        channel=0,
                        timestamp=self.frame_counter,
                        metadata=e
                    )
                )

        return music_events

    # MAIN PIPELINE
    def generate_music(
        self,
        segmentation_map: np.ndarray,
        frame_id: int = 0,
        class_labels: List[str] = None,
        confidence_map: np.ndarray = None,
        bounding_boxes: List[Dict] = None,
        masks: List[np.ndarray] = None,
        metadata: Dict[str, Any] = None
    ):

        self.frame_counter = frame_id
        
        
        self.roi_grid = ROIGrid(
            polygon=self.roi.polygon if self.roi else [],
            grid_size=(64, 64),
            width=segmentation_map.shape[1],
            height=segmentation_map.shape[0]
        )

        scene_events = self.detect_scene_events(
            bounding_boxes=bounding_boxes,
            class_labels=class_labels
        )

        music_events = self.decide_music(scene_events, class_labels)

        return MusicFrame(
            events=music_events,
            frame_id=frame_id,
            tempo=self.tempo,
            key_signature=self.key_signature,
            metadata={
                "scene_events": scene_events,
                "extra": metadata or {}
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

    def __init__(self, musician_type: str = "test", tempo: int = 120, key_signature: str = "C_major"):
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

    def __call__(self, segmentation_data: np.ndarray, frame_id: int = 0, class_labels: List[str] = None, metadata: Dict[str, Any] = None) -> MusicFrame:
        """
        Generate music based on segmentation data.

        Args:
            segmentation_data: Segmentation map as numpy array
            frame_id: Frame identifier for tracking

        Returns:
            MusicFrame containing generated music events
        """

        if not isinstance(segmentation_data, np.ndarray):
            raise ValueError("Segmentation data must be a numpy array")

        return self.musician(segmentation_data, frame_id, class_labels=class_labels, metadata=metadata)

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
