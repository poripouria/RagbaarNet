"""
Modular Music Generation Framework for Real-Time Visual-to-Audio Mapping
========================================================================

This module provides an extensible framework for generating music based on visual data,
particularly segmentation maps from computer vision models. It supports various music
generation strategies with easy integration for additional models.
"""

import os
import sys
import time
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


class RuleBasedMusician(BaseMusician):

    def __init__(self, tempo: int = 120, key_signature: str = "C_major", roi=None):
        """
        Initialize the rule-based musician.
        
        Args:
            tempo: Music tempo in BPM
            key_signature: Key signature for music generation
            roi: Region of interest (polygon points) for scene event detection
        """

        super().__init__(tempo, key_signature)

        self.roi = roi  # polygon / bezier converted to polygon points
        self.state = {
            "active_notes": {},   # object_id -> note
        }

    def extract_features(self, segmentation_data: np.ndarray) -> Dict[str, Any]:

        unique, counts = np.unique(segmentation_data, return_counts=True)

        total = segmentation_data.size

        class_pixels = {str(k): int(v) for k, v in zip(unique, counts)}
        class_ratios = {k: v / total for k, v in class_pixels.items()}

        return {
            "class_pixels": class_pixels,
            "class_ratios": class_ratios
        }

    # ------------------------------------------------------------
    # 2) SCENE EVENT DETECTION (ROI-based)
    # ------------------------------------------------------------
    def detect_scene_events(self, features: Dict[str, Any], frame_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        frame_data must include object blobs:
        [
            {object_id, class, mask, bbox, centroid}
        ]
        """

        events = []

        objects = frame_data.get("objects", [])

        for obj in objects:
            obj_id = obj["object_id"]
            obj_class = obj["class"]
            centroid = obj["centroid"]

            inside = self._inside_roi(centroid)

            prev_state = self.state["active_notes"].get(obj_id, None)

            # ENTER
            if inside and prev_state is None:
                events.append({
                    "type": "ROI_ENTER",
                    "object_id": obj_id,
                    "class": obj_class,
                    "centroid": centroid
                })
                self.state["active_notes"][obj_id] = None

            # EXIT
            elif not inside and prev_state is not None:
                events.append({
                    "type": "ROI_EXIT",
                    "object_id": obj_id,
                    "class": obj_class,
                    "centroid": centroid
                })
                del self.state["active_notes"][obj_id]

        return events

    # ------------------------------------------------------------
    # 3) MUSIC DECISION
    # ------------------------------------------------------------
    def decide_music(self, scene_events: List[Dict[str, Any]]) -> List[MusicEvent]:

        music_events = []

        for e in scene_events:

            obj_class = e["class"]

            # mapping rule (can later be replaced by LSTM/Transformer)
            note = self._map_class_to_note(obj_class)

            if e["type"] == "ROI_ENTER":

                music_events.append(
                    MusicEvent(
                        note=note,
                        velocity=self._velocity(obj_class, entering=True),
                        channel=0,
                        timestamp=self.frame_counter,
                        metadata={"event": "note_on", "object_id": e["object_id"]}
                    )
                )

                self.state["active_notes"][e["object_id"]] = note

            elif e["type"] == "ROI_EXIT":

                music_events.append(
                    MusicEvent(
                        note=self.state["active_notes"].get(e["object_id"], note),
                        velocity=0,
                        channel=0,
                        timestamp=self.frame_counter,
                        metadata={"event": "note_off", "object_id": e["object_id"]}
                    )
                )

        return music_events

    # ------------------------------------------------------------
    # 4) MAIN PIPELINE
    # ------------------------------------------------------------
    def generate_music(
        self,
        segmentation_data: np.ndarray,
        frame_data: Dict[str, Any],
        frame_id: int = 0,
        class_labels: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> MusicFrame:

        self.frame_counter = frame_id

        features = self.extract_features(segmentation_data)

        scene_events = self.detect_scene_events(features, frame_data)

        music_events = self.decide_music(scene_events)

        return MusicFrame(
            events=music_events,
            frame_id=frame_id,
            timestamp=float(frame_id),
            tempo=self.tempo,
            key_signature=self.key_signature,
            metadata={
                "features": features,
                "scene_events": scene_events,
                "extra": metadata or {}
            }
        )

    # ------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------
    def _inside_roi(self, point: Tuple[float, float]) -> bool:
        """
        simple polygon test (ray casting placeholder)
        """

        if self.roi is None:
            return True

        x, y = point
        poly = self.roi

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

    def _map_class_to_note(self, obj_class: str) -> int:
        mapping = {
            "car": 60,
            "truck": 48,
            "person": 72,
            "road": 36,
            "traffic_sign": 76
        }
        return mapping.get(obj_class, 60)

    def _velocity(self, obj_class: str, entering: bool = True) -> int:
        base = {
            "car": 90,
            "truck": 70,
            "person": 100,
            "road": 50
        }
        v = base.get(obj_class, 80)
        return v if entering else int(v * 0.6)

class Musician:
    """
    Main Musician class that provides a unified interface for different music generation models.

    This class acts as a factory and manager for different music generation models,
    allowing easy switching between models and unified result handling.
    """

    MUSICIAN_REGISTRY = {
        "test": {
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
