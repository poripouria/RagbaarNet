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

class TestMusician(BaseMusician):
    """
    Test musician implementation that deterministically assigns musical elements to objects.

    Maps segmentation classes to specific musical notes and patterns:
    - Cars → Piano notes (C major scale)
    - Traffic signs → B minor chord variations
    - Roads → Drum patterns
    - Other objects → Additional instrument assignments
    """

    def __init__(self, tempo: int = 120, key_signature: str = "C_major"):
        """
        Initialize Test Musician.

        Args:
            tempo: Music tempo in BPM
            key_signature: Key signature for music generation
        """

        super().__init__(tempo, key_signature)

        # Cityscapes class labels (matching Segformer model)
        self.cityscapes_labels = [
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic light",
            "traffic sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]
        self.class_labels = list(self.cityscapes_labels)

        # Musical mappings for different object classes
        self.class_to_music = {}
        self._setup_music_mappings(self.class_labels)

        logger.info("✅ Test Musician initialized successfully")

    def _setup_music_mappings(self, class_labels: List[str] = None) -> None:
        """Setup deterministic mappings from segmentation classes to musical elements."""

        labels = list(class_labels or self.class_labels or self.cityscapes_labels)
        self.class_labels = labels
        self.class_to_music = {}

        # C Major scale notes (MIDI numbers)
        c_major_scale = [60, 62, 64, 65, 67, 69, 71]  # C4, D4, E4, F4, G4, A4, B4

        # B Minor chord variations (MIDI numbers)
        b_minor_chord = [59, 62, 66]  # B3, D4, F#4
        b_minor_variations = [
            [59, 62, 66],  # Root position
            [62, 66, 71],  # First inversion
            [66, 71, 59 + 12],  # Second inversion
        ]

        # Drum patterns (using MIDI standard drum map on channel 9)
        drum_patterns = {
            "kick": 36,  # Bass drum
            "snare": 38,  # Acoustic snare
            "hihat": 42,  # Closed hi-hat
            "crash": 49,  # Crash cymbal
            "ride": 51,  # Ride cymbal
        }

        # Map each class to specific musical elements
        for i, class_name in enumerate(labels):
            if class_name == "car":
                # Cars get C major scale notes
                note_idx = i % len(c_major_scale)
                self.class_to_music[i] = {
                    "note": c_major_scale[note_idx],
                    "channel": 0,  # Piano channel
                    "velocity": 80,
                    "duration": 0.5,
                    "instrument": "piano",
                }

            elif class_name == "traffic sign":
                # Traffic signs get B minor variations
                chord_idx = i % len(b_minor_variations)
                self.class_to_music[i] = {
                    "note": b_minor_variations[chord_idx][0],  # Root note
                    "channel": 1,  # Different channel for traffic signs
                    "velocity": 70,
                    "duration": 0.8,
                    "instrument": "electric_piano",
                }

            elif class_name == "road":
                # Roads get drum patterns
                self.class_to_music[i] = {
                    "note": drum_patterns["kick"],
                    "channel": 9,  # Standard MIDI drum channel
                    "velocity": 90,
                    "duration": 0.3,
                    "instrument": "drums",
                }

            elif class_name == "truck":
                # Trucks get bass notes
                self.class_to_music[i] = {
                    "note": 48 + (i % 12),  # Bass octave
                    "channel": 2,
                    "velocity": 85,
                    "duration": 1.0,
                    "instrument": "bass",
                }

            elif class_name == "person":
                # People get violin-like sounds
                note_idx = i % len(c_major_scale)
                self.class_to_music[i] = {
                    "note": c_major_scale[note_idx] + 12,  # One octave higher
                    "channel": 3,
                    "velocity": 60,
                    "duration": 0.7,
                    "instrument": "strings",
                }

            elif class_name == "motorcycle":
                # Motorcycles get electric guitar
                self.class_to_music[i] = {
                    "note": 55 + (i % 8),  # Mid-range notes
                    "channel": 4,
                    "velocity": 95,
                    "duration": 0.4,
                    "instrument": "electric_guitar",
                }

            elif class_name == "bicycle":
                # Bicycles get acoustic guitar
                self.class_to_music[i] = {
                    "note": 50 + (i % 10),
                    "channel": 5,
                    "velocity": 65,
                    "duration": 0.6,
                    "instrument": "acoustic_guitar",
                }

            elif class_name in ["sidewalk", "building"]:
                # Infrastructure gets pad sounds
                self.class_to_music[i] = {
                    "note": 36 + (i % 24),  # Wide range for ambience
                    "channel": 6,
                    "velocity": 40,
                    "duration": 2.0,
                    "instrument": "pad",
                }

            else:
                # Default mapping for other classes
                note = 60 + (i % 12)  # Chromatic scale from C4
                self.class_to_music[i] = {
                    "note": note,
                    "channel": 7,
                    "velocity": 50,
                    "duration": 0.5,
                    "instrument": "synth",
                }

    def generate_music(self, segmentation_data: np.ndarray, frame_id: int = 0, class_labels: List[str] = None, metadata: Dict[str, Any] = None) -> MusicFrame:
        """
        Generate music based on segmentation data.

        Args:
            segmentation_data: Segmentation map as numpy array
            frame_id: Frame identifier for tracking

        Returns:
            MusicFrame containing generated music events
        """

        resolved_class_labels = list(class_labels or self.class_labels or self.cityscapes_labels)
        if resolved_class_labels != self.class_labels:
            self._setup_music_mappings(resolved_class_labels)

        timestamp = time.time()
        events = []

        # Analyze segmentation data
        unique_classes, counts = np.unique(segmentation_data, return_counts=True)
        total_pixels = segmentation_data.shape[0] * segmentation_data.shape[1]

        # Generate music events based on detected classes
        for class_id, pixel_count in zip(unique_classes, counts):
            # Skip background class (0) if it's too dominant
            if class_id == 0 and pixel_count > total_pixels * 0.8:
                continue

            # Calculate presence ratio
            presence_ratio = pixel_count / total_pixels

            # Only generate events for classes with significant presence
            if presence_ratio > 0.01:  # At least 1% of the frame
                if class_id in self.class_to_music:
                    mapping = self.class_to_music[class_id]

                    # Adjust velocity based on presence ratio
                    adjusted_velocity = min(
                        127, int(mapping["velocity"] * (1 + presence_ratio * 2))
                    )

                    # Adjust duration based on presence ratio
                    adjusted_duration = mapping["duration"] * (0.5 + presence_ratio)

                    event = MusicEvent(
                        note=mapping["note"],
                        velocity=adjusted_velocity,
                        duration=adjusted_duration,
                        channel=mapping["channel"],
                        timestamp=timestamp,
                        metadata={
                            "class_id": int(class_id),
                            "class_name": resolved_class_labels[class_id]
                            if class_id < len(resolved_class_labels)
                            else "unknown",
                            "presence_ratio": float(presence_ratio),
                            "pixel_count": int(pixel_count),
                            "instrument": mapping["instrument"],
                        },
                    )
                    events.append(event)

        # Create frame result
        music_frame = MusicFrame(
            events=events,
            frame_id=frame_id,
            timestamp=timestamp,
            tempo=self.tempo,
            key_signature=self.key_signature,
            metadata={
                "musician_type": "TestMusician",
                "total_classes_detected": len(unique_classes),
                "total_events_generated": len(events),
                "segmentation_shape": segmentation_data.shape,
            },
        )

        self.frame_counter += 1

        # Log occasionally for debugging
        if self.frame_counter % 30 == 0:  # Every 30 frames
            logger.debug(
                f"🎵 Generated {len(events)} music events for frame {frame_id}"
            )
            logger.debug(
                f"🎯 Detected classes: {[e.metadata['class_name'] for e in events]}"
            )

        return music_frame

class PianistTestMusician(BaseMusician):
    """
    Simplified pianist musician that maps all segmentation classes to piano notes only.

    Maps segmentation classes to piano notes using different scales and patterns:
    - All objects get piano sounds with different note ranges
    - Uses C major, D minor, and pentatonic scales for variety
    - Velocity and duration vary based on object type and presence
    """

    def __init__(self, tempo: int = 120, key_signature: str = "C_major"):
        """
        Initialize Pianist Test Musician.

        Args:
            tempo: Music tempo in BPM
            key_signature: Key signature for music generation
        """

        super().__init__(tempo, key_signature)

        # Cityscapes class labels (matching Segformer model)
        self.cityscapes_labels = [
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic light",
            "traffic sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]
        self.class_labels = list(self.cityscapes_labels)

        # Piano-only musical mappings
        self.class_to_piano = {}
        self._setup_piano_mappings(self.class_labels)

        logger.info("✅ Pianist Test Musician initialized successfully")

    def _setup_piano_mappings(self, class_labels: List[str] = None) -> None:
        """Setup piano-only mappings from segmentation classes to piano notes."""

        labels = list(class_labels or self.class_labels or self.cityscapes_labels)
        self.class_labels = labels
        self.class_to_piano = {}

        # Different piano note ranges and patterns
        c_major_scale = [60, 62, 64, 65, 67, 69, 71]  # C4, D4, E4, F4, G4, A4, B4
        d_minor_scale = [62, 64, 65, 67, 69, 70, 72]  # D4, E4, F4, G4, A4, Bb4, C5
        pentatonic_scale = [60, 62, 65, 67, 69]  # C4, D4, F4, G4, A4
        bass_notes = [36, 38, 40, 43, 45, 47, 48]  # Bass octave
        high_notes = [72, 74, 76, 77, 79, 81, 83]  # High octave

        # Map each class to specific piano elements
        for i, class_name in enumerate(labels):
            if class_name in ["car", "truck", "bus"]:
                # Vehicles get C major scale - mid range
                note_idx = i % len(c_major_scale)
                self.class_to_piano[i] = {
                    "note": c_major_scale[note_idx],
                    "velocity": 80,
                    "duration": 0.6,
                    "scale_type": "C_major",
                }

            elif class_name in ["person", "rider"]:
                # People get high piano notes - more delicate
                note_idx = i % len(high_notes)
                self.class_to_piano[i] = {
                    "note": high_notes[note_idx],
                    "velocity": 65,
                    "duration": 0.8,
                    "scale_type": "high_range",
                }

            elif class_name in ["road", "sidewalk"]:
                # Infrastructure gets bass notes - foundation
                note_idx = i % len(bass_notes)
                self.class_to_piano[i] = {
                    "note": bass_notes[note_idx],
                    "velocity": 90,
                    "duration": 1.2,
                    "scale_type": "bass_range",
                }

            elif class_name in ["building", "wall", "fence"]:
                # Structures get D minor scale - more complex
                note_idx = i % len(d_minor_scale)
                self.class_to_piano[i] = {
                    "note": d_minor_scale[note_idx],
                    "velocity": 70,
                    "duration": 1.0,
                    "scale_type": "D_minor",
                }

            elif class_name in ["traffic light", "traffic sign", "pole"]:
                # Traffic elements get pentatonic - pleasant
                note_idx = i % len(pentatonic_scale)
                self.class_to_piano[i] = {
                    "note": pentatonic_scale[note_idx] + 12,  # One octave higher
                    "velocity": 75,
                    "duration": 0.5,
                    "scale_type": "pentatonic",
                }

            elif class_name in ["vegetation", "terrain", "sky"]:
                # Natural elements get soft piano - ambient
                note_idx = i % len(c_major_scale)
                self.class_to_piano[i] = {
                    "note": c_major_scale[note_idx] - 12,  # One octave lower
                    "velocity": 50,
                    "duration": 1.5,
                    "scale_type": "ambient",
                }

            elif class_name in ["motorcycle", "bicycle"]:
                # Two-wheelers get pentatonic mid-range
                note_idx = i % len(pentatonic_scale)
                self.class_to_piano[i] = {
                    "note": pentatonic_scale[note_idx],
                    "velocity": 85,
                    "duration": 0.4,
                    "scale_type": "pentatonic_mid",
                }

            else:
                # Default piano mapping for other classes
                note_idx = i % len(c_major_scale)
                self.class_to_piano[i] = {
                    "note": c_major_scale[note_idx],
                    "velocity": 60,
                    "duration": 0.7,
                    "scale_type": "default",
                }

    def generate_music(self, segmentation_data: np.ndarray, frame_id: int = 0, class_labels: List[str] = None, metadata: Dict[str, Any] = None) -> MusicFrame:
        """
        Generate piano music based on segmentation data.

        Args:
            segmentation_data: Segmentation map as numpy array
            frame_id: Frame identifier for tracking

        Returns:
            MusicFrame containing generated piano music events
        """

        resolved_class_labels = list(class_labels or self.class_labels or self.cityscapes_labels)
        if resolved_class_labels != self.class_labels:
            self._setup_piano_mappings(resolved_class_labels)

        timestamp = time.time()
        events = []

        # Analyze segmentation data
        unique_classes, counts = np.unique(segmentation_data, return_counts=True)
        total_pixels = segmentation_data.shape[0] * segmentation_data.shape[1]

        # Generate piano events based on detected classes
        for class_id, pixel_count in zip(unique_classes, counts):
            # Skip background class (0) if it's too dominant
            if class_id == 0 and pixel_count > total_pixels * 0.8:
                continue

            # Calculate presence ratio
            presence_ratio = pixel_count / total_pixels

            # Only generate events for classes with significant presence
            if presence_ratio > 0.01:  # At least 1% of the frame
                if class_id in self.class_to_piano:
                    mapping = self.class_to_piano[class_id]

                    # Adjust velocity based on presence ratio
                    adjusted_velocity = min(
                        127, int(mapping["velocity"] * (1 + presence_ratio * 1.5))
                    )

                    # Adjust duration based on presence ratio
                    adjusted_duration = mapping["duration"] * (
                        0.5 + presence_ratio * 1.5
                    )

                    event = MusicEvent(
                        note=mapping["note"],
                        velocity=adjusted_velocity,
                        duration=adjusted_duration,
                        channel=0,  # All piano events on channel 0
                        timestamp=timestamp,
                        metadata={
                            "class_id": int(class_id),
                            "class_name": resolved_class_labels[class_id]
                            if class_id < len(resolved_class_labels)
                            else "unknown",
                            "presence_ratio": float(presence_ratio),
                            "pixel_count": int(pixel_count),
                            "instrument": "piano",
                            "scale_type": mapping["scale_type"],
                        },
                    )
                    events.append(event)

        # Create frame result
        music_frame = MusicFrame(
            events=events,
            frame_id=frame_id,
            timestamp=timestamp,
            tempo=self.tempo,
            key_signature=self.key_signature,
            metadata={
                "musician_type": "PianistTestMusician",
                "total_classes_detected": len(unique_classes),
                "total_events_generated": len(events),
                "segmentation_shape": segmentation_data.shape,
                "instrument": "piano_only",
            },
        )

        self.frame_counter += 1

        # Log occasionally for debugging
        if self.frame_counter % 30 == 0:  # Every 30 frames
            logger.debug(
                f"🎹 Generated {len(events)} piano events for frame {frame_id}"
            )
            logger.debug(
                f"🎯 Detected classes: {[e.metadata['class_name'] for e in events]}"
            )

        return music_frame

class ContinuousPianistMusician(BaseMusician):
    """
    Continuous pianist musician that plays notes continuously while objects touch image borders.

    This musician tracks object collisions with image edges and maintains continuous note playback
    while the collision persists. Notes start when collision begins and stop when collision ends.
    """

    def __init__(self, tempo: int = 120, key_signature: str = "C_major"):
        """
        Initialize Continuous Pianist Musician.

        Args:
            tempo: Music tempo in BPM
            key_signature: Key signature for music generation
        """

        super().__init__(tempo, key_signature)

        # Cityscapes class labels (matching Segformer model)
        self.cityscapes_labels = [
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic light",
            "traffic sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]

        self.class_labels = list(self.cityscapes_labels)

        # Piano-only musical mappings
        self.class_to_piano = {}
        self._setup_piano_mappings(self.class_labels)

        # Continuous playback state tracking
        self.active_notes = {}  # Track currently playing notes per class
        self.collision_history = {}  # Track collision state history
        self.note_start_times = {}  # Track when notes started playing

        logger.info("✅ Continuous Pianist Musician initialized successfully")

    def _setup_piano_mappings(self, class_labels: List[str] = None) -> None:
        """Setup piano-only mappings from segmentation classes to piano notes."""

        labels = list(class_labels or self.class_labels or self.cityscapes_labels)
        self.class_labels = labels
        self.class_to_piano = {}

        # Different piano note ranges and patterns
        c_major_scale = [60, 62, 64, 65, 67, 69, 71]  # C4, D4, E4, F4, G4, A4, B4
        d_minor_scale = [62, 64, 65, 67, 69, 70, 72]  # D4, E4, F4, G4, A4, Bb4, C5
        pentatonic_scale = [60, 62, 65, 67, 69]  # C4, D4, F4, G4, A4
        bass_notes = [36, 38, 40, 43, 45, 47, 48]  # Bass octave
        high_notes = [72, 74, 76, 77, 79, 81, 83]  # High octave

        # Map each class to specific piano elements
        for i, class_name in enumerate(labels):
            if class_name in ["car", "truck", "bus"]:
                # Vehicles get C major scale - mid range
                note_idx = i % len(c_major_scale)
                self.class_to_piano[i] = {
                    "note": c_major_scale[note_idx],
                    "velocity": 80,
                    "base_duration": 0.6,
                    "scale_type": "C_major",
                }

            elif class_name in ["person", "rider"]:
                # People get high piano notes - more delicate
                note_idx = i % len(high_notes)
                self.class_to_piano[i] = {
                    "note": high_notes[note_idx],
                    "velocity": 65,
                    "base_duration": 0.8,
                    "scale_type": "high_range",
                }

            elif class_name in ["road", "sidewalk"]:
                # Infrastructure gets bass notes - foundation
                note_idx = i % len(bass_notes)
                self.class_to_piano[i] = {
                    "note": bass_notes[note_idx],
                    "velocity": 90,
                    "base_duration": 1.2,
                    "scale_type": "bass_range",
                }

            elif class_name in ["building", "wall", "fence"]:
                # Structures get D minor scale - more complex
                note_idx = i % len(d_minor_scale)
                self.class_to_piano[i] = {
                    "note": d_minor_scale[note_idx],
                    "velocity": 70,
                    "base_duration": 1.0,
                    "scale_type": "D_minor",
                }

            elif class_name in ["traffic light", "traffic sign", "pole"]:
                # Traffic elements get pentatonic - pleasant
                note_idx = i % len(pentatonic_scale)
                self.class_to_piano[i] = {
                    "note": pentatonic_scale[note_idx] + 12,  # One octave higher
                    "velocity": 75,
                    "base_duration": 0.5,
                    "scale_type": "pentatonic",
                }

            elif class_name in ["vegetation", "terrain", "sky"]:
                # Natural elements get soft piano - ambient
                note_idx = i % len(c_major_scale)
                self.class_to_piano[i] = {
                    "note": c_major_scale[note_idx] - 12,  # One octave lower
                    "velocity": 50,
                    "base_duration": 1.5,
                    "scale_type": "ambient",
                }

            elif class_name in ["motorcycle", "bicycle"]:
                # Two-wheelers get pentatonic mid-range
                note_idx = i % len(pentatonic_scale)
                self.class_to_piano[i] = {
                    "note": pentatonic_scale[note_idx],
                    "velocity": 85,
                    "base_duration": 0.4,
                    "scale_type": "pentatonic_mid",
                }

            else:
                # Default piano mapping for other classes
                note_idx = i % len(c_major_scale)
                self.class_to_piano[i] = {
                    "note": c_major_scale[note_idx],
                    "velocity": 60,
                    "base_duration": 0.7,
                    "scale_type": "default",
                }

    def _check_edge_collision(self, seg_map: np.ndarray, class_id: int) -> dict:
        """
        Check if a class touches any of the four edges of the image.

        Args:
            seg_map: Segmentation map
            class_id: Class ID to check

        Returns:
            dict: Which edges are touched {'top': bool, 'bottom': bool, 'left': bool, 'right': bool}
        """

        class_mask = seg_map == class_id
        height, width = seg_map.shape

        edges_touched = {"top": False, "bottom": False, "left": False, "right": False}

        if np.any(class_mask):
            # Check top edge (row 0)
            edges_touched["top"] = np.any(class_mask[0, :])

            # Check bottom edge (last row)
            edges_touched["bottom"] = np.any(class_mask[height - 1, :])

            # Check left edge (column 0)
            edges_touched["left"] = np.any(class_mask[:, 0])

            # Check right edge (last column)
            edges_touched["right"] = np.any(class_mask[:, width - 1])

        return edges_touched

    def _has_edge_collision(self, edges_touched: dict) -> bool:
        """Check if any edge collision exists."""

        return any(edges_touched.values())

    def generate_music(self, segmentation_data: np.ndarray, frame_id: int = 0, class_labels: List[str] = None, metadata: Dict[str, Any] = None) -> MusicFrame:
        """
        Generate continuous piano music based on edge collisions.

        Args:
            segmentation_data: Segmentation map as numpy array
            frame_id: Frame identifier for tracking

        Returns:
            MusicFrame containing continuous piano music events
        """

        resolved_class_labels = list(class_labels or self.class_labels or self.cityscapes_labels)
        if resolved_class_labels != self.class_labels:
            self._setup_piano_mappings(resolved_class_labels)

        timestamp = time.time()
        events = []

        # Analyze segmentation data
        unique_classes, counts = np.unique(segmentation_data, return_counts=True)
        total_pixels = segmentation_data.shape[0] * segmentation_data.shape[1]

        # Track current collision states
        current_collisions = {}

        # Check edge collisions for each detected class
        for class_id, pixel_count in zip(unique_classes, counts):
            # Skip background class (0) if it's too dominant
            if class_id == 0 and pixel_count > total_pixels * 0.8:
                continue

            # Calculate presence ratio
            presence_ratio = pixel_count / total_pixels

            # Only process classes with significant presence
            if presence_ratio > 0.01:  # At least 1% of the frame
                # Check edge collision
                edges_touched = self._check_edge_collision(segmentation_data, class_id)
                has_collision = self._has_edge_collision(edges_touched)

                current_collisions[class_id] = {
                    "has_collision": has_collision,
                    "edges_touched": edges_touched,
                    "presence_ratio": presence_ratio,
                    "pixel_count": pixel_count,
                }

        # Process collision state changes and generate continuous events
        for class_id, collision_data in current_collisions.items():
            if class_id in self.class_to_piano:
                mapping = self.class_to_piano[class_id]
                class_name = (
                    resolved_class_labels[class_id]
                    if class_id < len(resolved_class_labels)
                    else "unknown"
                )

                has_collision = collision_data["has_collision"]
                previous_collision = self.collision_history.get(class_id, False)

                # Adjust velocity based on presence ratio
                presence_ratio = collision_data["presence_ratio"]
                adjusted_velocity = min(
                    127, int(mapping["velocity"] * (1 + presence_ratio * 1.5))
                )

                if has_collision:
                    # Object is colliding with edge
                    if not previous_collision:
                        # Collision just started - start new note
                        self.note_start_times[class_id] = timestamp
                        logger.debug(
                            f"🎹▶️ Starting continuous note for {class_name} (class {class_id})"
                        )

                    # Calculate continuous duration (how long has this been playing)
                    start_time = self.note_start_times.get(class_id, timestamp)
                    continuous_duration = timestamp - start_time

                    # Calculate fade-out velocity based on collision duration
                    # Fade out over time with configurable fade duration
                    fade_duration = 5.0  # Fade out over 5 seconds
                    fade_factor = max(0.0, 1.0 - (continuous_duration / fade_duration))

                    # Apply fade-out to velocity (but never go below 20% of original)
                    min_velocity_factor = 0.2  # Minimum 20% of original velocity
                    fade_factor = max(min_velocity_factor, fade_factor)

                    # Calculate final velocity with presence ratio and fade-out
                    base_velocity = min(
                        127, int(mapping["velocity"] * (1 + presence_ratio * 1.5))
                    )
                    faded_velocity = int(base_velocity * fade_factor)

                    # Create continuous event with fading velocity
                    event = MusicEvent(
                        note=mapping["note"],
                        velocity=faded_velocity,
                        duration=mapping["base_duration"]
                        + continuous_duration,  # Extend duration based on collision time
                        channel=0,  # All piano events on channel 0
                        timestamp=timestamp,
                        metadata={
                            "class_id": int(class_id),
                            "class_name": class_name,
                            "presence_ratio": float(presence_ratio),
                            "pixel_count": int(collision_data["pixel_count"]),
                            "instrument": "piano",
                            "scale_type": mapping["scale_type"],
                            "edge_collision": True,
                            "edges_touched": collision_data["edges_touched"],
                            "continuous_duration": continuous_duration,
                            "collision_state": "active",
                            "fade_factor": fade_factor,
                            "base_velocity": base_velocity,
                            "faded_velocity": faded_velocity,
                        },
                    )
                    events.append(event)
                    self.active_notes[class_id] = event

                else:
                    # Object is not colliding with edge
                    if previous_collision:
                        # Collision just ended - stop note
                        if class_id in self.note_start_times:
                            total_duration = timestamp - self.note_start_times[class_id]
                            logger.debug(
                                f"🎹⏹️ Stopping continuous note for {class_name} (class {class_id}) after {total_duration:.2f}s"
                            )
                            del self.note_start_times[class_id]

                        if class_id in self.active_notes:
                            del self.active_notes[class_id]

                # Update collision history
                self.collision_history[class_id] = has_collision

        # Clean up collision history for classes no longer present
        current_class_ids = set(current_collisions.keys())
        self.collision_history = {
            cid: state
            for cid, state in self.collision_history.items()
            if cid in current_class_ids
        }
        self.active_notes = {
            cid: note
            for cid, note in self.active_notes.items()
            if cid in current_class_ids
        }
        self.note_start_times = {
            cid: start_time
            for cid, start_time in self.note_start_times.items()
            if cid in current_class_ids
        }

        # Create frame result
        music_frame = MusicFrame(
            events=events,
            frame_id=frame_id,
            timestamp=timestamp,
            tempo=self.tempo,
            key_signature=self.key_signature,
            metadata={
                "musician_type": "ContinuousPianistMusician",
                "total_classes_detected": len(unique_classes),
                "total_events_generated": len(events),
                "active_continuous_notes": len(self.active_notes),
                "segmentation_shape": segmentation_data.shape,
                "instrument": "piano_only",
                "playback_mode": "continuous",
            },
        )

        self.frame_counter += 1

        # Log occasionally for debugging
        if self.frame_counter % 30 == 0:  # Every 30 frames
            logger.debug(
                f"🎹🔄 Generated {len(events)} continuous piano events for frame {frame_id}"
            )
            logger.debug(f"🎯 Active continuous notes: {len(self.active_notes)}")
            logger.debug(
                f"🎯 Collision classes: {[e.metadata['class_name'] for e in events]}"
            )

        return music_frame

    def stop_all_notes(self) -> None:
        """Stop all currently playing continuous notes."""
        logger.info("🎹⏹️ Stopping all continuous notes")
        self.active_notes.clear()
        self.collision_history.clear()
        self.note_start_times.clear()

class LSTMMusician(BaseMusician):
    """
    LSTM-based musician (Collision Trigger) that generates melodic sequences based on visual segmentation.
    Uses the trained LSTM_OnEssen model for musically coherent generation. More logical.
    """

    def __init__(self, tempo: int = 128, key_signature: str = "C_major", temperature: float = 1.0):

        super().__init__(tempo, key_signature)

        from Models.Music.LSTM_OnEssen.generator import MelodyGenerator

        self.generator = MelodyGenerator()
        self.temperature = temperature
        self.max_notes_per_trigger = 1

        self.last_seed_notes = [
            "67",
            "_",
            "67",
            "_",
            "67",
            "_",
            "_",
            "65",
            "64",
            "_",
            "64",
            "_",
            "64",
            "_",
            "_",
        ]

        # Collision state
        self.active_collision: bool = False
        self.current_collision_start: float = None
        self._symbol_buffer: list = []
        self._rt_generator = None

    def _resolve_important_class_ids(self, class_labels: List[str] = None) -> set:
        """Resolve the collision-relevant class IDs from model labels when available."""

        important_labels = {"person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"}

        if class_labels:
            normalized = {
                str(label).strip().lower()
                for label in class_labels
                if str(label).strip()
            }
            return {
                idx
                for idx, label in enumerate(class_labels)
                if str(label).strip().lower() in important_labels
            }

        return {11, 12, 13, 14, 15, 16, 17, 18}

    def _check_edge_collision(self, seg_map: np.ndarray, class_labels: List[str] = None) -> bool:
        """Return True if any important object class touches an image edge."""
        important_classes = self._resolve_important_class_ids(class_labels)

        mask = np.isin(seg_map, list(important_classes))
        if not np.any(mask):
            return False

        h, w = seg_map.shape
        return bool(
            (np.any(mask[0, :]))
            or (np.any(mask[h - 1, :]))
            or (np.any(mask[:, 0]))
            or (np.any(mask[:, w - 1]))
        )

    def generate_music(self, segmentation_data: np.ndarray, frame_id: int = 0, class_labels: List[str] = None, metadata: Dict[str, Any] = None) -> MusicFrame:
        """Generate music based on segmentation data and edge collisions."""

        timestamp = time.time()
        events = []
        step_dur = 60.0 / self.tempo / 3.5

        has_collision = self._check_edge_collision(segmentation_data, class_labels=class_labels)

        if has_collision:
            if not self.active_collision:
                logger.info(
                    f"▶️ COLLISION START - LSTM RT generator started (frame {frame_id})"
                )
                self.active_collision = True
                self.current_collision_start = timestamp
                # Spin up a persistent RT generator from the current seed.
                seed_str = " ".join(self.last_seed_notes)
                self._symbol_buffer = list(self.last_seed_notes)
                self._rt_generator = self.generator.generate_melody_RT(
                    seed=seed_str, num_steps=500, temperature=self.temperature
                )

            # Pull exactly one symbol — monophonic, one step per frame.
            try:
                symbol = next(self._rt_generator)

                # Accumulate for seed continuity on the next collision.
                self._symbol_buffer.append(symbol)
                if len(self._symbol_buffer) > 24:
                    self._symbol_buffer = self._symbol_buffer[-24:]

                if symbol.isdigit():
                    collision_duration = timestamp - self.current_collision_start
                    event = MusicEvent(
                        note=int(symbol),
                        velocity=np.random.randint(78, 96),
                        duration=step_dur,
                        channel=0,
                        timestamp=timestamp,
                        metadata={
                            "source": "lstm_onessen",
                            "symbol": symbol,
                            "collision": True,
                            "collision_duration": collision_duration,
                        },
                    )
                    events.append(event)
                # '_' (hold) and 'r' (rest) intentionally produce no event.

            except StopIteration:
                # Generator exhausted — restart from the accumulated buffer.
                logger.info("🔄 RT generator exhausted, restarting from buffer")
                seed_str = " ".join(self._symbol_buffer[-15:])
                self._rt_generator = self.generator.generate_melody_RT(
                    seed=seed_str, num_steps=500, temperature=self.temperature
                )

            except Exception as e:
                logger.error(f"LSTM Generation Error: {e}")
                events.append(MusicEvent(72, 85, step_dur, timestamp))

        else:
            if self.active_collision:
                logger.info(f"⏹️ Collision ended (frame {frame_id})")
                self.active_collision = False
                self.current_collision_start = None
                # Save accumulated symbols as the seed for the next collision.
                if self._symbol_buffer:
                    self.last_seed_notes = self._symbol_buffer[-15:]
                self.last_seed_notes.extend(["r", "_", "_"])
                self._rt_generator = None
                self._symbol_buffer = []

        music_frame = MusicFrame(
            events=events,
            frame_id=frame_id,
            timestamp=timestamp,
            tempo=self.tempo,
            key_signature=self.key_signature,
            metadata={
                "musician_type": "LSTMMusician",
                "collision_active": has_collision,
                "notes_generated": len(events),
                "last_seed_length": len(self.last_seed_notes),
            },
        )

        self.frame_counter += 1
        return music_frame


class Musician:
    """
    Main Musician class that provides a unified interface for different music generation models.

    This class acts as a factory and manager for different music generation models,
    allowing easy switching between models and unified result handling.
    """

    MUSICIAN_REGISTRY = {
        "test": {
            "class": TestMusician,
            "label": "Test Musician",
            "description": "Rule-based multi-instrument demo mapping (drums, bass, strings, etc.).",
        },
        "pianist": {
            "class": PianistTestMusician,
            "label": "Pianist (Rule-Based)",
            "description": "Rule-based musician that renders segmentation events as solo piano.",
        },
        "continuous_pianist": {
            "class": ContinuousPianistMusician,
            "label": "Continuous Pianist",
            "description": "Piano musician with sustained/continuous note playback.",
        },
        "lstm-onessen": {
            "class": LSTMMusician,
            "label": "LSTM (Essen Folk Song)",
            "description": "Neural LSTM model trained on the Essen folk song collection.",
        },
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
