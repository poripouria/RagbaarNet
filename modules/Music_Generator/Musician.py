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
import cv2
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

    def intersects_bbox(self, bbox, return_edges=False):

        x1, y1, x2, y2 = map(int, bbox["bbox"])

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(self.boundary_mask.shape[1], x2)
        y2 = min(self.boundary_mask.shape[0], y2)

        bbox_mask = np.zeros_like(self.boundary_mask, dtype=bool)
        bbox_mask[y1:y2, x1:x2] = True

        touching = np.logical_and(
            bbox_mask,
            self.boundary_mask
        ).any()

        if not return_edges:
            return touching

        edge_names = ["top", "right", "bottom", "left"]

        edges = []

        for name, edge_mask in zip(edge_names, self.edge_masks):

            if np.logical_and(bbox_mask, edge_mask).any():
                edges.append(name)

        return {
            "touching": touching,
            "edges": edges
        }

    def intersects_mask(self, mask, return_edges=False):

        touching = np.logical_and(
            mask,
            self.boundary_mask
        ).any()

        if not return_edges:
            return touching

        edge_names = ["top", "right", "bottom", "left"]
        edges = []

        for name, edge_mask in zip(edge_names, self.edge_masks):

            if np.logical_and(mask, edge_mask).any():
                edges.append(name)

        return {
            "touching": touching,
            "edges": edges
        }

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

        # state: keeps track of objects
        self.state = {
            "objects": {},          # object_id -> object info
            "next_object_id": 0,    
            "active_notes": {},     # note -> last frame it was active
        }
        self.max_missing_frames = 4  # Number of frames to keep an object in memory after it disappears

        self.roi = None  # Will be set per frame if provided
        self.prev_roi_payload = None  # To track changes in ROI between frames

    def __call__(self,
        input: SegmentationResult,
        frame_id: int = 0,
        roi: Dict[str, Any] = None
    ):
        
        if not isinstance(input, SegmentationResult):
            raise ValueError("Input must be a SegmentationResult instance")

        return self.generate_music(input, frame_id, roi)

    def _set_roi(self, roi_payload):
        
        if not roi_payload:
            return
        
        if self.prev_roi_payload != roi_payload:
            self.prev_roi_payload = roi_payload
            self.roi = ROI(corners=roi_payload.get("corners", []), 
                           controls=roi_payload.get("controls", []))
            
            logger.info(f"ROI updated for frame {self.frame_counter}")

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

        if bounding_boxes is None and masks is None:
            logger.warning("No bounding boxes or masks provided for scene event detection.")
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

            track = self.state["objects"].get(obj_id, {})
            prev = track.get("touching", False)

            if touching and not prev:
                events.append({
                    "type": "ROI_TOUCH",
                    "object_id": obj_id,
                    "class": obj_class,
                    "edges": edges
                })
                self.state["objects"][obj_id]["touching"] = True

            elif not touching and prev:
                events.append({
                    "type": "ROI_RELEASE",
                    "object_id": obj_id,
                    "class": obj_class,
                    "edges": edges
                })
                self.state["objects"][obj_id]["touching"] = False
        
        else:
            logger.warning("No bounding boxes or masks provided for scene event detection.")

        logger.info(f"Detected {len(events)} scene events")

        return events

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

    def extract_features(self, segmentation_data: SegmentationResult) -> Dict[str, Any]:
        """
        Default feature extractor (can be overridden).

        Args:
            segmentation_data: Segmentation result instance
        """

        return segmentation_data.metadata or {}

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
        """
        super().__init__(tempo, key_signature)

        logger.info(f"🎵 {self.__class__.__name__} initialized with tempo={tempo}, key_signature={key_signature}")

    def _map_classes(self, obj_class):
        """
        Map object class to MIDI note, velocity, and instrument."""

        base_class = obj_class.split("_")[0]

        mapping = {
            "car": (60, 100, 'piano'),
            "truck": (48, 80, 'electric_piano'),
            "bus": (48, 80, 'electric_piano'),
            "bicycle": (64, 90, 'acoustic_guitar'),
            "person": (72, 110, 'acoustic_guitar'),
            "motorcycle": (70, 100, 'electric_guitar'),
            "road": (36, 50, 'drums'),
            "traffic light": (67, 70, 'strings'),
            "traffic sign": (67, 70, 'strings'),
            "stop sign": (69, 80, 'strings'),
        }

        return mapping.get(base_class, None)
    
    def generate_music(self, result, frame_id, roi):
        """
        Generate music based on the input scene data.
        """

        logger.info(f"🎵 Generating music for frame {frame_id}")

        self.frame_counter = frame_id
        self._set_roi(roi)

        scene_events = self.detect_scene_events(result.bounding_boxes, result.masks)
        music_events = []

        for e in scene_events:

            obj_class = e["class"]
            mapped = self._map_classes(obj_class)
            if mapped is None:
                logger.warning(f"No mapping found for object class '{obj_class}'. Skipping event.")
                continue
            note, velocity, instrument = mapped

            event = None
            if e["type"] == "ROI_TOUCH":
                event = "note_on"
                self.state["active_notes"][e["object_id"]] = {
                    "voice_id": e["object_id"],
                    "note": note,
                    "velocity": velocity,
                    "instrument": instrument,
                }
            elif e["type"] == "ROI_RELEASE":
                event = "note_off"
                self.state["active_notes"].pop(e["object_id"], None)
            else:
                continue  # Skip event
                
            music_events.append(
                MusicEvent(
                    event_type=event,
                    note=note,
                    channel=0,
                    velocity=velocity if e["type"] == "ROI_TOUCH" else 0,
                    instrument=instrument,
                    timestamp=self.frame_counter,
                    metadata=e
                )
            )
            
            logger.info(f"Mapped scene event: {e} to music event: 'type': {event}, 'note': {note}, 'velocity': {velocity if e['type'] == 'ROI_TOUCH' else 0}, 'instrument': '{instrument}'")

        for object_id, note_info in list(self.state["active_notes"].items()):

            if self.state["objects"].get(object_id, {}).get("missing_frames", 0) > self.max_missing_frames:

                music_events.append(
                    MusicEvent(
                        event_type="note_off",
                        note=note_info["note"],
                        channel=0,
                        velocity=0,
                        instrument=note_info["instrument"],
                        timestamp=self.frame_counter,
                        metadata={"object_id": object_id, "class": self.state["objects"][object_id]["class_name"]}
                    )
                )

                self.state["active_notes"].pop(object_id, None)
                
                logger.info(f"Auto-released note for object_id {object_id} due to missing frames.")

        return MusicFrame(
            events=music_events,
            frame_id=frame_id,
            tempo=self.tempo,
            key_signature=self.key_signature,
            metadata={
                "scene_events": scene_events,
                "active_objects": list(self.state["objects"].values()),
                "extra": result.metadata or {}
            }
        )

class ContinuousPianistMusician(RuleBasedMusician):
    """
    Continuous Pianist musician that generates sustained piano notes based on scene events.
    This musician is designed to produce continuous and overlapping piano notes, allowing for
    a more fluid and expressive musical output in response to visual stimuli.
    """

    def __init__(self, tempo=120, key_signature="C_major"):
        """
        Args:
            tempo: Music tempo in BPM
            key_signature: Key signature for music generation
        """
        super().__init__(tempo, key_signature)

        logger.info(f"🎵 {self.__class__.__name__} initialized with tempo={tempo}, key_signature={key_signature}")

    def _map_classes(self, obj_class):
        """
        Map object class to MIDI note, velocity, and instrument."""

        base_class = obj_class.split("_")[0]

        mapping = {
            "car": (60, 110, 'piano'),
            "truck": (48, 80, 'piano'),
            "bus": (42, 80, 'piano'),
            "bicycle": (64, 90, 'piano'),
            "person": (72, 110, 'piano'),
            "motorcycle": (70, 100, 'piano'),
            "road": (36, 50, 'piano'),
            "traffic light": (80, 70, 'piano'),
            "traffic sign": (67, 70, 'piano'),
            "stop sign": (69, 80, 'piano'),
        }

        return mapping.get(base_class, None)

class LSTMMusician(BaseMusician):
    """
    LSTM-based musician that generates music using a trained LSTM model. This musician
    leverages a neural network to produce music based on learned patterns from training data.
    """

    def __init__(self, tempo=120, key_signature="C_major", temperature=1.0):
        """
        Args:
            tempo: Music tempo in BPM
            key_signature: Key signature for music generation
            temperature: Sampling temperature for LSTM model
        """
        super().__init__(tempo, key_signature)

        from Models.Music.LSTM_OnEssen.generator import MelodyGenerator
        self.generator = MelodyGenerator()
        self.temperature = temperature

        self.last_seed_notes = ["67", "_", "67", "_", 
                                "67", "_", "_", "65", 
                                "64", "_", "62", "_", 
                                "60", "_", "60", "_"]
        self._note_buffer = list(self.last_seed_notes)
        self._rt_generator = None

        self.important_labels = [
            "car", "truck", "bus", 
            "bicycle", "person", "motorcycle",
            "traffic light", "traffic sign", "stop sign"
        ]

        logger.info(f"🎵 {self.__class__.__name__} initialized with tempo={tempo}, key_signature={key_signature}, temperature={temperature}")

    def generate_music(self, result, frame_id, roi):
        """
        Generate music based on the input scene data using the LSTM model.
        """

        logger.info(f"🎵 Generating music with LSTM for frame {frame_id}")

        self.frame_counter = frame_id
        self._set_roi(roi)

        scene_events = self.detect_scene_events(result.bounding_boxes, result.masks)
        music_events = []

        for e in scene_events:

            obj_class = e["class"]
            if obj_class.split("_")[0] not in self.important_labels:
                logger.info(f"Skipping unimportant object class '{obj_class}'.")
                continue

            if e["type"] == "ROI_TOUCH":

                # Generate new notes using the LSTM model
                self._rt_generator = self.generator.generate_melody_RT(
                    seed=" ".join(self.last_seed_notes),
                    num_steps=400,
                    temperature=self.temperature
                )

                # Skip non-digit notes ('_' (hold) and 'r' (rest)) until we get a valid note
                while True:
                    new_note = next(self._rt_generator)
                    if new_note.isdigit():
                        break

                music_events.append(
                    MusicEvent(
                        event_type="note_on",
                        note=int(new_note),
                        channel=0,
                        velocity=100 if e["type"] == "ROI_TOUCH" else 0,
                        instrument="piano",
                        timestamp=self.frame_counter,
                        metadata=e
                    )
                )

                self.state["active_notes"][e["object_id"]] = {
                    "voice_id": e["object_id"],
                    "note": int(new_note),
                    "velocity": 100,
                    "instrument": "piano"
                }

                self._note_buffer.append(new_note)

                logger.info(f"Mapped scene event: {e} to music event: 'type': {"note_on"}, 'note': {new_note}, 'velocity': {100 if e["type"] == "ROI_TOUCH" else 0}, 'instrument': 'piano'")

            elif e["type"] == "ROI_RELEASE":
                
                # Find the related note for this object_id
                last_note = None
                if e["object_id"] in self.state["active_notes"]:
                    last_note = self.state["active_notes"][e["object_id"]]["note"]
                    self.state["active_notes"].pop(e["object_id"], None)

                if last_note is not None:
                    music_events.append(
                        MusicEvent(
                            event_type="note_off",
                            note=int(last_note),
                            channel=0,
                            velocity=0,
                            instrument="piano",
                            timestamp=self.frame_counter,
                            metadata=e
                        )
                    )

                else:
                    logger.warning("No previous note found to turn off on ROI_RELEASE event.")

                self._note_buffer.extend(["r", "_"])

                logger.info(f"Mapped scene event: {e} to music event: 'type': {"note_off"}, 'note': {last_note}, 'velocity': 0, 'instrument': 'piano'")

            else:
                self._note_buffer.append("_")
                continue

            self.last_seed_notes = self._note_buffer[-16:]

        for object_id, note_info in list(self.state["active_notes"].items()):

            if self.state["objects"].get(object_id, {}).get("missing_frames", 0) > self.max_missing_frames:

                music_events.append(
                    MusicEvent(
                        event_type="note_off",
                        note=note_info["note"],
                        channel=0,
                        velocity=0,
                        instrument=note_info["instrument"],
                        timestamp=self.frame_counter,
                        metadata={"object_id": object_id, "class": self.state["objects"][object_id]["class_name"]}
                    )
                )

                self.state["active_notes"].pop(object_id, None)

                logger.info(f"Auto-released note for object_id {object_id} due to missing frames.")

        return MusicFrame(
            events=music_events,
            frame_id=frame_id,
            tempo=self.tempo,
            key_signature=self.key_signature,
            metadata={
                "scene_events": scene_events,
                "active_objects": list(self.state["objects"].values()),
                "extra": result.metadata or {}
            }
        )


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

    def __init__(self, musician_type: str = "lstm-onessen", tempo: int = 120, key_signature: str = "C_major"):
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

        entry = self.MUSICIAN_REGISTRY.get(self.musician_type)
        if entry is None:
            available = ", ".join(sorted(self.MUSICIAN_REGISTRY.keys()))
            raise ValueError(f"Unsupported musician type: {musician_type}. Supported types: {available}")
        
        self.musician = entry["class"](tempo, key_signature)

        logger.info(f"🎵 Musician initialized: {musician_type}")

    def switch_musician(self, musician_type: str, tempo: int = None, key_signature: str = None) -> None:
        """
        Switch to a different music generation model.

        Args:
            musician_type: New musician type
            tempo: New tempo (keeps current if None)
            key_signature: New key signature (keeps current if None)
        """

        self.musician_type = musician_type.lower()
        self.tempo = tempo
        self.key_signature = key_signature

        entry = self.MUSICIAN_REGISTRY.get(self.musician_type)
        if entry is None:
            available = ", ".join(sorted(self.MUSICIAN_REGISTRY.keys()))
            raise ValueError(f"Unsupported musician type: {musician_type}. Supported types: {available}")
        
        self.musician = entry["class"](tempo, key_signature)

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
