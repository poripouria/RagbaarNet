"""
Modular Music Generation Framework for Real-Time Visual-to-Audio Mapping
========================================================================

This module provides an extensible framework for generating music based on visual data,
particularly segmentation maps from computer vision models. It supports various music
generation strategies with easy integration for additional models.
"""

import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.logging_setup import setup_logging

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
        self.active_notes = {}          # note -> last frame it was active

        self.frame_counter = 0
        self.max_missing_frames = 4     # Number of frames to keep an object in memory after it disappears

    def __call__(self, results: List[Dict[str, Any]], frame_id: int = 0):
        return self.generate_music(results, frame_id)

    @abstractmethod
    def generate_music(self,
        results: List[Dict[str, Any]],
        frame_id: int = 0
    ):
        """
        Convenience method to call generate_music directly.

        Args:
            results: Detection result as a list of dictionaries containing scene events
            frame_id: Frame identifier for tracking

        Returns:
            MusicFrame containing generated music events
        """
        pass

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
    
    def generate_music(self, results, frame_id):
        """
        Generate music based on the input scene data.
        """

        logger.info(f"🎵 Generating music for frame {frame_id}")

        self.frame_counter = frame_id

        scene_events = results
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
                self.active_notes[e["object_id"]] = {
                    "voice_id": e["object_id"],
                    "note": note,
                    "velocity": velocity,
                    "instrument": instrument,
                }
            elif e["type"] == "ROI_RELEASE":
                event = "note_off"
                self.active_notes.pop(e["object_id"], None)
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

        for object_id, note_info in list(self.active_notes.items()):

            if results.state["objects"].get(object_id, {}).get("missing_frames", 0) > self.max_missing_frames:

                music_events.append(
                    MusicEvent(
                        event_type="note_off",
                        note=note_info["note"],
                        channel=0,
                        velocity=0,
                        instrument=note_info["instrument"],
                        timestamp=self.frame_counter,
                        metadata={"object_id": object_id, "class": results.state["objects"][object_id]["class_name"]}
                    )
                )

                self.active_notes.pop(object_id, None)
                
                logger.info(f"Auto-released note for object_id {object_id} due to missing frames.")

        return MusicFrame(
            events=music_events,
            frame_id=frame_id,
            tempo=self.tempo,
            key_signature=self.key_signature,
            metadata={
                "scene_events": scene_events,
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

    AVAILABLE_INSTRUMENTS = (
        "piano", "electric_piano", "strings", "bass", "electric_guitar",
        "acoustic_guitar", "pad", "synth"
    )

    def __init__(self, tempo=120, key_signature="C_major", temperature=1.0, instrument="piano"):
        """
        Args:
            tempo: Music tempo in BPM
            key_signature: Key signature for music generation
            temperature: Sampling temperature for LSTM model
            instrument: Tone.js instrument used to play generated melodies
        """
        super().__init__(tempo, key_signature)

        if instrument not in self.AVAILABLE_INSTRUMENTS:
            available = ", ".join(self.AVAILABLE_INSTRUMENTS)
            raise ValueError(f"Unsupported LSTM instrument: {instrument}. Supported instruments: {available}")
        self.instrument = instrument

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

    def generate_music(self, results, frame_id):
        """
        Generate music based on the input scene data using the LSTM model.
        """

        logger.info(f"🎵 Generating music with LSTM for frame {frame_id}")

        self.frame_counter = frame_id

        scene_events = results
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
                        instrument=self.instrument,
                        timestamp=self.frame_counter,
                        metadata=e
                    )
                )

                self.active_notes[e["object_id"]] = {
                    "voice_id": e["object_id"],
                    "note": int(new_note),
                    "velocity": 100,
                    "instrument": self.instrument
                }

                self._note_buffer.append(new_note)

                logger.info(f"Mapped scene event: {e} to music event: 'type': {"note_on"}, 'note': {new_note}, 'velocity': {100 if e["type"] == "ROI_TOUCH" else 0}, 'instrument': '{self.instrument}'")

            elif e["type"] == "ROI_RELEASE":
                
                # Find the related note for this object_id
                last_note = None
                if e["object_id"] in self.active_notes:
                    last_note = self.active_notes[e["object_id"]]["note"]
                    self.active_notes.pop(e["object_id"], None)

                if last_note is not None:
                    music_events.append(
                        MusicEvent(
                            event_type="note_off",
                            note=int(last_note),
                            channel=0,
                            velocity=0,
                            instrument=self.instrument,
                            timestamp=self.frame_counter,
                            metadata=e
                        )
                    )

                else:
                    logger.warning("No previous note found to turn off on ROI_RELEASE event.")

                self._note_buffer.extend(["r", "_"])

                logger.info(f"Mapped scene event: {e} to music event: 'type': {"note_off"}, 'note': {last_note}, 'velocity': 0, 'instrument': '{self.instrument}'")

            else:
                self._note_buffer.append("_")
                continue

            self.last_seed_notes = self._note_buffer[-16:]

        for object_id, note_info in list(self.active_notes.items()):

            if results.state["objects"].get(object_id, {}).get("missing_frames", 0) > self.max_missing_frames:

                music_events.append(
                    MusicEvent(
                        event_type="note_off",
                        note=note_info["note"],
                        channel=0,
                        velocity=0,
                        instrument=note_info["instrument"],
                        timestamp=self.frame_counter,
                        metadata={"object_id": object_id, "class": results.state["objects"][object_id]["class_name"]}
                    )
                )

                self.active_notes.pop(object_id, None)

                logger.info(f"Auto-released note for object_id {object_id} due to missing frames.")

        return MusicFrame(
            events=music_events,
            frame_id=frame_id,
            tempo=self.tempo,
            key_signature=self.key_signature,
            metadata={
                "scene_events": scene_events,
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

    def __init__(self, musician_type: str = "lstm-onessen", tempo: int = 120, key_signature: str = "C_major", instrument: str = "piano"):
        """
        Initialize the main Musician.

        Args:
            musician_type: Type of musician, see Musician.MUSICIAN_REGISTRY for supported values.
            tempo: Music tempo in BPM
            key_signature: Key signature for music generation
            instrument: Instrument used by the LSTM musician
        """

        self.musician_type = musician_type.lower()
        self.tempo = tempo
        self.key_signature = key_signature
        self.instrument = instrument

        entry = self.MUSICIAN_REGISTRY.get(self.musician_type)
        if entry is None:
            available = ", ".join(sorted(self.MUSICIAN_REGISTRY.keys()))
            raise ValueError(f"Unsupported musician type: {musician_type}. Supported types: {available}")
        
        self.musician = self._create_musician(entry)

        logger.info(f"🎵 Musician initialized: {musician_type}")

    def _create_musician(self, entry):
        if entry["class"] is LSTMMusician:
            return entry["class"](self.tempo, self.key_signature, instrument=self.instrument)
        return entry["class"](self.tempo, self.key_signature)

    def switch_musician(
        self,
        musician_type: str,
        tempo: Optional[int] = None,
        key_signature: Optional[str] = None,
        instrument: Optional[str] = None
    ) -> None:
        """
        Switch to a different music generation model.

        Args:
            musician_type: New musician type
            tempo: New tempo (keeps current if None)
            key_signature: New key signature (keeps current if None)
            instrument: LSTM instrument (keeps current if None)
        """

        self.musician_type = musician_type.lower()
        self.tempo = self.tempo if tempo is None else tempo
        self.key_signature = self.key_signature if key_signature is None else key_signature
        self.instrument = self.instrument if instrument is None else instrument

        entry = self.MUSICIAN_REGISTRY.get(self.musician_type)
        if entry is None:
            available = ", ".join(sorted(self.MUSICIAN_REGISTRY.keys()))
            raise ValueError(f"Unsupported musician type: {musician_type}. Supported types: {available}")
        
        self.musician = self._create_musician(entry)

        logger.info(f"🔄 Switched to {musician_type} musician")

    def set_tempo(self, tempo: int) -> None:
        self.tempo = tempo
        self.musician.tempo = tempo

    def set_instrument(self, instrument: str) -> None:
        if instrument not in LSTMMusician.AVAILABLE_INSTRUMENTS:
            available = ", ".join(LSTMMusician.AVAILABLE_INSTRUMENTS)
            raise ValueError(f"Unsupported LSTM instrument: {instrument}. Supported instruments: {available}")
        self.instrument = instrument
        if isinstance(self.musician, LSTMMusician):
            self.musician.instrument = instrument

    @classmethod
    def list_available_musicians(cls) -> List[dict]:
        """
        Return metadata for every musician type that can be selected/switched to.

        Used by the Platform UI to populate the music settings picker without
        duplicating the list of supported types.
        """

        return [
            {
                "id": musician_id,
                "label": info["label"],
                "description": info["description"],
                "instruments": list(LSTMMusician.AVAILABLE_INSTRUMENTS) if musician_id == "lstm-onessen" else []
            }
            for musician_id, info in cls.MUSICIAN_REGISTRY.items()
        ]

    def __call__(self, 
                 results,
                 frame_id: int = 0,
                 ) -> MusicFrame:
        """
        Generate music based on segmentation data.

        Args:
            results: Detection results
            frame_id: Frame identifier for tracking

        Returns:
            MusicFrame containing generated music events
        """

        return self.musician(results, frame_id)
