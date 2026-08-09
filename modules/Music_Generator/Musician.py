"""
Modular Music Generation Framework for Real-Time Visual-to-Audio Mapping
========================================================================

This module provides an extensible framework for generating music based on visual data,
It supports various music generation strategies with easy integration for additional models.
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
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseMusician(ABC):
    """
    Abstract base class for all music generation models.
    This class defines the interface that all music generation models must implement,
    ensuring consistency and extensibility across different generation strategies.
    """

    NOTE_ON_TYPES = frozenset({"ROI_TOUCH", "NOTE_ON"})
    NOTE_OFF_TYPES = frozenset({"ROI_RELEASE", "NOTE_OFF"})

    def __init__(self, tempo: int = 120, key_signature: str = "C_major"):
        """
        Initialize the base musician.

        Args:
            tempo: Music tempo in BPM
            key_signature: Key signature for music generation
        """

        self.tempo = tempo
        self.key_signature = key_signature
        self.active_notes = {i: {} for i in range(16)}  # Initialize active notes for all 16 MIDI channels

        self.frame_counter = 0
        self.max_missing_frames = 4     # Number of frames to keep an object in memory after it disappears

    def __call__(self, results: List[Dict[str, Any]], frame_id: int = 0, state: Dict[str, Any] = None):
        return self.generate_music(results, frame_id, state)

    @staticmethod
    def _is_stale(state: Dict[str, Any], object_id: Any, max_missing_frames: int) -> bool:
        """Whether a held note's underlying object should be auto-released.

        Only meaningful for Detector states that track per-object 'missing_frames'
        (e.g. ROIEventsDetector's state = {"objects": {...}, ...}). Detector
        strategies without that concept - e.g. KeyEventsDetector, whose state is
        {"held": {...}} and relies on an explicit NOTE_OFF instead of frames going
        missing - simply never go stale here.
        """
        if not isinstance(state, dict) or "objects" not in state:
            return False
        return state["objects"].get(object_id, {}).get("missing_frames", 0) > max_missing_frames

    @abstractmethod
    def generate_music(self,
        results: List[Dict[str, Any]],
        frame_id: int = 0,
        state: Dict[str, Any] = None
    ):
        """
        Convenience method to call generate_music directly.

        Args:
            results: Detection result as a list of dictionaries containing scene events
            frame_id: Frame identifier for tracking
            state: Current state of the detection system

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
        Map object class to MIDI note, velocity, instrument, and channel."""

        base_class = obj_class.split("_")[0]

        mapping = {     # MIDI note, velocity, instrument, channel
            "car": (60, 100, 'piano', 0),
            "truck": (48, 120, 'piano', 0),
            "bus": (48, 90, 'piano', 0),
            "train": (55, 110, 'electric_piano', 1),
            "plane": (72, 100, 'electric_piano', 1),
            "bicycle": (64, 90, 'acoustic_guitar', 2),
            "person": (72, 110, 'acoustic_guitar', 2),
            "motorcycle": (70, 100, 'electric_guitar', 3),
            "traffic light": (67, 70, 'strings', 4),
            "traffic sign": (67, 70, 'strings', 4),
            "stop sign": (69, 80, 'strings', 4),
            # "road": (36, 50, 'drums', 9),
        }

        return mapping.get(base_class, None)
    
    def generate_music(self, results, frame_id, state):
        """
        Generate music based on the input scene data.
        """

        logger.info(f"🎵 Generating music for frame {frame_id}")

        self.frame_counter = frame_id

        scene_events = results
        music_events = []
        voice_id = 0

        for e in scene_events:

            obj_class = e["class"]
            mapped = self._map_classes(obj_class)
            if mapped is None:
                logger.warning(f"No mapping found for object class '{obj_class}'. Skipping event.")
                continue
            note, velocity, instrument, channel = mapped

            event = None
            if e["type"] in self.NOTE_ON_TYPES:
                event = "note_on"
                self.active_notes[channel][e["object_id"]] = {
                    "voice_id": voice_id,
                    "note": note,
                    "velocity": velocity,
                    "instrument": instrument,
                }
                voice_id += 1
            elif e["type"] in self.NOTE_OFF_TYPES:
                event = "note_off"
                self.active_notes[channel].pop(e["object_id"], None)
            else:
                continue

            music_events.append(
                MusicEvent(
                    event_type=event,
                    note=note,
                    channel=channel,
                    velocity=velocity if e["type"] in self.NOTE_ON_TYPES else 0,
                    instrument=instrument,
                    timestamp=self.frame_counter,
                    metadata=e
                )
            )
            logger.info(f"Mapped scene event: {e} to music event: 'type': {event}, 'note': {note}, 'velocity': {velocity if e['type'] in self.NOTE_ON_TYPES else 0}, 'instrument': '{instrument}'")

        for channel in self.active_notes:
            for object_id in list(self.active_notes[channel].keys()):
                if self._is_stale(state, object_id, self.max_missing_frames):
                    event = "note_off"
                    self.active_notes[channel].pop(object_id, None)
                    logger.info(f"Auto-released note for object_id {object_id} due to missing frames.")

        return MusicFrame(
            events=music_events,
            frame_id=frame_id,
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
        Map object class to MIDI note, velocity, instrument, and channel."""

        base_class = obj_class.split("_")[0]

        mapping = {
            "car": (60, 100, 'piano', 0),
            "truck": (48, 120, 'piano', 0),
            "bus": (42, 120, 'piano', 0),
            "train": (55, 110, 'piano', 0),
            "plane": (72, 100, 'piano', 0),
            "bicycle": (64, 90, 'electric_piano', 1),
            "person": (72, 110, 'electric_piano', 1),
            "motorcycle": (70, 92, 'electric_piano', 1),
            "traffic light": (80, 70, 'piano', 2),
            "traffic sign": (67, 70, 'piano', 2),
            "stop sign": (69, 80, 'piano', 2),
            # "road": (36, 50, 'piano', 0),
        }

        return mapping.get(base_class, None)

class LSTMMusician(BaseMusician):
    """
    LSTM-based musician that generates music using a trained LSTM model. This musician
    leverages a neural network to produce music based on learned patterns from training data.
    """

    AVAILABLE_INSTRUMENTS = (
        "piano", "electric_piano", 
        "strings", 
        "acoustic_guitar", "bass", "electric_guitar",
        "pad", "synth"
    )

    def __init__(self, tempo=120, key_signature="C_major", temperature=0.9, instrument="piano"):
        """
        Args:
            tempo: Music tempo in BPM
            key_signature: Key signature for music generation
            temperature: Sampling temperature for LSTM model
            instrument: Tone.js instrument used to play generated melodies
        """
        super().__init__(tempo, key_signature)
        self.temperature = temperature

        if instrument not in self.AVAILABLE_INSTRUMENTS:
            raise ValueError(f"Unsupported LSTM instrument: {instrument}. Supported instruments: {", ".join(self.AVAILABLE_INSTRUMENTS)}")
        self.instrument = instrument

        from Models.Music.LSTM_OnEssen.generator import MelodyGenerator
        self.generator = MelodyGenerator()
        self._rt_generator = None

        self.last_seed_notes = ["67", "_", "67", "_", 
                                "67", "_", "_", "65", 
                                "64", "_", "62", "_", 
                                "60", "_", "60", "_"]
        self._note_buffer = list(self.last_seed_notes)

        self.important_labels = [
            "car", "truck", "bus", "train", "plane",
            "bicycle", "motorcycle", "person",
            "traffic light", "traffic sign", "stop sign",
        ]

        logger.info(f"🎵 {self.__class__.__name__} initialized with tempo={tempo}, key_signature={key_signature}, temperature={temperature}")

    def generate_music(self, results, frame_id, state):
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
            
            event = None
            note = None
            velocity = 0

            if e["type"] in self.NOTE_ON_TYPES:
                event = "note_on"

                # Compute velocity based on the touching area size (larger area -> louder note).
                # Non-spatial events (e.g. a keyboard NOTE_ON) have no area - fall back to the
                # event's 'intensity' (0-1, set by whichever channel produced it) instead of
                # silently leaving velocity at 0.
                area = e.get("area/ROI", None)
                if area is not None:
                    # Scale area to velocity range (MinMax Scaler) Area:0.005-0.2, Velocity:16-128
                    scaled_area = (min(area, 0.2) - 0.005) / (0.2 - 0.005)
                    velocity = int(scaled_area * (127 - 17) + 17)
                    if area < 0.005:
                        logger.warning(f"Event with very small area ({area}). Skipping note generation for class '{obj_class}'.")
                        continue
                else:
                    intensity = max(0.0, min(1.0, e.get("intensity", 1.0)))
                    velocity = int(intensity * (127 - 17) + 17)

                # Generate new notes using the LSTM model
                self._rt_generator = self.generator.generate_melody_RT(
                    seed=" ".join(self.last_seed_notes),
                    length=200,
                    temperature=self.temperature
                )
                
                new_note = next(self._rt_generator)
                note = int(new_note)
                
                self.active_notes[0][e["object_id"]] = {
                    "voice_id": e["object_id"],
                    "note": note,
                    "velocity": velocity,
                    "instrument": self.instrument,
                }

                self._note_buffer.append(new_note)

            elif e["type"] in self.NOTE_OFF_TYPES:
                event = "note_off"
                
                # Find the related note for this object_id
                related_note = None
                if e["object_id"] in self.active_notes[0]:
                    related_note = self.active_notes[0][e["object_id"]]["note"]
                    self.active_notes[0].pop(e["object_id"], None)
                else:
                    logger.warning("No previous note found to turn off on release event.")
                    continue
                note = related_note

                self._note_buffer.extend(["r", "_"])

            else:
                self._note_buffer.append("_")
                continue

            music_events.append(
                MusicEvent(
                    event_type=event,
                    note=note,
                    channel=0,
                    velocity=velocity,
                    instrument=self.instrument,
                    timestamp=self.frame_counter,
                    metadata=e
                )
            )
            logger.info(f"Mapped scene event: {e} to music event: 'type': {event}, 'note': {note}, 'velocity': {velocity}, 'instrument': '{self.instrument}'")

            self.last_seed_notes = self._note_buffer[-16:]

        for object_id, note_info in list(self.active_notes[0].items()):
            if self._is_stale(state, object_id, self.max_missing_frames):
                music_events.append(
                    MusicEvent(
                        event_type="note_off",
                        note=note_info["note"],
                        channel=0,
                        velocity=0,
                        instrument=note_info["instrument"],
                        timestamp=self.frame_counter,
                        metadata={"object_id": object_id}
                    )
                )
                self.active_notes[0].pop(object_id, None)
                logger.info(f"Auto-released note for object_id {object_id} due to missing frames.")

        return MusicFrame(
            events=music_events,
            frame_id=frame_id,
            metadata={
                "scene_events": scene_events,
            }
        )

class LSTMOrchestralMusician(BaseMusician):
    """
    LSTM-based orchestral musician that generates music using a trained LSTM model. This musician 
    is similar to the LSTMMusician but is designed to produce orchestral sounds, allowing for a 
    richer and more diverse musical output.
    """

    def __init__(self, tempo=120, key_signature="C_major", temperature=0.9):
        """
        Args:
            tempo: Music tempo in BPM
            key_signature: Key signature for music generation
            temperature: Sampling temperature for LSTM model
        """
        super().__init__(tempo, key_signature)

        self.temperature = temperature

        from Models.Music.LSTM_OnEssen.generator import MelodyGenerator
        self.generator = MelodyGenerator()
        self._rt_generator = None

        self.last_seed_notes = {
            "piano": ["64", "_", "67", "_",
                      "65", "_", "65", "_",
                      "65", "_", "62", "_",
                      "62", "_", "64", "_"],
            "electric_piano": ["64", "_", "67", "_",
                               "65", "_", "65", "_",
                               "65", "_", "62", "_",
                               "62", "_", "64", "_"],
            "strings": ["67", "_", "67", "_", 
                        "65", "_", "_", "65",
                        "64", "_", "62", "_",
                        "62", "_", "64", "_"],
            "bass": ["48", "_", "48", "_",
                     "48", "_", "_", "50",
                     "50", "_", "52", "_",
                     "52", "_", "50", "_"],
        }
        # Note buffer to store generated notes for each instrument
        self._note_buffer = {
            instrument: list(self.last_seed_notes[instrument]) for instrument in self.last_seed_notes
        }

        logger.info(f"🎵 {self.__class__.__name__} initialized with tempo={tempo}, key_signature={key_signature}, temperature={temperature}")

    def _map_classes(self, obj_class):
        """
        Map object class to instrument and channel for orchestral sounds.
        """

        base_class = obj_class.split("_")[0]

        mapping = {
            "car": ('piano', 0),
            "truck": ('piano', 0),
            "bus": ('piano', 0),
            "train": ('electric_piano', 1),
            "plane": ('electric_piano', 1),
            "bicycle": ('bass', 2),
            "motorcycle": ('bass', 2),
            "person": ('bass', 2),
            "traffic light": ('strings', 3),
            "traffic sign": ('strings', 3),
            "stop sign": ('strings', 3),

            "typing": ('piano', 0),
            "scroll": ('strings', 3),
            "mousemove": ('pad', 5),
        }

        return mapping.get(base_class, None)

    def generate_music(self, results, frame_id, state):
        """
        Generate music based on the input scene data using the LSTM orchestral model.
        """

        logger.info(f"🎵 Generating orchestral music with LSTM for frame {frame_id}")

        self.frame_counter = frame_id

        scene_events = results
        music_events = []

        for e in scene_events:

            obj_class = e["class"]
            mapped = self._map_classes(obj_class)
            if mapped is None:
                logger.info(f"Skipping unimportant object class '{obj_class}'.")
                continue
            instrument, channel = mapped
            
            event = None
            note = None
            velocity = 0

            if e["type"] in self.NOTE_ON_TYPES:
                event = "note_on"

                area = e.get("area/ROI", None)
                if area is not None:
                    scaled_area = (min(area, 0.2) - 0.005) / (0.2 - 0.005)
                    velocity = int(scaled_area * (127 - 17) + 17)
                    if area < 0.005:
                        logger.warning(f"Event with very small area ({area}). Skipping note generation for class '{obj_class}'.")
                        continue
                else:
                    intensity = max(0.0, min(1.0, e.get("intensity", 1.0)))
                    velocity = int(intensity * (127 - 17) + 17)

                self._rt_generator = self.generator.generate_melody_RT(
                    seed=" ".join(self.last_seed_notes[instrument]),
                    length=200,
                    temperature=self.temperature
                )

                new_note = next(self._rt_generator)
                note = int(new_note)

                self.active_notes[channel][e["object_id"]] = {
                    "voice_id": e["object_id"],
                    "note": note,
                    "velocity": velocity,
                    "instrument": instrument,
                }

                self._note_buffer[instrument].append(new_note)

            elif e["type"] in self.NOTE_OFF_TYPES:
                event = "note_off"

                related_note = None
                if e["object_id"] in self.active_notes[channel]:
                    related_note = self.active_notes[channel][e["object_id"]]["note"]
                    self.active_notes[channel].pop(e["object_id"], None)
                else:
                    logger.warning("No previous note found to turn off on release event.")
                    continue
                note = related_note

                self._note_buffer[instrument].extend(["r", "_"])

            else:
                self._note_buffer[instrument].append("_")
                continue

            music_events.append(
                MusicEvent(
                    event_type=event,
                    note=note,
                    channel=channel,
                    velocity=velocity,
                    instrument=instrument,
                    timestamp=self.frame_counter,
                    metadata=e
                )
            )
            logger.info(f"Mapped scene event: {e} to music event: 'type': {event}, 'note': {note}, 'velocity': {velocity}, 'instrument': '{instrument}'")

            self.last_seed_notes[instrument] = self._note_buffer[instrument][-16:]

        for channel in self.active_notes:
            for object_id, note_info in list(self.active_notes[channel].items()):
                if self._is_stale(state, object_id, self.max_missing_frames):
                    music_events.append(
                        MusicEvent(
                            event_type="note_off",
                            note=note_info["note"],
                            channel=channel,
                            velocity=0,
                            instrument=note_info["instrument"],
                            timestamp=self.frame_counter,
                            metadata={"object_id": object_id}
                        )
                    )
                    self.active_notes[channel].pop(object_id, None)
                    logger.info(f"Auto-released note for object_id {object_id} due to missing frames.")

        return MusicFrame(
            events=music_events,
            frame_id=frame_id,
            metadata={
                "scene_events": scene_events,
            }
        )

class TestDrummerMusician(BaseMusician):
    """
    Test musician that generates a simple drum pattern for testing purposes.
    This musician is designed to produce a basic drum beat, allowing for quick testing
    of the music generation framework without relying on complex models or external data.
    """

    def __init__(self, tempo=120, key_signature="C_major"):
        """
        Args:
            tempo: Music tempo in BPM
            key_signature: Key signature for music generation
        """
        super().__init__(tempo, key_signature)

        logger.info(f"🎵 {self.__class__.__name__} initialized with tempo={tempo}, key_signature={key_signature}")

    def generate_music(self, results, frame_id, state):
        """
        Generate a simple drum pattern based on the input scene data.
        """

        logger.info(f"🎵 Generating test drum pattern for frame {frame_id}")

        self.frame_counter = frame_id

        # Simple drum pattern: kick on even frames, snare on odd frames
        music_events = []
        if frame_id % 4 == 0:
            music_events.append(
                MusicEvent(
                    event_type="note_on",
                    note=36,  # Kick drum MIDI note
                    channel=9,
                    velocity=100,
                    instrument="drums",
                    timestamp=self.frame_counter,
                )
            )
            logger.info(f"Generated kick drum event for frame {frame_id}")
        else:
            music_events.append(
                MusicEvent(
                    event_type="note_on",
                    note=38,  # Snare drum MIDI note
                    channel=9,
                    velocity=100,
                    instrument="drums",
                    timestamp=self.frame_counter,
                )
            )
            logger.info(f"Generated snare drum event for frame {frame_id}")
        return MusicFrame(
            events=music_events,
            frame_id=frame_id,
            metadata={
                "scene_events": results,
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
        "lstm-onessen-orchestral": {
            "class": LSTMOrchestralMusician,
            "label": "LSTM (Orchestral)",
            "description": "Just like the LSTM musician, but with orchestral instruments.",
        },
        "test-drummer": {
            "class": TestDrummerMusician,
            "label": "Test Drummer",
            "description": "Simple test musician that generates a basic drum pattern.",
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

        self.generated_melody = []

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

    def save_generated_melody(self):
        """
        Save the generated melody (List of MusicFrames) to a MIDI file.
        Use the current time for file name for easy access.

        Args:
        """

        pass

    def __call__(self, 
                 results,
                 frame_id: int = 0,
                 state: Dict[str, Any] = None
                ) -> MusicFrame:
        """
        Generate music based on segmentation data.

        Args:
            results: Detection results
            frame_id: Frame identifier for tracking

        Returns:
            MusicFrame containing generated music events
        """

        generated_frame =  self.musician(results, frame_id, state)
        self.generated_melody.append(generated_frame)

        # if frame_id % 200 == 0:
        #     self.save_generated_melody()  
        #     logger.info(f"✅ Saved generated melody at frame {frame_id}")

        return generated_frame
