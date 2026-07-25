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
        self.active_notes = {i: {} for i in range(16)}  # Initialize active notes for all 16 MIDI channels

        self.frame_counter = 0
        self.max_missing_frames = 4     # Number of frames to keep an object in memory after it disappears

    def __call__(self, results: List[Dict[str, Any]], frame_id: int = 0, state: Dict[str, Any] = None):
        return self.generate_music(results, frame_id, state)

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
            if e["type"] == "ROI_TOUCH":
                event = "note_on"
                self.active_notes[channel][e["object_id"]] = {
                    "voice_id": voice_id,
                    "note": note,
                    "velocity": velocity,
                    "instrument": instrument,
                }
                voice_id += 1
            elif e["type"] == "ROI_RELEASE":
                event = "note_off"
                self.active_notes[channel].pop(e["object_id"], None)
            else:
                if state["objects"].get(e["object_id"], {}).get("missing_frames", 0) > self.max_missing_frames:
                    event = "note_off"
                    self.active_notes[channel].pop(e["object_id"], None)
                    logger.info(f"Auto-released note for object_id {e['object_id']} due to missing frames.")
                else:
                    continue

            music_events.append(
                MusicEvent(
                    event_type=event,
                    note=note,
                    channel=channel,
                    velocity=velocity if e["type"] == "ROI_TOUCH" else 0,
                    instrument=instrument,
                    timestamp=self.frame_counter,
                    metadata=e
                )
            )
            logger.info(f"Mapped scene event: {e} to music event: 'type': {event}, 'note': {note}, 'velocity': {velocity if e['type'] == 'ROI_TOUCH' else 0}, 'instrument': '{instrument}'")

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
            "traffic light", "traffic sign", "stop sign"
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

            if e["type"] == "ROI_TOUCH":
                event = "note_on"

                # Compute velocity based on the touching area size (larger area -> louder note)
                area = e.get("area/ROI", None)
                if area is not None: 
                    # Scale area to velocity range (MinMax Scaler) Area:0.01-0.4, Velocity:32-128
                    scaled_area = (area - 0.01) / (0.4 - 0.01)
                    velocity = int(scaled_area * (127 - 31) + 31)
                if area < 0.01:
                    logger.warning(f"Event with very small area ({area}). Skipping note generation for class '{obj_class}'.")
                    continue

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

            elif e["type"] == "ROI_RELEASE":
                event = "note_off"
                
                # Find the related note for this object_id
                related_note = None
                if e["object_id"] in self.active_notes[0]:
                    related_note = self.active_notes[0][e["object_id"]]["note"]
                    self.active_notes[0].pop(e["object_id"], None)
                else:
                    logger.warning("No previous note found to turn off on ROI_RELEASE event.")
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
            if state["objects"].get(object_id, {}).get("missing_frames", 0) > self.max_missing_frames:
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
            tempo=self.tempo,
            key_signature=self.key_signature,
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

            if e["type"] == "ROI_TOUCH":
                event = "note_on"

                area = e.get("area/ROI", None)
                if area is not None: 
                    scaled_area = (area - 0.01) / (0.4 - 0.01)
                    velocity = int(scaled_area * (127 - 31) + 31)
                if area < 0.01:
                    logger.warning(f"Event with very small area ({area}). Skipping note generation for class '{obj_class}'.")
                    continue

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

            elif e["type"] == "ROI_RELEASE":
                event = "note_off"

                related_note = None
                if e["object_id"] in self.active_notes[channel]:
                    related_note = self.active_notes[channel][e["object_id"]]["note"]
                    self.active_notes[channel].pop(e["object_id"], None)
                else:
                    logger.warning("No previous note found to turn off on ROI_RELEASE event.")
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
                if state["objects"].get(object_id, {}).get("missing_frames", 0) > self.max_missing_frames:
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
        "lstm-onessen-orchestral": {
            "class": LSTMOrchestralMusician,
            "label": "LSTM (Orchestral)",
            "description": "Just like the LSTM musician, but with orchestral instruments.",
        }
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

    def save_generated_melody(self, step_duration: float = 0.25) -> None:
        """
        Save the generated melody (List of MusicFrames) to a MIDI file.
        Use the current time for file name for easy access.

        Args:
            step_duration: Duration of each time step in quarter length.
        """

        # import music21 as m21
        # import time
        # from collections import defaultdict, deque
    
        # if not self.generated_melody:
        #     logger.warning("No generated melody to save.")
        #     return
    
        # # ── Step 1: Flatten all events across frames with absolute timestamps ───
        # flat_events = []  # (timestamp, MusicEvent, frame_tempo)
        # for frame in self.generated_melody:
        #     frame_ts = frame.timestamp
        #     frame_tempo = frame.tempo
        #     for event in frame.events:
        #         ts = event.timestamp if event.timestamp else frame_ts
        #         flat_events.append((ts, event, frame_tempo))
    
        # if not flat_events:
        #     logger.warning("No events found across frames — nothing to save.")
        #     return
    
        # # Sort chronologically — required for correct note_on/note_off pairing
        # flat_events.sort(key=lambda x: x[0])
    
        # t0 = flat_events[0][0]
        # base_tempo = flat_events[0][2]  # BPM used to convert seconds -> quarterLength
    
        # def seconds_to_quarter_length(seconds: float) -> float:
        #     """Convert elapsed real-time seconds to a music21 quarterLength."""
        #     return seconds * (base_tempo / 60.0)
    
        # # ── Step 2: One Part per MIDI channel ────────────────────────────────────
        # parts: dict = {}
    
        # def get_part(channel: int, instrument_name: str = None) -> m21.stream.Part:
        #     if channel not in parts:
        #         part = m21.stream.Part(id=f"channel_{channel}")
        #         part.partName = f"Channel {channel}"
        #         try:
        #             instr = m21.instrument.fromString(instrument_name) if instrument_name else m21.instrument.Piano()
        #         except Exception:
        #             instr = m21.instrument.Piano()
        #         part.insert(0, instr)
        #         parts[channel] = part
        #     return parts[channel]
    
        # # ── Step 3: Match note_on -> note_off per (channel, note) via FIFO ───────
        # pending_note_on = defaultdict(deque)  # (channel, note) -> deque[(ts, velocity, instrument)]
        # matched_notes = []  # (channel, note, start_ts, end_ts, velocity, instrument)
        # unmatched_count = 0
    
        # for ts, event, _tempo in flat_events:
        #     if event.note is None:
        #         continue  # skip non-note events (control_change, etc.) for now
    
        #     key = (event.channel, event.note)
        #     etype = (event.event_type or "").lower()
    
        #     if etype == "note_on":
        #         instrument_name = event.instrument or (event.metadata or {}).get("instrument")
        #         pending_note_on[key].append((ts, event.velocity or 80, instrument_name))
    
        #     elif etype == "note_off":
        #         if pending_note_on[key]:
        #             start_ts, velocity, instrument_name = pending_note_on[key].popleft()
        #             matched_notes.append((event.channel, event.note, start_ts, ts, velocity, instrument_name))
        #         else:
        #             logger.debug(
        #                 "note_off with no matching note_on: channel=%s note=%s ts=%.3f",
        #                 event.channel, event.note, ts,
        #             )
    
        # # Close out any dangling note_on events (never got a note_off)
        # last_ts = flat_events[-1][0]
        # fallback_duration_seconds = step_duration * (60.0 / base_tempo)  # quarterLength -> seconds
        # for (channel, note), queue in pending_note_on.items():
        #     while queue:
        #         start_ts, velocity, instrument_name = queue.popleft()
        #         end_ts = max(start_ts + fallback_duration_seconds, last_ts)
        #         matched_notes.append((channel, note, start_ts, end_ts, velocity, instrument_name))
        #         unmatched_count += 1
    
        # if unmatched_count:
        #     logger.warning(
        #         "%d note_on event(s) had no matching note_off — closed with fallback duration.",
        #         unmatched_count,
        #     )
    
        # if not matched_notes:
        #     logger.warning("No matched notes to write — resulting MIDI would be empty.")
        #     return
    
        # # ── Step 4: Insert every note into its channel's Part at absolute time ──
        # for channel, note, start_ts, end_ts, velocity, instrument_name in matched_notes:
        #     if not (0 <= int(note) <= 127):
        #         logger.debug("Skipping out-of-range MIDI note: %s", note)
        #         continue
    
        #     part = get_part(channel, instrument_name)
    
        #     offset = seconds_to_quarter_length(start_ts - t0)
        #     quarter_length = max(step_duration, seconds_to_quarter_length(end_ts - start_ts))
    
        #     n = m21.note.Note(int(note), quarterLength=quarter_length)
        #     n.volume.velocity = int(max(0, min(127, velocity)))
    
        #     # insert() places the note at an absolute offset, preserving gaps
        #     # (silence) and avoiding clobbering notes on other channels.
        #     part.insert(offset, n)
    
        # # ── Step 5: Assemble the score ───────────────────────────────────────────
        # stream = m21.stream.Score()
    
        # tempo_mark = m21.tempo.MetronomeMark(number=base_tempo)
        # if parts:
        #     first_part = next(iter(parts.values()))
        #     first_part.insert(0, tempo_mark)
    
        # for channel in sorted(parts.keys()):
        #     stream.insert(0, parts[channel])
    
        # # ── Step 6: Save ──────────────────────────────────────────────────────────
        # output_dir = os.path.join(os.getcwd(), "modules", "Music_Generator", "Generated Melodies")
        # os.makedirs(output_dir, exist_ok=True)
        # filename = f"generated_melody_{int(time.time())}.mid"
        # file_path = os.path.join(output_dir, filename)
    
        # try:
        #     stream.write("midi", file_path)
        #     logger.info(
        #         "✅ Generated melody saved to %s | channels=%d | notes=%d | unmatched=%d",
        #         file_path, len(parts), len(matched_notes), unmatched_count,
        #     )
        #     return file_path
        # except Exception as e:
        #     logger.error(f"❌ Error saving MIDI: {e}")
        #     return None

        # if not self.generated_melody:
        #     logger.warning("No generated melody to save.")
        #     return
        
        # # Folder to save generated melodies
        # output_dir = os.path.join(os.getcwd(), "modules", "Music_Generator", "Generated Melodies")
        # os.makedirs(output_dir, exist_ok=True)

        # filename = f"generated_melody_{int(time.time())}.mid"
        # file_path = os.path.join(output_dir, filename)

        # stream = m21.stream.Stream()
        # active_notes = {}   # Key: (channel, note), Value: (start_time, velocity, instrument)

        # for frame in self.generated_melody:

        #     current_time = frame.frame_id * step_duration

        #     for event in frame.events:

        #         key = (event.channel, event.note)

        #         if event.event_type == "note_on":

        #             active_notes[key] = {
        #                 "start_time": current_time,
        #                 "velocity": event.velocity,
        #                 "instrument": event.instrument,
        #             }
                
        #         elif event.event_type == "note_off":

        #             if key in active_notes:
        #                 note_info = active_notes.pop(key)
        #                 start_time = note_info["start_time"]
        #                 duration = current_time - start_time

        #                 midi_note = m21.note.Note(event.note)
        #                 midi_note.volume.velocity = note_info["velocity"]
        #                 midi_note.quarterLength = duration
        #                 midi_note.offset = start_time
        #                 midi_note.storedInstrument = m21.instrument.fromString(note_info["instrument"])

        #                 stream.append(midi_note)

        #     # Close remaining active notes
        #     if self.generated_melody:

        #         end_time = (self.generated_melody[-1].frame_id + 1) * step_duration

        #         for (channel, note), note_info in active_notes.items():

        #             start_time = note_info["start_time"]
        #             duration = end_time - start_time

        #             midi_note = m21.note.Note(note)
        #             midi_note.volume.velocity = note_info["velocity"]
        #             midi_note.quarterLength = duration
        #             midi_note.offset = start_time
        #             midi_note.storedInstrument = m21.instrument.fromString(note_info["instrument"])

        #             stream.append(midi_note)    

        # try:
        #     stream.write("midi", file_path)
        #     logger.info(f"✅ Generated melody saved to {file_path}")
        #     return file_path
        # except Exception as e:
        #     logger.error(f"❌ Error saving MIDI: {e}")
        #     return None

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
