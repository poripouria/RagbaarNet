"""
Modular Music Generation Framework for Real-Time Visual-to-Audio Mapping
========================================================================

This module provides an extensible framework for generating music based on visual data,
It supports various music generation strategies with easy integration for additional models.
"""

import time
import json
import mido
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

from modules import config
from modules.utils.logging_setup import setup_logging
logger = setup_logging("INFO", name="Music_Generator.Musician")


# Fixed MIDI channel per instrument voice, shared by every musician implementation
INSTRUMENT_MIDI_CHANNELS = config.INSTRUMENT_MIDI_CHANNELS


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
        timeid: Unique identifier for the event, useful for tracking and debugging
        timestamp: Time at which the event occurs
        metadata: Additional event-specific information
    """

    event_type: str             # e.g. "note_on", "note_off"
    note: Optional[int] = None
    channel: int = 0
    velocity: Optional[int] = None
    instrument: Optional[str] = None
    timeid: Optional[int] = None
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
        metadata: Additional frame-specific information
    """

    events: List[MusicEvent]
    frame_id: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseMusician(ABC):
    """
    Abstract base class for all music generation models.
    This class defines the interface that all music generation models must implement,
    ensuring consistency and extensibility across different generation strategies.
    """

    NOTE_ON_TYPES = frozenset({"ROI_TOUCH", "NOTE_ON"})
    NOTE_OFF_TYPES = frozenset({"ROI_RELEASE", "NOTE_OFF"})

    def __init__(self, key_signature: str=config.DEFAULT_KEY_SIGNATURE, time_signature: tuple=config.DEFAULT_TIME_SIGNATURE):
        """
        Initialize the base musician.

        Args:
            key_signature: Key signature for music generation
            time_signature: Time signature for music generation
        """

        self.key_signature = key_signature
        self.time_signature = time_signature
        self.active_notes = {i: {} for i in range(16)}  # Initialize active notes for all 16 MIDI channels

        self.frame_counter = 0
        self.max_missing_frames = config.MAX_MISSING_FRAMES     # Number of frames to keep an object in memory after it disappears

    def __call__(self, results: List[Dict[str, Any]], frame_id: int=0, state: Dict[str, Any]=None):

        return self.generate_music(results, frame_id, state)

    @staticmethod
    def _is_stale(state: Dict[str, Any], object_id: Any) -> bool:
        """
        Whether a held note's underlying object should be auto-released.
        Only meaningful for Detector states that track per-object 'missing_frames'. 
        Detector strategies without that concept simply never go stale here.
        """

        if not isinstance(state, dict) or "objects" not in state:
            return False

        return object_id not in state["objects"]

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

    def __init__(self, key_signature=config.DEFAULT_KEY_SIGNATURE, time_signature: tuple=config.DEFAULT_TIME_SIGNATURE):
        """
        Args:
            tempo: Music tempo in BPM
            key_signature: Key signature for music generation
            time_signature: Time signature for music generation
        """
        super().__init__(key_signature, time_signature)

        logger.info(f"🎵 {self.__class__.__name__} initialized with key_signature={key_signature}, time_signature={time_signature}")

    def _map_classes(self, obj_class):
        """
        Map object class to MIDI note, velocity, instrument, and channel."""

        base_class = obj_class.split("_")[0]
        mapping = config.RULE_BASED_CLASS_MAPPING

        entry = mapping.get(base_class, None)
        if entry is None:
            return None
        note, velocity, instrument = entry
        return note, velocity, instrument, INSTRUMENT_MIDI_CHANNELS.get(instrument, 0)
    
    def generate_music(self, results, frame_id, state):
        """
        Generate music based on the input scene data.
        """

        logger.info(f"🎵 Generating Rule-Based music for frame {frame_id}")

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
                    timeid=self.frame_counter,
                    timestamp=time.time(),
                    metadata=e
                )
            )
            logger.info(f"Mapped scene event: {e} to music event: 'type': {event}, 'note': {note}, 'velocity': {velocity if e['type'] in self.NOTE_ON_TYPES else 0}, 'instrument': '{instrument}'")

        for channel in self.active_notes:
            for object_id in list(self.active_notes[channel].keys()):
                if self._is_stale(state, object_id):
                    note_info = self.active_notes[channel].get(object_id)
                    if note_info:
                        music_events.append(MusicEvent(
                            event_type="note_off",
                            note=note_info["note"],
                            channel=channel,
                            velocity=0,
                            instrument=note_info["instrument"],
                            timeid=self.frame_counter,
                            timestamp=time.time(),
                            metadata={"object_id": object_id}
                        ))
                        self.active_notes[channel].pop(object_id, None)
                    logger.warning(f"Auto-released note for object_id {object_id} due to missing frames.")

        return MusicFrame(
            events=music_events,
            frame_id=frame_id,
            metadata={
                "scene_events": scene_events,
            }
        )

class LSTMMusician(BaseMusician):
    """
    LSTM-based musician that generates music using a trained LSTM model. This musician
    leverages an LSTM to produce monophonic melodies based on learned patterns from Essen folk song training data.
    """

    AVAILABLE_INSTRUMENTS = (
        "piano", "electric_piano", 
        "strings", 
        "acoustic_guitar", "bass", "electric_guitar",
        "pad", "synth"
    )

    def __init__(self, key_signature=config.DEFAULT_KEY_SIGNATURE, time_signature: tuple=config.DEFAULT_TIME_SIGNATURE, temperature=0.9, instrument="piano"):
        """
        Args:
            key_signature: Key signature for music generation
            time_signature: Time signature for music generation
            temperature: Sampling temperature for LSTM model
            instrument: Tone.js instrument used to play generated melodies
        """
        super().__init__(key_signature, time_signature)
        self.temperature = temperature

        if instrument not in self.AVAILABLE_INSTRUMENTS:
            raise ValueError(f"Unsupported LSTM instrument: {instrument}. Supported instruments: {", ".join(self.AVAILABLE_INSTRUMENTS)}")
        self.instrument = instrument

        from modules.Models.Music.LSTM_OnEssen.generator import MelodyGenerator
        self.generator = MelodyGenerator()
        self._rt_generator = None

        self.last_seed_notes = ["67", "_", "67", "_", "67", "_", "_", "65", 
                                "64", "_", "62", "_", "60", "_", "60", "_",
                                "48", "_", "_", "_", "50", "_", "52", "_",
                                "60", "61", "62", "_", "60", "_", "60", "_"]
        self._note_buffer = list(self.last_seed_notes)

        self.important_labels = [
            "car", "truck", "bus", "train", "plane",
            "bicycle", "motorcycle", "person",
            "traffic light", "traffic sign", "stop sign",
        ]

        logger.info(f"🎵 {self.__class__.__name__} initialized with key_signature={key_signature}, time_signature={time_signature}, temperature={temperature}")

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

                area = e.get("area/ROI", None)
                if area is not None:
                    if area < 0.005:
                        logger.info(f"Event with very small area ({area}). Using minimum velocity for class '{obj_class}'.")
                        velocity = 15
                    else:
                        # Scale area to velocity range (Power Curve Scaler) Area:0.005-0.5, Velocity:32-128
                        normalized_area = min(1.0, (area - 0.005) / (0.5 - 0.005))
                        curved_area = normalized_area ** 0.8
                        velocity = int(curved_area * (127 - 31) + 31)
                else:
                    # Non-spatial events (e.g. a keyboard NOTE_ON) have no area - fall back to the event's 'intensity'
                    intensity = max(0.0, min(1.0, e.get("intensity", 1.0)))
                    velocity = int(intensity * (127 - 31) + 31)

                # Generate new notes using the LSTM model
                self._rt_generator = self.generator.generate_melody_RT(
                    seed=" ".join(self.last_seed_notes),
                    length=100,
                    temperature=self.temperature
                )

                new_note = next(self._rt_generator)
                note = int(new_note)


                channel = INSTRUMENT_MIDI_CHANNELS.get(self.instrument, 0)

                self.active_notes[0][e["object_id"]] = {
                    "voice_id": e["object_id"],
                    "note": note,
                    "velocity": velocity,
                    "instrument": self.instrument,
                    "channel": channel,
                }

                self._note_buffer.append(new_note)

            elif e["type"] in self.NOTE_OFF_TYPES:
                event = "note_off"
                
                # Find the related note for this object_id
                related_note = None
                if e["object_id"] in self.active_notes[0]:
                    related_note = self.active_notes[0][e["object_id"]]["note"]
                    channel = self.active_notes[0][e["object_id"]]["channel"]
                    self.active_notes[0].pop(e["object_id"], None)
                else:
                    logger.warning(f"No previous note found to turn off on release event for object_id {e['object_id']}.")
                    continue
                note = related_note

            else:
                # Check if there are any active notes or not, to append a rest
                if any(self.active_notes[0].values()):
                    self._note_buffer.append("_")
                else:
                    self._note_buffer.append("r")
                
                continue

            music_events.append(
                MusicEvent(
                    event_type=event,
                    note=note,
                    channel=channel,
                    velocity=velocity,
                    instrument=self.instrument,
                    timeid=self.frame_counter,
                    timestamp=time.time(),
                    metadata=e
                )
            )
            logger.info(f"Mapped scene event: {e} to music event: 'type': {event}, 'note': {note}, 'velocity': {velocity}, 'instrument': '{self.instrument}', 'channel': {channel}")

            self.last_seed_notes = self._note_buffer[-32:]

        for object_id, note_info in list(self.active_notes[0].items()):
            if self._is_stale(state, object_id):
                music_events.append(
                    MusicEvent(
                        event_type="note_off",
                        note=note_info["note"],
                        channel=note_info.get("channel", 0),
                        velocity=0,
                        instrument=note_info["instrument"],
                        timeid=self.frame_counter,
                        timestamp=time.time(),
                        metadata={"object_id": object_id}
                    )
                )
                self.active_notes[0].pop(object_id, None)
                logger.warning(f"Auto-released note for object_id {object_id} due to missing frames.")

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
    is similar to the LSTMMusician but is designed to produce orchestral sounds.
    """

    def __init__(self, key_signature=config.DEFAULT_KEY_SIGNATURE, time_signature=config.DEFAULT_TIME_SIGNATURE, temperature=0.9):
        """
        Args:
            key_signature: Key signature for music generation
            time_signature: Time signature for music generation
            temperature: Sampling temperature for LSTM model
        """
        super().__init__(key_signature, time_signature)

        self.temperature = temperature

        from modules.Models.Music.LSTM_OnEssen.generator import MelodyGenerator
        self.generator = MelodyGenerator()
        self._rt_generator = None

        self.last_seed_notes = {
            instrument: ["64", "_", "67", "_", "65", "_", "65", "_",
                         "65", "_", "_", "_", "62", "_", "64", "_",
                         "64", "_", "67", "_", "65", "_", "65", "_",
                         "48", "_", "_", "50", "62", "_", "64", "_"]
            for instrument in ["piano", "electric_piano", "bass", "strings", "pad"]
        }
        
        # Note buffer to store generated notes for each instrument
        self._note_buffer = self.last_seed_notes.copy()

        logger.info(f"🎵 {self.__class__.__name__} initialized with key_signature={key_signature}, time_signature={time_signature}, temperature={temperature}")

    def _map_classes(self, obj_class):
        """
        Map object class to instrument and channel for orchestral sounds.
        """

        base_class = obj_class.split("_")[0]
        mapping = config.LSTM_ORCHESTRAL_CLASS_MAPPING

        instrument = mapping.get(base_class, None)
        if instrument is None:
            return None
        return instrument, INSTRUMENT_MIDI_CHANNELS.get(instrument, 0)

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
                    if area < 0.005:
                        logger.info(f"Event with very small area ({area}). Using minimum velocity for class '{obj_class}'.")
                        velocity = 15
                    else:
                        normalized_area = min(1.0, (area - 0.005) / (0.5 - 0.005))
                        curved_area = normalized_area ** 0.8
                        velocity = int(curved_area * (127 - 31) + 31)
                else:
                    intensity = max(0.0, min(1.0, e.get("intensity", 1.0)))
                    velocity = int(intensity * (127 - 31) + 31)

                self._rt_generator = self.generator.generate_melody_RT(
                    seed=" ".join(self.last_seed_notes[instrument]),
                    length=100,
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
                    logger.warning(f"No previous note found to turn off on release event for object_id {e['object_id']}.")
                    continue
                note = related_note

            else:
                if any(self.active_notes[channel].values()):
                    self._note_buffer[instrument].append("_")
                else:
                    self._note_buffer[instrument].append("r")
                continue

            music_events.append(
                MusicEvent(
                    event_type=event,
                    note=note,
                    channel=channel,
                    velocity=velocity,
                    instrument=instrument,
                    timeid=self.frame_counter,
                    timestamp=time.time(),
                    metadata=e
                )
            )
            logger.info(f"Mapped scene event: {e} to music event: 'type': {event}, 'note': {note}, 'velocity': {velocity}, 'instrument': '{instrument}'")

            self.last_seed_notes[instrument] = self._note_buffer[instrument][-32:]

        for channel in self.active_notes:
            for object_id, note_info in list(self.active_notes[channel].items()):
                if self._is_stale(state, object_id):
                    music_events.append(
                        MusicEvent(
                            event_type="note_off",
                            note=note_info["note"],
                            channel=channel,
                            velocity=0,
                            instrument=note_info["instrument"],
                            timeid=self.frame_counter,
                            timestamp=time.time(),
                            metadata={"object_id": object_id}
                        )
                    )
                    self.active_notes[channel].pop(object_id, None)
                    logger.warning(f"Auto-released note for object_id {object_id} due to missing frames.")

        return MusicFrame(
            events=music_events,
            frame_id=frame_id,
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
    }

    def __init__(self, musician_type: str=config.DEFAULT_MUSICIAN_TYPE,
                 tempo: int=config.DEFAULT_TEMPO, key_signature: str=config.DEFAULT_KEY_SIGNATURE,
                 time_signature: tuple=config.DEFAULT_TIME_SIGNATURE, instrument: str="piano"):
        """
        Initialize the main Musician.

        Args:
            musician_type: Type of musician, see Musician.MUSICIAN_REGISTRY for supported values.
            tempo: Music tempo in BPM
            key_signature: Key signature for music generation
            time_signature: Time signature for music generation
            instrument: Instrument used by the LSTM musician
        """

        self.musician_type = musician_type.lower()
        self.tempo = tempo
        self.key_signature = key_signature
        self.time_signature = time_signature
        self.instrument = instrument

        entry = self.MUSICIAN_REGISTRY.get(self.musician_type)
        if entry is None:
            raise ValueError(f"Unsupported musician type: {musician_type}. Supported types: {", ".join(sorted(self.MUSICIAN_REGISTRY.keys()))}")
        self.musician = self._create_musician(entry)

        self.generated_melody = []

        # Timeline of tempo/key/time-signature snapshots
        self._settings_history = []
        self._record_settings_change()

    def _record_settings_change(self) -> None:
        """
        Snapshot the current tempo/key/time-signature (plus musician/instrument) with a timestamp, 
        but only if it actually differs from the last recorded snapshot.
        """

        snapshot = {
            "timestamp": time.time(),
            "tempo": self.tempo,
            "key_signature": self.key_signature,
            "time_signature": tuple(self.time_signature),
            "musician_type": self.musician_type,
            "instrument": self.instrument,
        }

        if self._settings_history:
            last = self._settings_history[-1]
            if all(last[k] == snapshot[k] for k in ("tempo", "key_signature", "time_signature", "musician_type", "instrument")):
                return

        self._settings_history.append(snapshot)

    def _create_musician(self, entry):
        if entry["class"] is LSTMMusician:
            return entry["class"](self.key_signature, self.time_signature, instrument=self.instrument)
        return entry["class"](self.key_signature, self.time_signature)

    def switch_musician(
        self,
        musician_type: str,
        tempo: Optional[int] = None,
        key_signature: Optional[str] = None,
        time_signature: Optional[tuple] = None,
        instrument: Optional[str] = None
    ):
        """
        Switch to a different music generation model.

        Args:
            musician_type: New musician type
            tempo: New tempo (keeps current if None)
            key_signature: New key signature (keeps current if None)
            time_signature: New time signature (keeps current if None)
            instrument: LSTM instrument (keeps current if None)
        """

        self.musician_type = musician_type.lower()
        self.tempo = self.tempo if tempo is None else tempo
        self.key_signature = self.key_signature if key_signature is None else key_signature
        self.time_signature = self.time_signature if time_signature is None else time_signature
        self.instrument = self.instrument if instrument is None else instrument

        entry = self.MUSICIAN_REGISTRY.get(self.musician_type)
        if entry is None:
            available = ", ".join(sorted(self.MUSICIAN_REGISTRY.keys()))
            raise ValueError(f"Unsupported musician type: {musician_type}. Supported types: {available}")
        self.musician = self._create_musician(entry)
        self._record_settings_change()

        logger.info(f"🎭 Musician switched to: {musician_type}")

    def set_tempo(self, tempo: int) -> None:
        self.tempo = tempo
        self._record_settings_change()

    def set_key_signature(self, key_signature: str) -> None:
        self.key_signature = key_signature
        if hasattr(self.musician, "key_signature"):
            self.musician.key_signature = key_signature
        self._record_settings_change()

    def set_time_signature(self, time_signature: tuple) -> None:
        self.time_signature = time_signature
        if hasattr(self.musician, "time_signature"):
            self.musician.time_signature = time_signature
        self._record_settings_change()

    def set_instrument(self, instrument: str) -> None:
        if instrument not in LSTMMusician.AVAILABLE_INSTRUMENTS:
            raise ValueError(f"Unsupported instrument: {instrument}. Supported instruments: {", ".join(LSTMMusician.AVAILABLE_INSTRUMENTS)}")
        self.instrument = instrument
        if isinstance(self.musician, LSTMMusician):
            self.musician.instrument = instrument
        self._record_settings_change()

    @classmethod
    def list_available_musicians(cls) -> List[dict]:
        """
        Return metadata for every musician type that can be selected/switched to. Used by the 
        Platform UI to populate the music settings picker without duplicating the list of supported types.
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

    def save_generated_melody(self, save_path: Optional[str] = None, format: str = "both") -> None:
        """
        Save every MusicFrame generated so far as a MIDI file and a companion JSON dump.

        Args:
            format: The format to save the melody in ("midi", "json", or "both").
            save_path: Optional directory or base file path (without extension) for the output.
        """

        def _resolve_track_channel(instrument: str, event_channel: int, channel_owners: dict) -> int:

            preferred = config.INSTRUMENT_MIDI_CHANNELS.get(instrument, event_channel)
            if preferred not in channel_owners or channel_owners[preferred] == instrument:
                channel_owners[preferred] = instrument
                return preferred

            logger.warning(f"MIDI channel {preferred} is already used by '{channel_owners[preferred]}' - reassigning '{instrument}' to avoid a program-change collision.")
            for candidate in range(16):
                if candidate == 9:  # reserved for the fixed GM percussion kit
                    continue
                if candidate not in channel_owners:
                    channel_owners[candidate] = instrument
                    return candidate

            logger.warning(f"No free MIDI channel left for '{instrument}' - reusing channel {preferred}, which may still collide.")
            return preferred

        if format not in ("both", "midi", "json"):
            logger.warning(f"Unknown save format '{format}' - defaulting to 'both'.")
            format = "both"

        logger.info(f"🎼 Settings history for this melody ({len(self._settings_history)} change(s)):")
        for snapshot in self._settings_history:
            logger.info(
                f"   - tempo={snapshot['tempo']} key={snapshot['key_signature']} "
                f"time_signature={snapshot['time_signature']} musician={snapshot['musician_type']}"
            )

        if not any(frame.events for frame in self.generated_melody):
            logger.warning("No generated melody events to save - skipping.")
            return None

        # Path handling
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        default_name = f"melody_{self.musician_type}_{timestamp}"
        if save_path is None:
            base = config.GENERATED_MELODIES_DIR / default_name
        else:
            base = Path(save_path)
            if base.is_dir() or str(save_path).endswith(("/", "\\")):
                base = base / default_name
            else:
                base = base.with_suffix("")
        base.parent.mkdir(parents=True, exist_ok=True)
        base_path =  base

        if format == "both" or format == "midi":
            midi_path = base_path.with_suffix(".mid")
            try:
                ppq = config.MIDI_TICKS_PER_BEAT
                
                timeline = [(snapshot["timestamp"], "settings", snapshot) 
                            for snapshot in self._settings_history]
                for frame in self.generated_melody:
                    for event in frame.events:
                        if event.note is None:
                            continue
                        timeline.append((event.timestamp, "note", event))
                timeline.sort(key=lambda item: item[0])
                if not timeline:
                    raise ValueError("Nothing to export: no settings history or note events were recorded.")
        
                # --- Pass 1: assign every timeline item an absolute tick position ---
                current_tempo_bpm = timeline[0][2]["tempo"] if timeline[0][1] == "settings" else self.tempo
                current_time = timeline[0][0]
                current_tick = 0.0
                ticked_timeline: List[Tuple[float, str, Any]] = []
                for abs_time, kind, payload in timeline:
                    dt = max(0.0, abs_time - current_time)
                    current_tick += dt * (current_tempo_bpm / 60.0) * ppq
                    ticked_timeline.append((current_tick, kind, payload))
                    current_time = abs_time
                    if kind == "settings":
                        current_tempo_bpm = payload["tempo"]
        
                midi_file = mido.MidiFile(type=1, ticks_per_beat=ppq)
        
                # --- Meta track: tempo, key signature, and time signature changes ---
                meta_track = mido.MidiTrack()
                meta_track.name = "meta"
                midi_file.tracks.append(meta_track)
        
                last_meta_tick, last_tempo, last_key, last_time_sig = 0, None, None, None
                for tick, kind, payload in ticked_timeline:
                    if kind != "settings":
                        continue
                    tick_int = round(tick)
        
                    if payload["tempo"] != last_tempo:
                        meta_track.append(mido.MetaMessage(
                            "set_tempo", tempo=mido.bpm2tempo(payload["tempo"]),
                            time=max(0, tick_int - last_meta_tick)
                        ))
                        last_meta_tick, last_tempo = tick_int, payload["tempo"]
        
                    mido_key = config._KEY_SIGNATURE_TO_MIDO.get(str(payload["key_signature"]).strip().lower().replace(' ', '_'))
                    if mido_key is None:
                        logger.warning(f"Skipping unrecognized key signature '{payload['key_signature']}' in MIDI export.")
                    elif mido_key != last_key:
                        meta_track.append(mido.MetaMessage(
                            "key_signature", key=mido_key, time=max(0, tick_int - last_meta_tick)
                        ))
                        last_meta_tick, last_key = tick_int, mido_key
        
                    if payload["time_signature"] != last_time_sig:
                        numerator, denominator = payload["time_signature"]
                        meta_track.append(mido.MetaMessage(
                            "time_signature", numerator=numerator, denominator=denominator,
                            time=max(0, tick_int - last_meta_tick)
                        ))
                        last_meta_tick, last_time_sig = tick_int, payload["time_signature"]
        
                # --- One note track per distinct instrument ---
                channel_owners = {}
                tracks_by_instrument = {}
                last_tick_by_instrument = {}
        
                for tick, kind, payload in ticked_timeline:
                    if kind != "note":
                        continue
                    event: MusicEvent = payload
                    instrument = event.instrument or "unknown"
        
                    if instrument not in tracks_by_instrument:
                        channel = _resolve_track_channel(instrument, event.channel, channel_owners)
                        track = mido.MidiTrack()
                        track.name = f"{instrument} (ch{channel})"
                        if channel != 9:  # channel 9 is the fixed GM percussion kit, no program needed
                            program = config.GM_INSTRUMENT_PROGRAMS.get(instrument, 0)
                            track.append(mido.Message("program_change", program=program, channel=channel, time=0))
                        midi_file.tracks.append(track)
                        tracks_by_instrument[instrument] = (track, channel)
                        last_tick_by_instrument[instrument] = 0
        
                    track, channel = tracks_by_instrument[instrument]
                    tick_int = round(tick)
                    delta = max(0, tick_int - last_tick_by_instrument[instrument])
                    track.append(mido.Message(
                        event.event_type, note=event.note, velocity=event.velocity or 0,
                        channel=channel, time=delta
                    ))
                    last_tick_by_instrument[instrument] = tick_int
        
                midi_file.save(str(midi_path))
                logger.info(f"🎼 Generated melody saved as MIDI: {midi_path}")
            except Exception:
                logger.exception("❌ Failed to save generated melody as MIDI.")
                midi_path = None

        if format == "both" or format == "json":
            json_path = base_path.with_suffix(".json")
            try:
                payload = {
                    "musician_type": self.musician_type,
                    "saved_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    "settings_history": self._settings_history,
                    "frames": [
                        {
                            "frame_id": frame.frame_id,
                            "metadata": frame.metadata,
                            "events": [asdict(event) for event in frame.events],
                        }
                        for frame in self.generated_melody
                    ],
                }
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, default=str)
                logger.info(f"🖋️ Generated melody saved as JSON: {json_path}")
            except Exception:
                logger.exception("❌ Failed to save generated melody as JSON.")
                json_path = None

    def __call__(self, results, frame_id: int = 0, state: Dict[str, Any] = None) -> MusicFrame:
        """
        Generate music based on segmentation data.

        Args:
            results: Detection results
            frame_id: Frame identifier for tracking
            state: Current state of the music generation process
        Returns:
            MusicFrame containing generated music events
        """

        generated_frame =  self.musician(results, frame_id, state)
        self.generated_melody.append(generated_frame)

        return generated_frame
