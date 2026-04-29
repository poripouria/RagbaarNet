from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import mido


@dataclass(frozen=True)
class PianoRollConfig:
    steps_per_beat: int = 4
    segment_bars: int = 4
    beats_per_bar: int = 4
    ignore_drums: bool = True

    @property
    def segment_steps(self) -> int:
        return int(self.segment_bars * self.beats_per_bar * self.steps_per_beat)


def midi_to_pianoroll(
    midi_path: str,
    *,
    steps_per_beat: int = 4,
    ignore_drums: bool = True,
) -> np.ndarray:
    """Convert a MIDI file to a binary piano-roll (time_steps, 128).

    Time is quantized in *beats* using ticks_per_beat, so it is robust to tempo changes.
    """
    if not os.path.exists(midi_path):
        raise FileNotFoundError(midi_path)

    mid = mido.MidiFile(midi_path)
    ticks_per_beat = int(mid.ticks_per_beat)
    if ticks_per_beat <= 0:
        raise ValueError("Invalid ticks_per_beat")

    merged = mido.merge_tracks(mid.tracks)

    # key: (channel, note)
    active_notes: Dict[Tuple[int, int], int] = {}
    events: list[tuple[int, int, int]] = []  # (start_tick, end_tick, note)

    current_tick = 0
    for msg in merged:
        current_tick += int(msg.time)
        if msg.type not in {"note_on", "note_off"}:
            continue

        channel = getattr(msg, "channel", 0)
        note = int(getattr(msg, "note", 0))
        if note < 0 or note > 127:
            continue
        if ignore_drums and channel == 9:
            continue

        is_note_on = msg.type == "note_on" and int(getattr(msg, "velocity", 0)) > 0
        is_note_off = msg.type == "note_off" or (msg.type == "note_on" and int(getattr(msg, "velocity", 0)) == 0)

        key = (int(channel), int(note))
        if is_note_on:
            # If already active, close previous note.
            if key in active_notes:
                start = active_notes.pop(key)
                if current_tick > start:
                    events.append((start, current_tick, note))
            active_notes[key] = current_tick
        elif is_note_off:
            if key in active_notes:
                start = active_notes.pop(key)
                if current_tick > start:
                    events.append((start, current_tick, note))

    # Close any dangling notes at end.
    end_tick = current_tick
    for (_, note), start in list(active_notes.items()):
        if end_tick > start:
            events.append((start, end_tick, note))

    if not events:
        # Empty roll
        return np.zeros((1, 128), dtype=np.float32)

    max_tick = max(e[1] for e in events)
    total_steps = int(np.ceil(max_tick * steps_per_beat / ticks_per_beat))
    total_steps = max(total_steps, 1)

    roll = np.zeros((total_steps, 128), dtype=np.float32)
    for start_tick, end_tick, note in events:
        start_step = int(start_tick * steps_per_beat / ticks_per_beat)
        end_step = int(np.ceil(end_tick * steps_per_beat / ticks_per_beat))
        start_step = max(start_step, 0)
        end_step = max(end_step, start_step + 1)
        if start_step >= total_steps:
            continue
        end_step = min(end_step, total_steps)
        roll[start_step:end_step, note] = 1.0

    return roll


def extract_random_segment(
    roll: np.ndarray,
    *,
    segment_steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a fixed-length segment (segment_steps, 128) from a roll.

    Pads with zeros if roll is shorter.
    """
    if roll.ndim != 2 or roll.shape[1] != 128:
        raise ValueError("roll must have shape (T, 128)")

    if roll.shape[0] >= segment_steps:
        max_start = roll.shape[0] - segment_steps
        start = int(rng.integers(0, max_start + 1))
        return roll[start : start + segment_steps]

    pad = np.zeros((segment_steps, 128), dtype=np.float32)
    pad[: roll.shape[0]] = roll
    return pad
