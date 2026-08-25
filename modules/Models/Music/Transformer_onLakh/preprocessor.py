"""
Preprocessing module for Lakh MIDI (LMD) dataset.
========================================================================

This module contains functions to preprocess the Lakh MIDI dataset for training an Transformer model.
"""

import argparse
import mido
from dataclasses import dataclass
from pathlib import Path

from modules.config import LMD_DATASET_PATH
from modules.utils.logging_setup import setup_logging

logger = setup_logging("INFO", name="Music_Generator.Transformer_OnLakh")


@dataclass(frozen=True)
class MusicEvent:
    event_type: str
    pitch: int
    track: int
    velocity: int


@dataclass(frozen=True)
class MusicFrame:
    index: int
    events: tuple[MusicEvent, ...]


def _iter_midi_files(dataset_path: Path):
    yield from dataset_path.rglob("*.mid")
    yield from dataset_path.rglob("*.midi")


def _is_percussion(message: mido.Message) -> bool:
    return message.type in {"note_on", "note_off"} and message.channel == 9


def _is_note_message(message: mido.Message) -> bool:
    return message.type in {"note_on", "note_off"}


def _canonical_event_type(message: mido.Message) -> str:
    if message.type == "note_off":
        return "note_off"

    if message.type == "note_on" and message.velocity == 0:
        return "note_off"

    return "note_on"


def _quantize_tick(tick: int, ticks_per_beat: int) -> int:
    sixteenth = ticks_per_beat / 4
    return int(round(tick / sixteenth))


def midi_to_music_frames(
    midi_path: Path,
    remove_percussion: bool = True,
) -> list[MusicFrame]:
    midi = mido.MidiFile(str(midi_path))

    absolute_events = []

    for track_index, track in enumerate(midi.tracks):
        absolute_tick = 0

        for message in track:
            absolute_tick += message.time

            if not _is_note_message(message):
                continue

            if remove_percussion and _is_percussion(message):
                continue

            event_type = _canonical_event_type(message)

            absolute_events.append(
                (
                    absolute_tick,
                    event_type,
                    message.note,
                    track_index,
                    message.velocity,
                )
            )

    if not absolute_events:
        return []

    frame_events: dict[int, list[MusicEvent]] = {}

    for tick, event_type, pitch, track_index, velocity in absolute_events:
        frame_index = _quantize_tick(tick, midi.ticks_per_beat)

        event = MusicEvent(
            event_type=event_type,
            pitch=pitch,
            track=track_index,
            velocity=velocity,
        )

        frame_events.setdefault(frame_index, []).append(event)

    frames = []

    for frame_index in sorted(frame_events):
        events = frame_events[frame_index]

        # A note cannot be turned on and off at the same quantized position.
        # If malformed MIDI data produces both events, keep the note-off state.
        normalized_events: dict[tuple[int, int], MusicEvent] = {}

        for event in events:
            key = (event.track, event.pitch)

            if key not in normalized_events:
                normalized_events[key] = event
                continue

            existing = normalized_events[key]

            if event.event_type == "note_off":
                normalized_events[key] = event
            elif existing.event_type != "note_off":
                normalized_events[key] = event

        frames.append(
            MusicFrame(
                index=frame_index,
                events=tuple(
                    sorted(
                        normalized_events.values(),
                        key=lambda event: (
                            event.track,
                            event.pitch,
                            event.event_type,
                        ),
                    )
                ),
            )
        )

    return frames


def _format_event(event: MusicEvent) -> str:
    return (
        f"{event.event_type.upper():8s} "
        f"pitch={event.pitch:3d} "
        f"track={event.track:3d} "
        f"velocity={event.velocity:3d}"
    )


def inspect_midi(midi_path: Path) -> None:
    frames = midi_to_music_frames(midi_path)

    logger.info(
        f"{midi_path.name}: "
        f"{len(frames)} non-empty MusicFrames, "
        f"{sum(len(frame.events) for frame in frames)} events."
    )

    for frame in frames[:20]:
        logger.info(f"Frame {frame.index}:")

        for event in frame.events:
            logger.info(f"  {_format_event(event)}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Lakh MIDI files into quantized MusicFrames."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Inspect only the first N MIDI files.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=LMD_DATASET_PATH,
        help="Path to the Clean Lakh MIDI dataset.",
    )
    parser.add_argument(
        "--keep-percussion",
        action="store_true",
        help="Keep MIDI channel 10 percussion events.",
    )
    args = parser.parse_args()

    midi_files = list(_iter_midi_files(args.dataset))

    if args.limit is not None:
        midi_files = midi_files[:args.limit]

    logger.info(f"Found {len(midi_files)} MIDI files to inspect.")

    for midi_path in midi_files:
        try:
            frames = midi_to_music_frames(
                midi_path,
                remove_percussion=not args.keep_percussion,
            )

            logger.info(
                f"{midi_path.name}: "
                f"{len(frames)} non-empty MusicFrames, "
                f"{sum(len(frame.events) for frame in frames)} events."
            )

            for frame in frames[:10]:
                logger.info(f"Frame {frame.index}:")

                for event in frame.events:
                    logger.info(f"  {_format_event(event)}")

        except (OSError, EOFError, ValueError) as error:
            logger.warning(
                f"Skipping invalid MIDI '{midi_path}': {type(error).__name__}: {error}"
            )


if __name__ == "__main__":
    main()
