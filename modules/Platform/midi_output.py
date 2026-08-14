"""
RagbaarNet MIDI output backend.

This module sends generated MusicFrame events to a MIDI output port 
such as LoopMIDI, allowing an external DAW to render the music.
"""

import os
from typing import Optional
import mido

from Music_Generator.Musician import MusicFrame
from utils.logging_setup import setup_logging

logger = setup_logging("INFO", name="Platform.MidiOutput")


class MidiOutput:
    """Send MusicFrame events to a configured MIDI output port."""

    def __init__(self, port_name: Optional[str] = None):
        self.port_name = port_name or os.environ.get(
            "RAGBAARNET_MIDI_PORT",
            "RagbaarNetMIDI Port 1",
        ).strip()
        self.port = None

        self._open()

    def _open(self):
        """Open the configured MIDI output port."""

        available_ports = mido.get_output_names()
        if self.port_name not in available_ports:
            raise RuntimeError(
                f"MIDI output port '{self.port_name}' was not found. "
                f"Available ports: {available_ports}"
            )

        self.port = mido.open_output(self.port_name)
        logger.info("🎹 MIDI output opened: %s", self.port_name)

    def send_music_frame(self, music_frame: MusicFrame):
        """Send all MIDI-compatible events in a MusicFrame."""

        if self.port is None:
            return

        for event in music_frame.events:
            if event.note is None:
                continue

            self.port.send(mido.Message(
                event.event_type,
                note=event.note,
                velocity=event.velocity,
                channel=event.channel,
            ))

    def close(self):
        """Close the MIDI output port."""

        if self.port is not None:
            self.port.close()
            self.port = None
            logger.info("🎹 MIDI output closed: %s", self.port_name)
