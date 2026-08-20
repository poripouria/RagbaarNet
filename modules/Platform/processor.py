"""
Modular Processing Framework for Turning Sequential Input into Music
======================================================================

This module implements the Processor class, which orchestrates the flow of data from input sources 
through a selected processing channel and into a music generation pipeline. 
The architecture is designed to be modular, allowing for different input modalities to be processed 
by different channels, each with its own detection strategy.
"""

import time
import threading
import traceback
from typing import Any, Dict
from queue import Queue, Empty

from modules import config
from modules.Detection.Detector import Detector
from modules.Music_Generator.Musician import Musician
from modules.Platform.channels import BaseChannel, AVAILABLE_CHANNELS
from modules.Platform.midi_output import MidiOutput
from modules.utils.logging_setup import setup_logging, set_level
logger = setup_logging("INFO", name="Platform.processor")


class Processor:
    """
    Orchestrates turning queued input into music.
    """

    def __init__(self, socketio_instance=None):
        """
        Initialize the Processor, including selecting the active channel and starting the processing loop.
        """

        self.socketio = socketio_instance
        self.frame_counter = 0

        self.input_queue = Queue(maxsize=config.INPUT_QUEUE_MAXSIZE)

        self.is_processing = False
        self.current_frame = None
        self.current_display = None

        self.debug_mode = False
        self.last_debug_time = 0
        self.last_socket_debug_time = 0
        self.debug_interval = config.DEBUG_INTERVAL

        self.main_ui_connected = False
        self.status_page_clients = set()

        self._is_shutdown = False
        self._shutdown_lock = threading.Lock()

        logger.info("🔄 Initializing music generation platfom...")
        try:
            self.musician = Musician(
                config.DEFAULT_MUSICIAN_TYPE,
                tempo=config.DEFAULT_TEMPO,
                key_signature=config.DEFAULT_KEY_SIGNATURE,
                time_signature=config.DEFAULT_TIME_SIGNATURE
            )
            self.current_music = None
            self.music_enabled = True
            logger.info("✅ Music Generator initialized successfully")
        except Exception as e:
            logger.exception("❌ Error initializing musician: %s", e)
            self.musician = None
            self.music_enabled = False

        # Determine audio backend for music output (tone, midi, or both)
        self.audio_backend = config.DEFAULT_AUDIO_BACKEND
        if self.audio_backend not in ('tone', 'midi', 'both'):
            logger.warning("⚠️ Invalid audio backend '%s' - defaulting to 'tone'.", self.audio_backend)
            self.audio_backend = 'tone'
        self.midi_output = None
        if self.audio_backend in ('midi', 'both'):
            try:
                logger.info("🔄 Initializing MIDI output backend...")
                self.midi_output = MidiOutput()
            except Exception as e:
                logger.exception("❌ Error initializing MIDI output: %s", e)
                self.midi_output = None
        else:
            logger.info("🔊 Audio backend set to '%s' - MIDI output disabled.", self.audio_backend)

        # Pick exactly one channel for this run and the single Detector strategy that goes with it.
        self.channel = self._select_channel()
        self.detector = Detector(self.channel.detector_strategy)

        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()

    def _select_channel(self) -> BaseChannel:
        """
        Prompt once at startup for which input modality this run will use.
        """

        print("CHOOSE A PROCESSING CHANNEL (this cannot be changed without restarting):")
        for key, cls in AVAILABLE_CHANNELS.items():
            print(f"  {key}. {cls.name}", "[DEFAULT]" if key == 1 else "")

        choice_raw = input("Enter your choice: ").strip()
        try:
            choice = int(choice_raw) if choice_raw else 1
        except ValueError:
            choice = None

        channel_cls = AVAILABLE_CHANNELS.get(choice)
        if channel_cls is None:
            logger.warning("⚠️ Invalid choice '%s' - defaulting to 'driving'.", choice_raw)
            channel_cls = AVAILABLE_CHANNELS.get(1)
        
        return channel_cls()

    # --- ingress ---------------------------------------------------------------

    def add_frame(self, frame, frame_id=None, timestamp=None, roi_points=None, roi_controls=None):
        """
        Queue a video frame. Only consumed if the active channel expects 'frame'.
        """
        
        self._enqueue({
            'kind': 'frame',
            'frame': frame,
            'frame_id': frame_id or f"frame_{self.frame_counter}",
            'timestamp': timestamp if timestamp is not None else time.time(),
            'roi_points': roi_points,
            'roi_controls': roi_controls,
        })

    def add_event(self, source_name: str, raw_payload: dict, timestamp=None):
        """
        Queue a generic event (keystroke, scroll, ...). Only consumed if the active channel expects 'event'.
        """

        self._enqueue({
            'kind': 'event',
            'source_name': source_name,
            'raw_payload': raw_payload,
            'timestamp': timestamp if timestamp is not None else time.time(),
        })

    def _enqueue(self, item: Dict[str, Any]):
        if item['kind'] != self.channel.expected_kind:
            logger.warning(
                "⚠️ Ignoring '%s' input - active channel is '%s', which expects '%s' input. "
                "Restart with a different channel choice to use this input instead.",
                item['kind'], self.channel.name, self.channel.expected_kind,
            )
            return

        if self.input_queue.full():
            try:
                logger.warning("⚠️ Input queue full - dropping oldest item to make room for new input.")
                self.input_queue.get_nowait()
            except Empty:
                logger.error("❌ Error occurred while trying to drop an item from the input queue.")
                pass

        self.input_queue.put(item)

    # --- processing loop ---------------------------------------------------------

    def _processing_loop(self):
        """
        Main loop that continuously processes items from the input queue, applies the selected channel's
        logic, and updates the display and music generation accordingly.
        """
        logger.info("🚀 Processing loop started (channel: %s, interval: %d)", 
                    self.channel.name, 
                    self.channel.processing_interval if hasattr(self.channel, 'processing_interval') else 1)

        while True:
            try:
                item = self.input_queue.get(timeout=1.0)
                if item is None:
                    break   # Shutdown signal received, exit the loop

                dropped_stale = 0
                if item.get('kind') == 'frame':
                    while True:
                        try:
                            newer_item = self.input_queue.get_nowait()
                        except Empty:
                            break
                        if newer_item is None:
                            self.input_queue.put(None)
                            break
                        item = newer_item
                        dropped_stale += 1
                if dropped_stale:
                    logger.debug("⏩ Skipped %d stale queued frame(s) to catch up to the latest.", dropped_stale)

                detector_input, display_payload = self.channel.to_observation(item)

                self.current_frame = item['frame'] if item['kind'] == 'frame' else None
                item_id = item.get('frame_id') if item['kind'] == 'frame' else item.get('source_name')

                if display_payload is not None:
                    display_payload = {
                        **display_payload,
                        'frame_id': item_id,
                        'timestamp': item['timestamp'],
                        'frame_counter': self.frame_counter,
                    }
                    self.current_display = display_payload
                    self._broadcast_display_update()

                if detector_input is not None:
                    scene_events = []
                    try:
                        scene_events = self.detector(input=detector_input, frame_id=self.frame_counter)
                    except Exception as detector_err:
                        logger.error("❌ Detector error: %s", detector_err)

                    # Run the musician every frame!
                    if self.music_enabled and self.musician is not None:

                        music_frame = self.musician(
                            results=scene_events, 
                            frame_id=self.frame_counter, 
                            state=getattr(self.detector.detector, 'state', None)
                        )

                        music_data = {
                            'frame_id': item_id,
                            'timestamp': item['timestamp'],
                            'frame_counter': self.frame_counter,
                            'music_frame': music_frame,
                            'events_count': len(music_frame.events),
                            'tempo': self.musician.tempo,
                            'key_signature': self.musician.key_signature,
                            'time_signature': self.musician.time_signature,
                        }

                        self.current_music = music_data
                        self._broadcast_music_update(music_data)

                logger.info("🎞️ Frame %d processed.", self.frame_counter)
                if self.debug_mode and (time.time() - self.last_debug_time) > self.debug_interval:
                    logger.debug("🖥️ Frame %d info: Queue size: %d, Last debug time: %f",
                                 self.frame_counter, self.input_queue.qsize(), time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.last_debug_time)))
                    self.last_debug_time = time.time()

                self.frame_counter += 1

            except Empty:
                continue
            except Exception as e:
                logger.exception("❌ Error in processing loop: %s", e)
                logger.error("Traceback:\n%s", traceback.format_exc())

    # --- broadcasting -----------------------------------------------------------

    def _broadcast_display_update(self):
        """
        Immediately broadcast the current display state to connected clients.
        """

        try:
            if self.main_ui_connected and self.socketio:
                display_data = self.get_synchronized_display(for_main_ui=True)
                state = self.get_current_state()
                response_data = {**display_data, 'queue_size': state['queue_size']}
                self.socketio.emit('frame_update', response_data)

        except Exception as e:
            logger.error("❌ Error broadcasting display update: %s", e)

    def _broadcast_music_update(self, music_data):
        """
        Broadcast music events to connected WebSocket clients.
        """

        try:
            if self.main_ui_connected and self.socketio:
                music_frame = music_data['music_frame']
                events_data = []

                for event in music_frame.events:
                    instrument_name = event.instrument
                    if instrument_name in ('unknown', None, ''):
                        logger.error("❌ Event has unknown instrument: %s", event)

                    events_data.append({
                        'event_type': event.event_type,
                        'note': event.note,
                        'channel': event.channel,
                        'velocity': event.velocity,
                        'instrument': instrument_name,
                        'timestamp': event.timestamp,
                    })

                music_response = {
                    'frame_id': music_data['frame_id'],
                    'frame_counter': music_data['frame_counter'],
                    'events': events_data,
                    'events_count': music_data['events_count'],
                    'tempo': music_data['tempo'],
                    'key_signature': music_data['key_signature'],
                    'time_signature': music_data['time_signature'],
                    'timestamp': music_data['timestamp'],
                    'audio_backend': self.audio_backend,
                }

                self.socketio.emit('music_update', music_response)

                if self.midi_output is not None:
                    self.midi_output.send_music_frame(music_frame)

        except Exception as e:
            logger.error("❌ Error broadcasting music update: %s", e)

    # --- state / display queries used by main.py's routes ------------------------

    def get_current_state(self):
        """
        Return a dictionary representing the current state of the processor.
        """

        return {
            'frame_counter': self.frame_counter,
            'current_frame_available': self.current_frame is not None,
            'current_display_available': self.current_display is not None,
            'current_music_available': self.current_music is not None if hasattr(self, 'current_music') else False,
            'music_enabled': self.music_enabled if hasattr(self, 'music_enabled') else False,
            'active_channel': getattr(self.channel, 'name', None),
            'queue_size': self.input_queue.qsize(),
        }

    def get_synchronized_display(self, for_main_ui=True):
        """
        Get synchronized frame and segmentation data for display.
        """

        display_data = {
            'original_frame': None,
            'segmentation_overlay': None,
            'segmentation_info': None,
            'music_info': None,
            'frame_counter': self.frame_counter,
            'timestamp': time.time(),
        }

        if self.current_display is not None:
            payload = self.current_display
            frame_diff = self.frame_counter - payload['frame_counter']

            if frame_diff <= 10:
                if for_main_ui:
                    display_data['segmentation_overlay'] = payload.get('overlay_b64')

                display_data['segmentation_info'] = {
                    'frame_id': payload['frame_id'],
                    'timestamp': payload['timestamp'],
                    'frame_counter': payload['frame_counter'],
                    'frames_since_segmentation': frame_diff,
                    'class_labels': payload.get('detected_classes') or [],
                    'model_type': payload.get('model_type'),
                }

        if hasattr(self, 'current_music') and self.current_music is not None:
            music_data = self.current_music
            frame_diff = self.frame_counter - music_data['frame_counter']

            if frame_diff <= 10:
                display_data['music_info'] = {
                    'frame_id': music_data['frame_id'],
                    'frame_counter': music_data['frame_counter'],
                    'events_count': music_data['events_count'],
                    'tempo': music_data['tempo'],
                    'key_signature': music_data['key_signature'],
                    'time_signature': music_data['time_signature'],
                    'frames_since_music': frame_diff,
                    'timestamp': music_data['timestamp'],
                }

        return display_data

    # --- music controls ----------------------------------------------------------

    def get_available_musicians(self):
        """Get the list of musician types the UI can offer, plus the current selection"""

        musicians = Musician.list_available_musicians()
        current = None
        instrument = 'piano'
        if hasattr(self, 'musician') and self.musician is not None:
            current = self.musician.musician_type
            instrument = self.musician.instrument

        return {'musicians': musicians, 'current': current, 'instrument': instrument}

    def apply_music_settings(self, musician_type: str, tempo: int, instrument: str):
        """Apply musician, tempo, and LSTM instrument settings together."""

        if not hasattr(self, 'musician') or self.musician is None:
            return {'success': False, 'error': 'Musician system not initialized'}

        try:
            tempo = int(tempo)
            if not 60 <= tempo <= 300:
                raise ValueError('Tempo must be between 60 and 300 BPM')

            if musician_type != self.musician.musician_type:
                self.musician.switch_musician(musician_type, tempo=tempo, instrument=instrument)
            else:
                self.musician.set_tempo(tempo)
                if musician_type == 'lstm-onessen':
                    self.musician.set_instrument(instrument)

            return {
                'success': True,
                'musician_type': self.musician.musician_type,
                'tempo': self.musician.tempo,
                'instrument': self.musician.instrument,
            }
        except Exception as e:
            logger.error(f"❌ Error applying music settings: {e}")
            return {'success': False, 'error': str(e)}

    # --- lifecycle / misc ---------------------------------------------------------

    def shutdown(self):
        """Shutdown the processor safely and only once."""

        with self._shutdown_lock:

            if self._is_shutdown:
                logger.debug("Processor shutdown already completed.")
                return

            self._is_shutdown = True

            if self.midi_output is not None:
                self.midi_output.close()

            logger.info("🎼 Saving generated music...")
            if self.musician is not None:
                self.musician.save_generated_melody()

            logger.info("🛑 Shutting down Main processor...")
            # Send shutdown signal to the processing loop
            try:
                self.input_queue.put(None)
            except Exception:
                logger.exception("❌ Failed to send shutdown signal to input queue.")

        # Wait for processing thread outside the lock
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)

        logger.info("✅ Processor shutdown complete.")

    def enable_debug_mode(self, enable=True):
        """Enable or disable debug mode for verbose logging"""

        self.debug_mode = enable
        set_level(logger, "DEBUG" if enable else "INFO")
        if getattr(self, 'channel', None) is not None:
            self.channel.set_debug_mode(enable)
        logger.info("🐛 Debug mode %s", "enabled - verbose logging activated" if enable else "disabled - minimal logging activated")

    def set_main_ui_connected(self, connected=True):
        """Mark main UI as connected/disconnected to prioritize it over status page"""

        if self.main_ui_connected != connected:
            self.main_ui_connected = connected
            logger.info("🎯 Main UI connected - prioritizing data for main interface" if connected else "📄 Main UI disconnected")
