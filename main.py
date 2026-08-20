"""
RagbaarNet - Application Entrypoint
====================================

Run: `python main.py`.
"""

import os
import time
import cv2
import numpy as np
import threading
import base64
import signal
import argparse
from flask import Flask, jsonify, send_from_directory, redirect, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS

from modules import config
from modules.Platform.processor import Processor
from modules.utils.logging_setup import setup_logging, set_level
logger = setup_logging("INFO", name="Platform.main")


# Initialize Flask app and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = config.SECRET_KEY
CORS(app)  # Enable CORS for all routes

# Reduce Socket.IO/engineio log noise in production
socketio = SocketIO(app, cors_allowed_origins=config.CORS_ALLOWED_ORIGINS)

# Global processor instance - pass socketio for real-time broadcasting
processor = Processor(socketio_instance=socketio)

# Paths for serving the existing web UI (so mobile devices can load it from the laptop)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PLATFORM_DIR = os.path.join(PROJECT_ROOT, 'modules', 'Platform')
ASSETS_DIR = os.path.join(PROJECT_ROOT, 'assets')

# Additional CORS headers for all routes
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response


# Telemetry receiver (Android app)
telemetry_app = Flask('telemetry_receiver')

@telemetry_app.route('/telemetry', methods=['GET', 'POST'])
def receive_telemetry():
    """Receive telemetry data from the Android app."""

    if request.method == 'GET':
        return jsonify({'success': True, 'message': 'Telemetry server is alive!'})

    data = request.get_json(silent=True) or {}
    socketio.emit('telemetry_update', {
        'speed_kmh': data.get('speed_kmh'),
        'accel': data.get('accel'),
        'rpm': data.get('rpm'),
    })

    return jsonify({'success': True})

def run_telemetry_server(host=config.SERVER_HOST, port=config.TELEMETRY_PORT):
    logger.info("📡 Starting telemetry receiver on %s:%s (POST /telemetry)", host, port)
    telemetry_app.run(host=host, port=port, debug=False, use_reloader=False)


@app.route('/')
def index():
    """Redirect the root URL to the main UI so the processor server is usable directly."""
    return redirect('/ui/', code=302)

@app.route('/ui')
def ui_redirect():
    """Redirect /ui to /ui/ so static assets resolve correctly."""
    return redirect('/ui/', code=302)

@app.route('/ui/')
def ui_index():
    """Serve the main Platform UI entrypoint (UI.html).

    Keeping UI.html as-is means all existing responsive behavior and JS logic stays identical;
    relative links (styles.css/script.js) resolve under /ui/ automatically.
    """
    return send_from_directory(PLATFORM_DIR, 'UI.html')

@app.route('/ui/<path:filename>')
def ui_static(filename: str):
    """Serve Platform UI static files (script.js, styles.css, etc.)."""

    return send_from_directory(PLATFORM_DIR, filename)

@app.route('/ui2')
def ui2_redirect():
    """Redirect /ui2 to /ui2/ so static assets resolve correctly."""
    return redirect('/ui2/', code=302)

@app.route('/ui2/')
def ui2_index():
    """Serve the code-editor mockup (UI2.html) that drives the 'typing' channel.

    Self-contained (inline CSS/JS, no external script.js dependency) since it's
    a small standalone demo, not part of the main video UI.
    """
    return send_from_directory(PLATFORM_DIR, 'UI2.html')

@app.route('/assets/<path:filename>')
def serve_assets(filename: str):
    """Serve shared project assets (icons, etc.) referenced by UI.html."""

    return send_from_directory(ASSETS_DIR, filename)

@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    """Receive frame data from UI and add to processing queue"""

    try:
        data = request.get_json()

        if 'frame' not in data:
            return jsonify({'error': 'No frame data provided'}), 400

        # Decode base64 frame
        frame_data = data['frame']
        if frame_data.startswith('data:image'):
            # Remove data URL prefix
            frame_data = frame_data.split(',')[1]

        # Decode image
        img_buffer = base64.b64decode(frame_data)
        img_array = np.frombuffer(img_buffer, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Invalid frame data'}), 400

        # Convert BGR to RGB for proper processing
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Add frame to processor
        frame_id = data.get('frame_id', f"frame_{int(time.time() * 1000)}")
        timestamp = data.get('timestamp', time.time())

        roi_points = data.get("roi_points", [])
        roi_controls = data.get("roi_controls", [])

        processor.add_frame(
            frame,
            frame_id,
            timestamp,
            roi_points=roi_points,
            roi_controls=roi_controls
        )

        # Get current state
        state = processor.get_current_state()

        return jsonify({
            'success': True,
            'frame_counter': state['frame_counter'],
            'queue_size': state['queue_size'],
            'message': 'Frame processed successfully'
        })

    except Exception as e:
        logger.exception("❌ Error processing frame: %s", e)
        return jsonify({'error': str(e)}), 500

@app.route('/api/input_event', methods=['POST'])
def input_event():
    """Generic ingress for event-based channels."""

    try:
        data = request.get_json(silent=True) or {}
        source_name = data.get('source')
        payload = data.get('payload')
        if not source_name or payload is None:
            return jsonify({'error': "'source' and 'payload' are required"}), 400
        processor.add_event(source_name, payload, timestamp=data.get('timestamp'))
        return jsonify({'success': True})
    
    except Exception as e:
        logger.exception("❌ Error handling input event: %s", e)
        return jsonify({'error': str(e)}), 500

@socketio.on('input_event')
def handle_input_event(data):
    """Socket.IO twin of /api/input_event, for low-latency streaming sources (e.g. the VSCode extension)."""

    try:
        data = data or {}
        source_name = data.get('source')
        payload = data.get('payload')
        if not source_name or payload is None:
            emit('input_event_ack', {'success': False, 'error': "'source' and 'payload' are required"})
            return
        processor.add_event(source_name, payload, timestamp=data.get('timestamp'))

    except Exception as e:
        emit('input_event_ack', {'success': False, 'error': str(e)})
        logger.error("❌ Error handling input_event (socket): %s", e)

@app.route('/api/get_display', methods=['GET'])
def get_display():
    """Get synchronized display data - prioritized for main UI"""

    try:
        # Mark main UI as connected when it requests data
        processor.set_main_ui_connected(True)
        display_data = processor.get_synchronized_display(for_main_ui=True)
        return jsonify(display_data)  
    except Exception as e:
        logger.exception("❌ Error getting display data: %s", e)
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get processor status"""

    try:
        state = processor.get_current_state()
        return jsonify(state)
    except Exception as e:
        logger.exception("❌ Error getting status: %s", e)
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug/<action>', methods=['POST'])
def toggle_debug(action):
    """Toggle debug mode for performance monitoring"""

    try:
        if action == 'enable':
            processor.enable_debug_mode(True)
            set_level(logger, "DEBUG")
            return jsonify({'success': True, 'debug_mode': True})
        elif action == 'disable':
            processor.enable_debug_mode(False)
            set_level(logger, "INFO")
            return jsonify({'success': True, 'debug_mode': False})
        else:
            return jsonify({'error': 'Invalid action. Use enable or disable'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@socketio.on('request_update')
def handle_update_request():
    """Handle real-time update requests via WebSocket - PRIORITIZED FOR MAIN UI"""

    try:
        # Check if this is from main UI or status page
        is_main_ui = request.sid not in processor.status_page_clients

        if is_main_ui:
            # Mark main UI as connected and get full data
            processor.set_main_ui_connected(True)
            display_data = processor.get_synchronized_display(for_main_ui=True)
        else:
            # Status page gets limited data to avoid conflicts
            display_data = processor.get_synchronized_display(for_main_ui=False)

        state = processor.get_current_state()

        # Combine display data with state
        response_data = {**display_data, 'queue_size': state['queue_size']}

        # Always emit, even if no new segmentation data - client decides what to display
        try:
            emit('frame_update', response_data)
        except Exception as emit_err:
            if isinstance(emit_err, (BrokenPipeError, ConnectionResetError, OSError, RuntimeError)):
                logger.debug("Client disconnected while emitting frame update: %s", emit_err)
            else:
                logger.exception("❌ Error emitting frame update: %s", emit_err)

        # Debug logging (only when enabled and interval reached)
        if processor.debug_mode and (time.time() - processor.last_socket_debug_time) > processor.debug_interval:
            processor.last_socket_debug_time = time.time()
            has_overlay = 'segmentation_overlay' in response_data and response_data['segmentation_overlay'] is not None
            client_type = "Main UI" if is_main_ui else "Status Page"
            logger.debug("📡 Update sent to %s - Frame: %s, Has overlay: %s, Queue: %s",
                         client_type, response_data.get('frame_counter', 0), has_overlay, response_data.get('queue_size', 0))

    except Exception as e:
        emit('error', {'message': str(e)})
        logger.error("❌ Error handling update request: %s", e)

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""

    # Determine if this is status page or main UI based on referrer.
    referrer = (request.headers.get('Referer', '') or '').lower()
    # When Referer is missing (e.g., some WebViews), default to Main UI.
    is_main_ui = (not referrer) or ('/ui/' in referrer) or ('/ui2/' in referrer) or referrer.endswith(('/ui', '/ui2'))

    if is_main_ui:
        processor.set_main_ui_connected(True)
        logger.info("🎯 Main UI connected: %s", request.sid)
        return

    processor.status_page_clients.add(request.sid)
    logger.info("📄 Status page connected: %s", request.sid)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""

    if request.sid in processor.status_page_clients:
        processor.status_page_clients.remove(request.sid)
        logger.info("📄 Status page disconnected: %s", request.sid)
    else:
        processor.set_main_ui_connected(False)
        logger.info("🎯 Main UI disconnected: %s", request.sid)

@socketio.on('toggle_music')
def handle_toggle_music(data):
    """Enable or disable music generation (Handle music generation toggle from client)"""

    try:
        enable = data.get('enabled', True)
        if hasattr(processor, 'music_enabled'):
            processor.music_enabled = (not processor.music_enabled) if enable is None else enable
            logger.info(f"🎵 Music generation {'enabled' if processor.music_enabled else 'disabled'}")
            result = processor.music_enabled
        emit('music_status', {'enabled': result, 'success': True})
    except Exception as e:
        emit('music_status', {'error': str(e), 'success': False})
        logger.error("❌ Error toggling music: %s", e)

@socketio.on('set_music_tempo')
def handle_set_music_tempo(data):
    """Set music tempo (BPM) (Handle music tempo change from client)"""

    try:
        tempo = data.get('tempo', 120)
        if hasattr(processor, 'musician') and processor.musician is not None:
            processor.musician.set_tempo(tempo)
            logger.info(f"🎵 Music tempo set to {tempo} BPM")
        emit('music_status', {'tempo': tempo, 'success': True})
    except Exception as e:
        emit('music_status', {'error': str(e), 'success': False})
        logger.error("❌ Error setting music tempo: %s", e)

@socketio.on('set_music_key')
def handle_set_music_key(data):
    """Handle music key change from client"""

    try:
        key_signature = data.get('key_signature', 'C_major')
        if hasattr(processor, 'musician') and processor.musician is not None:
            processor.musician.set_key_signature(key_signature)
            logger.info("🎵 Music key set to: %s", key_signature)
        emit('music_status', {'key_signature': key_signature, 'success': True})
    except Exception as e:
        emit('music_status', {'error': str(e), 'success': False})
        logger.error("❌ Error setting music key: %s", e)

@socketio.on('set_music_time')
def handle_set_music_time(data):
    """Handle music time signature change"""

    try:
        time_signature = data.get('time_signature', (4, 4))
        if hasattr(processor, 'musician') and processor.musician is not None:
            processor.musician.set_time_signature(time_signature)
            logger.info("🎵 Music time signature set to: %s", time_signature)
        emit('music_status', {'time_signature': time_signature, 'success': True})
    except Exception as e:
        emit('music_status', {'error': str(e), 'success': False})
        logger.error("❌ Error setting music time signature: %s", e)

@socketio.on('get_music_status')
def handle_get_music_status():
    """Get current music generation status"""

    try:
        if hasattr(processor, 'musician') and processor.musician is not None:
            status = {
                'enabled': getattr(processor, 'music_enabled', False),
                'tempo': processor.musician.tempo,
                'key_signature': processor.musician.key_signature,
                'time_signature': processor.musician.time_signature,
                'musician_type': processor.musician.musician_type,
                'instrument': processor.musician.instrument,
            }
        else:
            status = {'enabled': False, 'musician_available': False}
        emit('music_status', status)
    except Exception as e:
        emit('music_status', {'error': str(e), 'success': False})
        logger.error("❌ Error getting music status: %s", e)

@socketio.on('get_available_musicians')
def handle_get_available_musicians():
    """Send the available musicians and current music settings to the client."""

    try:
        data = processor.get_available_musicians()
        emit('musicians_list', data)
    except Exception as e:
        emit('musicians_list', {'error': str(e), 'musicians': [], 'current': None})
        logger.error("❌ Error getting available musicians: %s", e)

@socketio.on('set_music_settings')
def handle_set_music_settings(data):
    """Apply the combined music settings from the platform UI."""

    try:
        settings = data or {}
        musician_type = settings.get('musician_type')
        if not musician_type:
            emit('music_settings_updated', {'success': False, 'error': 'musician_type is required'})
            return

        result = processor.apply_music_settings(
            musician_type=musician_type,
            tempo=settings.get('tempo', 120),
            instrument=settings.get('instrument', 'piano')
        )
        emit('music_settings_updated', result)
        if result.get('success'):
            logger.info("🎵 Music settings updated: musician=%s, instrument=%s, tempo=%s",
                        result.get('musician_type'), result.get('instrument'), result.get('tempo'))
        
    except Exception as e:
        emit('music_settings_updated', {'success': False, 'error': str(e)})
        logger.error("❌ Error applying music settings: %s", e)

def shutdown_server(signum, frame):
    """Handle server shutdown signals."""

    logger.info("\n🛑 Shutdown signal received (%s).", signum)
    try:
        processor.shutdown()
    except Exception:
        logger.exception("❌ Error while shutting down processor.")
    logger.info("✅ Server shutdown complete.")
    raise SystemExit(0)

def run_processor_server(host=config.SERVER_HOST, port=config.PROCESSOR_PORT, debug=False, interval=1):
    """Run the processor server"""

    signal.signal(signal.SIGINT, shutdown_server)
    signal.signal(signal.SIGTERM, shutdown_server)

    logger.info("🚀 Starting Video Processor Server on %s:%s", host, port)
    logger.info("🎥 Active channel: %s", processor.get_current_state().get('active_channel'))
    logger.info("🌐 Web interface available at:")
    logger.info("   - UI:     http://%s:%s/ui/", host, port)
    logger.info("   - UI2:    http://%s:%s/ui2/", host, port)
    logger.info("📡 API endpoints:")
    logger.info("   - POST /api/process_frame - Send frame data")
    logger.info("   - GET  /api/get_display   - Get synchronized display")
    logger.info("   - GET  /api/status        - Get processor status")
    logger.info("   - POST /api/debug/enable  - Enable verbose debug logging")
    logger.info("   - POST /api/debug/disable - Disable debug logging for performance")
    logger.info("   - POST http://%s:5500/telemetry - Receive Android telemetry (speed/accel/rpm)", host)
    logger.info("🚀 Performance Mode:")
    logger.info("   - Processing interval: %s frames", interval)
    logger.info("   - Debug mode: %s", "Enabled" if debug else "Disabled")

    threading.Thread(target=run_telemetry_server, daemon=True).start()

    try:
        socketio.run(app, host=host, port=port, debug=debug, use_reloader=False)
    except Exception as e:
        logger.exception("❌ Server error: %s", e)
        processor.shutdown()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Main Processing Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (use 0.0.0.0 for LAN/mobile access)')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--interval', type=int, default=1, help='Processing interval in frames (only used by the segmentation channel)')

    args = parser.parse_args()

    if args.interval != 1 and hasattr(processor.channel, 'processing_interval'):
        processor.channel.processing_interval = args.interval
        logger.info("🔄 Updated '%s' channel interval to %s frames", processor.channel.name, args.interval)

    # Set debug mode based on argument
    if args.debug:
        processor.enable_debug_mode(True)
        set_level(logger, "DEBUG")

    run_processor_server(args.host, args.port, args.debug, args.interval)
