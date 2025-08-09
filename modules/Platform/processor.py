"""
Modular Processing Framework for Receiving Data and Performing Segmentation
=================================================

This module receives data from UI.html and processes it using the Segmentor class.
Then sends the processed data back to UI.html.
"""

import cv2
import numpy as np
import base64
import time
import threading
import colorsys
import argparse
from queue import Queue, Empty
from flask import Flask, request, jsonify, render_template_string
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import os
import sys

# Add the modules directory to the path to import Segmentor
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Segmentation.Segmentor import Segmentor

class VideoProcessor:
    """
    Video Processing Class that handles frame reception, segmentation, and synchronization
    """
    
    def __init__(self, socketio_instance=None):
        """Initialize the video processor with segmentation models"""
        self.socketio = socketio_instance  # Store socketio instance for broadcasting
        self.frame_counter = 0
        self.segmentation_interval = 2  # Reduced from 3 to 2 for smoother updates
        self.frame_queue = Queue(maxsize=10)  # Reduced buffer size for lower latency
        self.segmentation_queue = Queue(maxsize=5)  # Smaller queue for faster updates
        self.current_frame = None
        self.current_segmentation = None
        self.is_processing = False
        
        # Performance optimization flags
        self.debug_mode = False  # Turn off debug prints for production
        self.last_debug_time = 0
        self.debug_interval = 5.0  # Only print debug info every 5 seconds
        
        # Connection management to avoid dual streaming conflicts
        self.main_ui_connected = False  # Track if main UI is connected
        self.status_page_clients = set()  # Track status page connections
        
        # Pre-compute color mapping arrays for faster lookup
        self.color_mapping_array = None
        
        # Create consistent color mapping for segmentation classes
        self.color_map = self._create_consistent_color_map()
        self._prepare_color_mapping_array()
        
        # Cache for image encoding to avoid repeated allocations
        self.encode_params = [cv2.IMWRITE_JPEG_QUALITY, 75]  # Reduced quality for speed
        
        # Initialize segmentation models
        print("🔄 Initializing segmentation models...")
        try:
            # YOLO model
            # model_path = os.path.join(os.path.dirname(__file__), '..', 'Segmentation', 'Pre-trained Models', 'yolov8m-seg.pt')
            # self.segmentor = Segmentor('yolo', model_path)
            # print("✅ YOLO Segmentor initialized successfully")
            
            # SegFormer model
            self.segmentor = Segmentor('segformer')
            print("✅ SegFormer Segmentor initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing segmentor: {e}")
            self.segmentor = None
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
    
    def _create_consistent_color_map(self):
        """Create a consistent color mapping for segmentation classes"""
        # Cityscapes color palette (SegFormer model classes)
        # These colors match the standard Cityscapes dataset visualization
        color_map = {
            0: [128, 64, 128],    # Road - purple
            1: [244, 35, 232],    # Sidewalk - magenta  
            2: [70, 70, 70],      # Building - gray
            3: [102, 102, 156],   # Wall - blue-gray
            4: [190, 153, 153],   # Fence - pink-gray
            5: [153, 153, 153],   # Pole - light gray
            6: [250, 170, 30],    # Traffic light - orange
            7: [220, 220, 0],     # Traffic sign - yellow
            8: [107, 142, 35],    # Vegetation - olive green
            9: [152, 251, 152],   # Terrain - light green
            10: [70, 130, 180],   # Sky - sky blue
            11: [220, 20, 60],    # Person - red
            12: [255, 0, 0],      # Rider - bright red
            13: [0, 0, 142],      # Car - dark blue
            14: [0, 0, 70],       # Truck - darker blue
            15: [0, 60, 100],     # Bus - blue
            16: [0, 80, 100],     # Train - teal
            17: [0, 0, 230],      # Motorcycle - blue
            18: [119, 11, 32],    # Bicycle - dark red
        }
        
        # Add a background/unlabeled class
        color_map[255] = [0, 0, 0]  # Black for unlabeled regions
        
        # Extend with additional colors for more classes if needed
        for i in range(19, 255):
            # Generate consistent colors using HSV color space
            hue = (i * 137.5) % 360  # Golden angle for good distribution
            saturation = 70 + (i % 3) * 15  # Vary saturation
            value = 180 + (i % 4) * 20       # Vary brightness
            
            # Convert HSV to RGB
            import colorsys
            r, g, b = colorsys.hsv_to_rgb(hue/360, saturation/100, value/255)
            color_map[i] = [int(r*255), int(g*255), int(b*255)]
        
        print("🎨 Color mapping for Cityscapes classes:")
        cityscapes_labels = [
            "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
            "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
            "truck", "bus", "train", "motorcycle", "bicycle"
        ]
        for i, label in enumerate(cityscapes_labels):
            print(f"   Class {i}: {label} → RGB{color_map[i]}")
        
        return color_map
    
    def _prepare_color_mapping_array(self):
        """Pre-compute color mapping array for vectorized operations"""
        # Create a lookup table for all possible class IDs (0-255)
        self.color_mapping_array = np.zeros((256, 3), dtype=np.uint8)
        for class_id, color in self.color_map.items():
            self.color_mapping_array[class_id] = color
        
    def _processing_loop(self):
        """Main processing loop that runs in a separate thread"""
        print("🚀 Processing loop started")
        
        while True:
            try:
                # Get frame from queue (timeout prevents blocking)
                frame_data = self.frame_queue.get(timeout=1.0)
                
                if frame_data is None:  # Shutdown signal
                    break
                    
                frame = frame_data['frame']
                frame_id = frame_data['frame_id']
                timestamp = frame_data['timestamp']
                
                self.current_frame = frame
                
                # Process segmentation every N frames
                if self.frame_counter % self.segmentation_interval == 0 and self.segmentor is not None:
                    # Reduced logging for performance
                    if self.debug_mode and (time.time() - self.last_debug_time) > self.debug_interval:
                        print(f"🔍 Processing segmentation for frame {self.frame_counter}")
                        self.last_debug_time = time.time()
                    
                    try:
                        # Perform segmentation
                        result = self.segmentor(frame)
                        
                        # Create segmentation visualization (optimized)
                        segmentation_overlay = self._create_segmentation_overlay_optimized(frame, result)
                        
                        # Store result
                        segmentation_data = {
                            'frame_id': frame_id,
                            'timestamp': timestamp,
                            'frame_counter': self.frame_counter,
                            'segmentation_map': result.segmentation_map,
                            'overlay': segmentation_overlay,
                            'class_labels': result.class_labels,
                            'metadata': result.metadata
                        }
                        
                        # Add to segmentation queue (remove old ones if full)
                        if self.segmentation_queue.full():
                            try:
                                self.segmentation_queue.get_nowait()
                            except Empty:
                                pass
                                
                        self.segmentation_queue.put(segmentation_data)
                        self.current_segmentation = segmentation_data
                        
                        # Immediately broadcast to connected WebSocket clients for smooth display
                        self._broadcast_segmentation_update(segmentation_data)
                        
                        if self.debug_mode and (time.time() - self.last_debug_time) > self.debug_interval:
                            print(f"✅ Segmentation completed for frame {self.frame_counter}")
                        
                    except Exception as e:
                        print(f"❌ Error processing segmentation: {e}")
                
                self.frame_counter += 1
                
            except Empty:
                # No frame available, continue loop
                continue
            except Exception as e:
                print(f"❌ Error in processing loop: {e}")
                
    def _broadcast_segmentation_update(self, segmentation_data):
        """Immediately broadcast segmentation update to connected WebSocket clients"""
        try:
            # Only broadcast to main UI for smooth display
            if self.main_ui_connected and self.socketio:
                display_data = self.get_synchronized_display(for_main_ui=True)
                state = self.get_current_state()
                response_data = {**display_data, 'queue_size': state['queue_size']}
                
                # Use socketio to broadcast to all connected clients
                self.socketio.emit('frame_update', response_data)
                
                if self.debug_mode and (time.time() - self.last_debug_time) > self.debug_interval:
                    print(f"📡 Broadcasted segmentation update for frame {self.frame_counter}")
        except Exception as e:
            if self.debug_mode:
                print(f"❌ Error broadcasting update: {e}")
    
    def _create_segmentation_overlay_optimized(self, frame, result):
        """Create an optimized visualization overlay for the segmentation result"""
        try:
            segmentation_map = result.segmentation_map
            
            # Occasional debug info (not every frame)
            if self.debug_mode and (time.time() - self.last_debug_time) > self.debug_interval:
                unique_classes = np.unique(segmentation_map)
                print(f"🔍 Classes: {unique_classes}, Shape: {segmentation_map.shape}")
                
                # Quick road detection check
                road_pixels = np.sum(segmentation_map == 0)
                road_percentage = (road_pixels / segmentation_map.size) * 100
                print(f"🛣️ Road: {road_percentage:.1f}% of image")
            
            # Vectorized color mapping using pre-computed lookup table
            overlay = self.color_mapping_array[segmentation_map]
            
            # Resize overlay to match original frame size if needed
            if overlay.shape[:2] != frame.shape[:2]:
                overlay = cv2.resize(overlay, (frame.shape[1], frame.shape[0]), 
                                   interpolation=cv2.INTER_NEAREST)  # Use nearest neighbor for segmentation
            
            # Optimized blending (reduced alpha for better performance)
            blended = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)
            
            # Convert back to BGR for encoding
            blended = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
            
            return blended
            
        except Exception as e:
            print(f"❌ Error creating segmentation overlay: {e}")
            return frame
    
    def _create_segmentation_overlay(self, frame, result):
        """Legacy function - redirects to optimized version"""
        return self._create_segmentation_overlay_optimized(frame, result)
    
    def add_frame(self, frame, frame_id=None, timestamp=None):
        """Add a frame to the processing queue"""
        if timestamp is None:
            timestamp = time.time()
            
        if frame_id is None:
            frame_id = f"frame_{self.frame_counter}"
            
        frame_data = {
            'frame': frame,
            'frame_id': frame_id,
            'timestamp': timestamp
        }
        
        # Add to queue (remove old frame if full)
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                pass
                
        self.frame_queue.put(frame_data)
        
    def get_current_state(self):
        """Get current processing state for display"""
        return {
            'frame_counter': self.frame_counter,
            'current_frame_available': self.current_frame is not None,
            'current_segmentation_available': self.current_segmentation is not None,
            'processing_interval': self.segmentation_interval,
            'queue_size': self.frame_queue.qsize()
        }
    
    def get_synchronized_display(self, for_main_ui=True):
        """Get synchronized frame and segmentation data for display - OPTIMIZED"""
        display_data = {
            'original_frame': None,
            'segmentation_overlay': None,
            'segmentation_info': None,
            'frame_counter': self.frame_counter,
            'timestamp': time.time()
        }
        
        # Only provide segmentation overlay to main UI to avoid conflicts
        if self.current_segmentation is not None and for_main_ui:
            seg_data = self.current_segmentation
            
            # Check if this segmentation is recent enough (within last 10 frames)
            frame_diff = self.frame_counter - seg_data['frame_counter']
            
            if frame_diff <= 10:  # Only send if recent
                # Fast encoding with optimized parameters
                _, buffer = cv2.imencode('.jpg', seg_data['overlay'], self.encode_params)
                overlay_b64 = base64.b64encode(buffer).decode('utf-8')
                display_data['segmentation_overlay'] = f"data:image/jpeg;base64,{overlay_b64}"
                
                # Minimal segmentation info
                display_data['segmentation_info'] = {
                    'frame_id': seg_data['frame_id'],
                    'timestamp': seg_data['timestamp'],
                    'frame_counter': seg_data['frame_counter']
                }
        
        # For status page, provide basic info without heavy data
        elif not for_main_ui and self.current_segmentation is not None:
            seg_data = self.current_segmentation
            display_data['segmentation_info'] = {
                'frame_id': seg_data['frame_id'],
                'frame_counter': seg_data['frame_counter'],
                'frames_since_segmentation': self.frame_counter - seg_data['frame_counter']
            }
        
        return display_data
    
    def shutdown(self):
        """Shutdown the processor"""
        print("🛑 Shutting down video processor...")
        self.frame_queue.put(None)  # Shutdown signal
        if self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
    
    def enable_debug_mode(self, enable=True):
        """Enable or disable debug mode for verbose logging"""
        self.debug_mode = enable
        if enable:
            print("🐛 Debug mode enabled - verbose logging activated")
        else:
            print("🔇 Debug mode disabled - minimal logging activated")
    
    def set_main_ui_connected(self, connected=True):
        """Mark main UI as connected/disconnected to prioritize it over status page"""
        self.main_ui_connected = connected
        if connected:
            print("🎯 Main UI connected - prioritizing segmentation data for main interface")
        else:
            print("📄 Main UI disconnected - status page can receive data")

# Initialize Flask app and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'video_processing_secret'
CORS(app)  # Enable CORS for all routes
socketio = SocketIO(app, cors_allowed_origins="*")

# Additional CORS headers for all routes
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Global processor instance - pass socketio for real-time broadcasting
processor = VideoProcessor(socketio_instance=socketio)

@app.route('/')
def index():
    """Serve a simple test page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Video Processor Status</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #2b2b2b; color: white; }
            .status { background: #1e1e1e; padding: 20px; border-radius: 10px; margin: 20px 0; }
            .frame-display { display: flex; gap: 20px; }
            .frame-container { flex: 1; text-align: center; }
            .frame-container img { max-width: 100%; border-radius: 5px; }
            .info { background: #333; padding: 10px; border-radius: 5px; margin: 10px 0; }
        </style>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    </head>
    <body>
        <h1>🎥 Video Processor Status</h1>
        <div class="status" id="status">
            <h2>Status: Waiting for connection...</h2>
            <p>Frame Counter: <span id="frameCounter">0</span></p>
            <p>Queue Size: <span id="queueSize">0</span></p>
            <p>Processing Interval: <span id="interval">5</span> frames</p>
        </div>
        
        <div class="frame-display">
            <div class="frame-container">
                <h3>Original Video</h3>
                <img id="originalFrame" src="" alt="Original frame will appear here" style="display: none;">
                <div id="noOriginal">No frame received yet</div>
            </div>
            <div class="frame-container">
                <h3>Segmentation Overlay</h3>
                <img id="segmentationFrame" src="" alt="Segmentation will appear here" style="display: none;">
                <div id="noSegmentation">No segmentation available yet</div>
            </div>
        </div>
        
        <div class="info" id="segmentationInfo" style="display: none;">
            <h3>Segmentation Information</h3>
            <p>Frame ID: <span id="segFrameId">-</span></p>
            <p>Frames since last segmentation: <span id="framesSince">-</span></p>
            <p>Detected classes: <span id="detectedClasses">-</span></p>
        </div>

        <script>
            const socket = io();
            
            socket.on('connect', function() {
                console.log('Connected to video processor');
                document.querySelector('.status h2').textContent = 'Status: Connected';
            });
            
            socket.on('frame_update', function(data) {
                // Update status
                document.getElementById('frameCounter').textContent = data.frame_counter;
                document.getElementById('queueSize').textContent = data.queue_size || 0;
                
                // Update original frame
                if (data.original_frame) {
                    const originalImg = document.getElementById('originalFrame');
                    originalImg.src = data.original_frame;
                    originalImg.style.display = 'block';
                    document.getElementById('noOriginal').style.display = 'none';
                }
                
                // Update segmentation overlay
                if (data.segmentation_overlay) {
                    const segImg = document.getElementById('segmentationFrame');
                    segImg.src = data.segmentation_overlay;
                    segImg.style.display = 'block';
                    document.getElementById('noSegmentation').style.display = 'none';
                    
                    // Update segmentation info
                    if (data.segmentation_info) {
                        document.getElementById('segmentationInfo').style.display = 'block';
                        document.getElementById('segFrameId').textContent = data.segmentation_info.frame_id;
                        document.getElementById('framesSince').textContent = data.segmentation_info.frames_since_segmentation;
                        
                        const classes = data.segmentation_info.class_labels || [];
                        document.getElementById('detectedClasses').textContent = classes.join(', ') || 'None';
                    }
                } else {
                    document.getElementById('segmentationFrame').style.display = 'none';
                    document.getElementById('noSegmentation').style.display = 'block';
                    document.getElementById('segmentationInfo').style.display = 'none';
                }
            });
            
            // Request updates every 100ms
            setInterval(() => {
                socket.emit('request_update');
            }, 100);
            
            // Show status message when main UI is active
            socket.on('frame_update', function(data) {
                // Update status
                document.getElementById('frameCounter').textContent = data.frame_counter;
                document.getElementById('queueSize').textContent = data.queue_size || 0;
                
                // Check if segmentation data is being sent to main UI
                if (!data.segmentation_overlay) {
                    document.getElementById('noSegmentation').innerHTML = 
                        '<strong>🎯 Main UI Priority Mode</strong><br>Segmentation data is being sent to the main AI platform interface.<br>Close the main UI to see segmentation data here.';
                    document.getElementById('noOriginal').innerHTML = 
                        '<strong>📱 Monitoring Mode</strong><br>Video processing active for main interface.';
                } else {
                    // Original behavior for when main UI is not connected
                    if (data.original_frame) {
                        const originalImg = document.getElementById('originalFrame');
                        originalImg.src = data.original_frame;
                        originalImg.style.display = 'block';
                        document.getElementById('noOriginal').style.display = 'none';
                    }
                    
                    if (data.segmentation_overlay) {
                        const segImg = document.getElementById('segmentationFrame');
                        segImg.src = data.segmentation_overlay;
                        segImg.style.display = 'block';
                        document.getElementById('noSegmentation').style.display = 'none';
                        
                        if (data.segmentation_info) {
                            document.getElementById('segmentationInfo').style.display = 'block';
                            document.getElementById('segFrameId').textContent = data.segmentation_info.frame_id;
                            document.getElementById('framesSince').textContent = data.segmentation_info.frames_since_segmentation;
                            
                            const classes = data.segmentation_info.class_labels || [];
                            document.getElementById('detectedClasses').textContent = classes.join(', ') || 'None';
                        }
                    }
                }
            });
        </script>
    </body>
    </html>
    """

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
        
        processor.add_frame(frame, frame_id, timestamp)
        
        # Get current state
        state = processor.get_current_state()
        
        return jsonify({
            'success': True,
            'frame_counter': state['frame_counter'],
            'queue_size': state['queue_size'],
            'message': 'Frame processed successfully'
        })
        
    except Exception as e:
        print(f"❌ Error processing frame: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_display', methods=['GET'])
def get_display():
    """Get synchronized display data - prioritized for main UI"""
    try:
        # Mark main UI as connected when it requests data
        processor.set_main_ui_connected(True)
        display_data = processor.get_synchronized_display(for_main_ui=True)
        return jsonify(display_data)
    except Exception as e:
        print(f"❌ Error getting display data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get processor status"""
    try:
        state = processor.get_current_state()
        return jsonify(state)
    except Exception as e:
        print(f"❌ Error getting status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug/<action>', methods=['POST'])
def toggle_debug(action):
    """Toggle debug mode for performance monitoring"""
    try:
        if action == 'enable':
            processor.enable_debug_mode(True)
            return jsonify({'success': True, 'debug_mode': True})
        elif action == 'disable':
            processor.enable_debug_mode(False)
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
        emit('frame_update', response_data)
        
        # Debug logging (only when enabled)
        if processor.debug_mode:
            has_overlay = 'segmentation_overlay' in response_data and response_data['segmentation_overlay'] is not None
            client_type = "Main UI" if is_main_ui else "Status Page"
            print(f"📡 Update sent to {client_type} - Frame: {response_data.get('frame_counter', 0)}, "
                  f"Has overlay: {has_overlay}, Queue: {response_data.get('queue_size', 0)}")
        
    except Exception as e:
        print(f"❌ Error handling update request: {e}")
        emit('error', {'message': str(e)})

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    # Determine if this is status page or main UI based on referrer
    referrer = request.headers.get('Referer', '')
    if '127.0.0.1:5000' in referrer and 'modules/Platform' not in referrer:
        # This is the status page
        processor.status_page_clients.add(request.sid)
        print(f"📄 Status page connected: {request.sid}")
    else:
        # This is the main UI
        print(f"🎯 Main UI connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    if request.sid in processor.status_page_clients:
        processor.status_page_clients.remove(request.sid)
        print(f"📄 Status page disconnected: {request.sid}")
    else:
        # Check if any main UI clients are still connected
        # If not, mark main UI as disconnected
        print(f"🎯 Main UI disconnected: {request.sid}")
        # In a simple case, assume main UI is disconnected
        processor.set_main_ui_connected(False)

def run_processor_server(host='0.0.0.0', port=5000, debug=False):
    """Run the processor server"""
    print(f"🚀 Starting Video Processor Server on {host}:{port}")
    print(f"📊 Processing every {processor.segmentation_interval} frames for optimal performance")
    print(f"🌐 Web interface available at: http://{host}:{port}")
    print(f"📡 API endpoints:")
    print(f"   - POST /api/process_frame - Send frame data")
    print(f"   - GET  /api/get_display  - Get synchronized display")
    print(f"   - GET  /api/status       - Get processor status")
    print(f"   - POST /api/debug/enable - Enable verbose debug logging")
    print(f"   - POST /api/debug/disable - Disable debug logging for performance")
    print(f"🚀 Performance Mode: Debug logging {'ON' if processor.debug_mode else 'OFF'}")
    print(f"⚡ Optimizations: Reduced queues, vectorized color mapping, throttled updates")
    
    try:
        socketio.run(app, host=host, port=port, debug=debug)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down server...")
        processor.shutdown()
    except Exception as e:
        print(f"❌ Server error: {e}")
        processor.shutdown()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Video Processing Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (use 0.0.0.0 for LAN/mobile access)')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--interval', type=int, default=5, help='Segmentation processing interval (frames)')
    
    args = parser.parse_args()
    
    # Update processing interval if specified
    if args.interval != 5:
        processor.segmentation_interval = args.interval
        print(f"🔄 Updated segmentation interval to {args.interval} frames")
    
    # Set debug mode based on argument
    if args.debug:
        processor.enable_debug_mode(True)
        print("🐛 Debug mode enabled via command line")
    
    run_processor_server(args.host, args.port, args.debug)
