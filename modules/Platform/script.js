/**
 * AI Music Generation Platform - Main JavaScript Module
 * Handles video input sources, ROI drawing, and music generation
 * 
 * PERFORMANCE OPTIMIZATIONS (2025):
 * - Reduced frame send interval to 150ms for better responsiveness
 * - Added throttling to prevent concurrent frame processing
 * - Optimized canvas operations and reduced logging
 * - Implemented update throttling for smoother display
 * 
 * For monitoring performance, use: python performance_monitor.py
 */

// Global variables
let inputSource = null;
let videoElement = null;
let canvas = null;
let ctx = null;
let segmentationCanvas = null;
let segmentationCtx = null;
let roiPoints = []; // Will be initialized based on video dimensions
let controlPoints = []; // Bézier control points for curves
let draggingPoint = null;
let draggingControl = null;
let scale = {x: 1, y: 1};
let offset = {x: 0, y: 0};
let isPaused = false;
let settings = {};
let showControlPoints = true;

// Frame processing variables
let frameProcessingEnabled = true; // Always keep processing enabled
let segmentationDisplayEnabled = false; // Only control display - Start with segmentation display OFF

// Dynamic processor URL detection for mobile/desktop compatibility
let processorUrl = detectProcessorUrl();
let frameCounter = 0;
let lastFrameSentTime = 0;
// Adapt frame send rate on mobile to reduce bandwidth/CPU contention
let frameSendInterval = isMobileDevice() ? 250 : 150; // ms
let processingCanvas = null;
let processingCtx = null;
let segmentationSocket = null;
let currentSegmentationOverlay = null;
let currentSegmentationInfo = null;
// Prevent stale/out-of-order overlays from replacing newer ones on mobile
let latestOverlayFrameCounter = -1;
let drawToken = 0;

// Performance optimization variables
let isProcessingFrame = false; // Prevent concurrent frame processing
let lastUpdateTime = 0;
let updateThrottleInterval = 50; // Throttle updates to 50ms (20 FPS)

// Audio system variables
let audioContext = null;
let masterGain = null;
let isMusicGenerationActive = false;
let activeNotes = new Map(); // Track currently playing notes
let instrumentVoices = {}; // Store instrument voice settings
let musicEventQueue = []; // Queue for scheduling music events
let lastMusicEventTime = 0;

// Color scheme
const colors = {
    bg: '#2b2b2b',
    menu: '#1e1e1e',
    button: '#4a4a4a',
    accent: '#00ff88',
    text: '#ffffff'
};

/**
 * Device Detection
 */
function isMobileDevice() {
    return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent) || 
           (navigator.maxTouchPoints && navigator.maxTouchPoints > 2);
}

/**
 * Processor URL Detection for Mobile/Desktop Compatibility
 */
function detectProcessorUrl() {
    // Resolve backend URL based on where UI is loaded from.
    const currentHost = window.location.hostname;
    // If UI is opened via file:// or without a hostname, fall back to localhost.
    const isLocalhost = currentHost === 'localhost' || currentHost === '127.0.0.1' || currentHost === '';
    const baseHost = isLocalhost ? '127.0.0.1' : currentHost;
    const url = `http://${baseHost}:5000`;
    console.log(`🌐 Using processor URL: ${url} (page host: ${currentHost || 'file://'})`);
    return url;
}

/**
 * Application Initialization
 */
window.onload = function() {
    setupEventListeners();
    showInputSelection();
};

/**
 * Event Listeners Setup
 */
function setupEventListeners() {
    // Canvas mouse events
    document.addEventListener('mousedown', onCanvasClick);
    document.addEventListener('mousemove', onCanvasMove);
    document.addEventListener('mouseup', onCanvasRelease);
    
    // Canvas touch events for mobile
    document.addEventListener('touchstart', onCanvasTouch);
    document.addEventListener('touchmove', onCanvasTouchMove);
    document.addEventListener('touchend', onCanvasTouchEnd);
    
    // Window resize
    window.addEventListener('resize', onWindowResize);
    
    // Video file input
    document.getElementById('videoFileInput').addEventListener('change', handleVideoFile);
    
    // Mobile-specific setup for segmentation button
    setupMobileButtons();
    
    // Initialize frame processing
    initializeFrameProcessing();
}

function setupMobileButtons() {
    if (isMobileDevice()) {
        // Add additional touch event handling for segmentation button
        document.addEventListener('DOMContentLoaded', function() {
            setupSegmentationButtonMobile();
        });
    }
}

function setupSegmentationButtonMobile() {
    if (isMobileDevice()) {
        const segButton = document.getElementById('toggleSegmentationBtn');
        if (segButton) {
            console.log('Setting up mobile touch events for segmentation button');
            
            // Remove ALL existing event listeners to avoid conflicts
            segButton.removeEventListener('touchstart', handleSegButtonTouchStart);
            segButton.removeEventListener('touchend', handleSegButtonTouchEnd);
            segButton.removeEventListener('click', handleSegButtonClick);
            
            // Add new event listeners
            segButton.addEventListener('touchstart', handleSegButtonTouchStart, { passive: false });
            segButton.addEventListener('touchend', handleSegButtonTouchEnd, { passive: false });
            
            // Prevent click events on mobile to avoid double triggering
            segButton.addEventListener('click', handleSegButtonClick, { passive: false });
            
            // Remove the onclick attribute to prevent conflicts
            segButton.removeAttribute('onclick');
            segButton.removeAttribute('ontouchend');
            
            // Start monitoring button state to prevent reversion
            startButtonStateMonitoring();
            
        } else {
            console.warn('Segmentation button not found for mobile setup');
        }
    }
}

// Add a flag to prevent double triggering
let isSegmentationButtonPressed = false;

function handleSegButtonTouchStart(e) {
    e.preventDefault();
    e.stopPropagation();
    this.style.transform = 'scale(0.95)';
    this.style.opacity = '0.8';
}

function handleSegButtonTouchEnd(e) {
    e.preventDefault();
    e.stopPropagation();
    this.style.transform = 'scale(1)';
    this.style.opacity = '1';
    
    // Prevent double triggering
    if (isSegmentationButtonPressed) return;
    isSegmentationButtonPressed = true;
    
    // Call the toggle function with a slight delay
    setTimeout(() => {
        console.log('Mobile touch triggered segmentation toggle');
        toggleFrameProcessing();
        isSegmentationButtonPressed = false;
    }, 100);
}

function handleSegButtonClick(e) {
    if (isMobileDevice()) {
        // On mobile, prevent click events to avoid double triggering
        e.preventDefault();
        e.stopPropagation();
        return false;
    } else {
        // On desktop, allow normal click behavior
        if (isSegmentationButtonPressed) return;
        isSegmentationButtonPressed = true;
        
        setTimeout(() => {
            toggleFrameProcessing();
            isSegmentationButtonPressed = false;
        }, 50);
    }
}

/**
 * Frame Processing Functions
 */
function initializeFrameProcessing() {
    console.log('🔄 Initializing frame processing...');
    
    // Create processing canvas (hidden, used for frame capture)
    processingCanvas = document.createElement('canvas');
    processingCtx = processingCanvas.getContext('2d');
    
    // Always start processing automatically in the background
    connectToProcessor();
    
    console.log('✅ Frame processing initialized - background processing will start automatically');
}

function connectToProcessor() {
    // First check if processor is running
    checkProcessorStatus()
        .then(() => {
            console.log('🔗 Connecting to segmentation processor...');
            updateSegmentationStatus('Connecting...');
            
            // Load Socket.IO library if not already loaded
            if (typeof io === 'undefined') {
                const script = document.createElement('script');
                script.src = 'https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js';
                script.onload = () => initializeSocketConnection();
                script.onerror = () => {
                    console.error('❌ Failed to load Socket.IO library');
                    updateSegmentationStatus('Socket.IO load failed');
                };
                document.head.appendChild(script);
            } else {
                initializeSocketConnection();
            }
        })
        .catch(error => {
            console.warn('⚠️ Processor not available:', error);
            updateStatus('Processor offline - Run: python modules/Platform/processor.py');
            updateSegmentationStatus('Offline - Start processor.py');
            
            const statusDiv = document.getElementById('segmentationStatus');
            if (statusDiv) {
                statusDiv.textContent = 'Processor offline - Start processor.py first';
            }
        });
}

function initializeSocketConnection() {
    try {
        console.log(`🔄 Attempting to connect to processor at: ${processorUrl}`);
        segmentationSocket = io(processorUrl, {
            timeout: 10000,
            reconnection: true,
            reconnectionAttempts: 5,
            reconnectionDelay: 2000,
            transports: ['websocket', 'polling'] // Allow fallback to polling
        });
        
        segmentationSocket.on('connect', function() {
            console.log('✅ Connected to segmentation processor');
            updateStatus('Processor connected - Background segmentation processing active');
            updateSegmentationStatus('Connected');
            
            // Maintain button state after connection
            updateSegmentationButtonState();
            
            // Start requesting updates
            startRequestingUpdates();
        });
        
        segmentationSocket.on('disconnect', function() {
            console.log('⚠️ Disconnected from segmentation processor');
            updateStatus('Processor disconnected');
            updateSegmentationStatus('Disconnected');
        });
        
        segmentationSocket.on('frame_update', function(data) {
            updateSegmentationDisplay(data);
        });
        
        segmentationSocket.on('music_update', function(musicData) {
            if (isMusicGenerationActive) {
                handleMusicEvents(musicData);
            }
        });
        
        segmentationSocket.on('connect_error', function(error) {
            console.error('❌ Connection error:', error);
            updateStatus('Segmentation Error: Connection Error - Check CORS');
            updateSegmentationStatus('Connection Error - Check CORS');
            
            // Try alternative URLs if available
            tryAlternativeConnections();
        });
        
        segmentationSocket.on('error', function(error) {
            console.error('❌ Socket error:', error);
            updateStatus('Processor error: ' + error.message);
            updateSegmentationStatus('Error');
        });
        
    } catch (error) {
        console.error('❌ Failed to initialize socket connection:', error);
        updateStatus('Connection failed');
        updateSegmentationStatus('Connection Failed');
    }
}

function tryAlternativeConnections() {
    const host = window.location.hostname;
    const candidates = new Set([
        processorUrl,
        'http://127.0.0.1:5000',
        'http://localhost:5000',
        host ? `http://${host}:5000` : null
    ].filter(Boolean));
    const alternativeUrls = Array.from(candidates).filter(url => url !== processorUrl);
    
    console.log('🔄 Trying alternative processor URLs:', alternativeUrls);
    
    // Try each alternative URL
    alternativeUrls.forEach((url, index) => {
        setTimeout(() => {
            console.log(`🔄 Trying alternative URL: ${url}`);
            fetch(`${url}/api/status`, { mode: 'cors' })
                .then(response => {
                    if (response.ok) {
                        console.log(`✅ Found working processor at: ${url}`);
                        processorUrl = url;
                        
                        // Disconnect current socket and reconnect to working URL
                        if (segmentationSocket) {
                            segmentationSocket.disconnect();
                        }
                        initializeSocketConnection();
                    }
                })
                .catch(err => {
                    console.log(`❌ ${url} not reachable:`, err.message);
                });
        }, index * 1000);
    });
}

function requestImmediateUpdate() {
    /**
     * Force an immediate update request to get the latest segmentation data
     * Useful when toggling segmentation view ON
     */
    if (segmentationSocket && segmentationSocket.connected) {
        console.log('🔄 Requesting immediate segmentation update');
        segmentationSocket.emit('request_update');
    }
}

function startRequestingUpdates() {
    const FAST = 100; // base interval
    const SLOW = 300; // when display is OFF
    let lastSlowEmit = 0;
    setInterval(() => {
        if (!(segmentationSocket && segmentationSocket.connected)) return;
        const now = Date.now();
        if (segmentationDisplayEnabled) {
            segmentationSocket.emit('request_update');
        } else if (now - lastSlowEmit >= SLOW) {
            segmentationSocket.emit('request_update');
            lastSlowEmit = now;
        }
    }, FAST);
}

async function checkProcessorStatus() {
    const response = await fetch(`${processorUrl}/api/status`, { mode: 'cors' });
    if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
    }
    return response.json();
}

/**
 * Audio System Functions
 */
function initializeAudioSystem() {
    try {
        // Create audio context
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        audioContext = new AudioContextClass();
        
        // Create master gain node
        masterGain = audioContext.createGain();
        masterGain.gain.setValueAtTime(0.3, audioContext.currentTime); // Set volume to 30%
        masterGain.connect(audioContext.destination);
        
        // Initialize instrument voices
        initializeInstrumentVoices();
        
        console.log('🎵 Audio system initialized successfully');
        updateStatus('Audio system ready');
        
    } catch (error) {
        console.error('❌ Failed to initialize audio system:', error);
        updateStatus('Audio initialization failed');
    }
}

function initializeInstrumentVoices() {
    instrumentVoices = {
        piano: {
            waveform: 'sawtooth',
            attack: 0.01,
            decay: 0.3,
            sustain: 0.3,
            release: 1.0,
            filterFreq: 2000
        },
        electric_piano: {
            waveform: 'square',
            attack: 0.01,
            decay: 0.2,
            sustain: 0.4,
            release: 0.8,
            filterFreq: 1500
        },
        drums: {
            waveform: 'noise',
            attack: 0.001,
            decay: 0.1,
            sustain: 0.0,
            release: 0.2,
            filterFreq: 100
        },
        bass: {
            waveform: 'triangle',
            attack: 0.02,
            decay: 0.4,
            sustain: 0.6,
            release: 1.2,
            filterFreq: 400
        },
        strings: {
            waveform: 'sawtooth',
            attack: 0.1,
            decay: 0.2,
            sustain: 0.8,
            release: 1.5,
            filterFreq: 3000
        },
        electric_guitar: {
            waveform: 'square',
            attack: 0.005,
            decay: 0.1,
            sustain: 0.5,
            release: 0.6,
            filterFreq: 2500
        },
        acoustic_guitar: {
            waveform: 'sawtooth',
            attack: 0.01,
            decay: 0.3,
            sustain: 0.4,
            release: 1.0,
            filterFreq: 2000
        },
        pad: {
            waveform: 'sine',
            attack: 0.3,
            decay: 0.5,
            sustain: 0.7,
            release: 2.0,
            filterFreq: 1000
        },
        synth: {
            waveform: 'square',
            attack: 0.01,
            decay: 0.2,
            sustain: 0.3,
            release: 0.5,
            filterFreq: 1800
        }
    };
}

function handleMusicEvents(musicData) {
    try {
        if (!musicData || !musicData.events) {
            return;
        }
        
        console.log(`🎵 Received ${musicData.events.length} music events for frame ${musicData.frame_counter}`);
        
        // Schedule each music event
        musicData.events.forEach((event, index) => {
            // Slight delay between events to avoid overwhelming
            const scheduleTime = audioContext.currentTime + (index * 0.01);
            playMusicEvent(event, scheduleTime);
        });
        
        // Update UI with music info
        updateMusicInfo(musicData);
        
    } catch (error) {
        console.error('❌ Error handling music events:', error);
    }
}

function playMusicEvent(event, scheduleTime) {
    try {
        const frequency = midiNoteToFrequency(event.note);
        const instrument = event.instrument || 'synth';
        const voice = instrumentVoices[instrument] || instrumentVoices.synth;
        
        // Create oscillator and envelope for the note
        if (instrument === 'drums') {
            playDrumSound(event, scheduleTime);
        } else {
            playTonalInstrument(event, frequency, voice, scheduleTime);
        }
        
    } catch (error) {
        console.error('❌ Error playing music event:', error);
    }
}

function playTonalInstrument(event, frequency, voice, scheduleTime) {
    // Create oscillator
    const osc = audioContext.createOscillator();
    osc.type = voice.waveform;
    osc.frequency.setValueAtTime(frequency, scheduleTime);
    
    // Create gain envelope
    const gainNode = audioContext.createGain();
    const velocity = event.velocity / 127; // Convert MIDI velocity to 0-1
    const duration = Math.min(event.duration, 3.0); // Cap duration at 3 seconds
    
    // ADSR envelope
    gainNode.gain.setValueAtTime(0, scheduleTime);
    gainNode.gain.linearRampToValueAtTime(velocity * 0.8, scheduleTime + voice.attack);
    gainNode.gain.linearRampToValueAtTime(velocity * voice.sustain, scheduleTime + voice.attack + voice.decay);
    gainNode.gain.linearRampToValueAtTime(0, scheduleTime + duration);
    
    // Create filter for timbre
    const filter = audioContext.createBiquadFilter();
    filter.type = 'lowpass';
    filter.frequency.setValueAtTime(voice.filterFreq, scheduleTime);
    
    // Connect audio graph
    osc.connect(filter);
    filter.connect(gainNode);
    gainNode.connect(masterGain);
    
    // Schedule playback
    osc.start(scheduleTime);
    osc.stop(scheduleTime + duration);
    
    // Track active note for cleanup
    const noteKey = `${event.note}-${scheduleTime}`;
    activeNotes.set(noteKey, { osc, gainNode, filter });
    
    // Clean up after playback
    setTimeout(() => {
        activeNotes.delete(noteKey);
    }, (duration + 0.1) * 1000);
}

function playDrumSound(event, scheduleTime) {
    // Create noise buffer for drum sounds
    const bufferSize = 2 * audioContext.sampleRate;
    const noiseBuffer = audioContext.createBuffer(1, bufferSize, audioContext.sampleRate);
    const output = noiseBuffer.getChannelData(0);
    
    // Generate noise
    for (let i = 0; i < bufferSize; i++) {
        output[i] = Math.random() * 2 - 1;
    }
    
    // Create buffer source
    const noise = audioContext.createBufferSource();
    noise.buffer = noiseBuffer;
    
    // Create gain and filter for drum character
    const gainNode = audioContext.createGain();
    const filter = audioContext.createBiquadFilter();
    
    // Configure based on drum type (kick, snare, etc.)
    const velocity = event.velocity / 127;
    const drumType = getDrumType(event.note);
    
    switch (drumType) {
        case 'kick':
            filter.type = 'lowpass';
            filter.frequency.setValueAtTime(100, scheduleTime);
            break;
        case 'snare':
            filter.type = 'highpass';
            filter.frequency.setValueAtTime(200, scheduleTime);
            break;
        default:
            filter.type = 'bandpass';
            filter.frequency.setValueAtTime(1000, scheduleTime);
    }
    
    // Envelope
    gainNode.gain.setValueAtTime(0, scheduleTime);
    gainNode.gain.linearRampToValueAtTime(velocity * 0.6, scheduleTime + 0.001);
    gainNode.gain.exponentialRampToValueAtTime(0.001, scheduleTime + 0.2);
    
    // Connect and play
    noise.connect(filter);
    filter.connect(gainNode);
    gainNode.connect(masterGain);
    
    noise.start(scheduleTime);
    noise.stop(scheduleTime + 0.2);
}

function midiNoteToFrequency(note) {
    // Convert MIDI note number to frequency
    return 440 * Math.pow(2, (note - 69) / 12);
}

function getDrumType(midiNote) {
    // Standard MIDI drum mapping
    switch (midiNote) {
        case 36: return 'kick';
        case 38: case 40: return 'snare';
        case 42: case 44: return 'hihat';
        case 49: case 57: return 'crash';
        default: return 'generic';
    }
}

function stopAllActiveNotes() {
    try {
        activeNotes.forEach((noteData, key) => {
            try {
                if (noteData.gainNode) {
                    noteData.gainNode.gain.exponentialRampToValueAtTime(0.001, audioContext.currentTime + 0.1);
                }
            } catch (e) {
                // Note may have already ended
            }
        });
        activeNotes.clear();
        console.log('🔇 Stopped all active notes');
    } catch (error) {
        console.error('❌ Error stopping notes:', error);
    }
}

function updateMusicInfo(musicData) {
    // Update status with music information
    const eventCount = musicData.events.length;
    const tempo = musicData.tempo;
    const key = musicData.key_signature;
    
    updateStatus(`🎵 Playing: ${eventCount} events | ${tempo} BPM | ${key}`);
    
    // Update ROI info section with music data
    const roiInfo = document.getElementById('roiInfo');
    if (roiInfo && eventCount > 0) {
        const instruments = {};
        musicData.events.forEach(event => {
            const instr = event.instrument || 'unknown';
            instruments[instr] = (instruments[instr] || 0) + 1;
        });
        
        const instrText = Object.entries(instruments)
            .map(([instr, count]) => `${instr}: ${count}`)
            .join(', ');
        
        roiInfo.innerHTML = `
            <div style="color: #00ff88; font-size: 12px; margin-top: 5px;">
                <div>🎵 Music: ${eventCount} events</div>
                <div>${instrText}</div>
            </div>
        `;
    }
}

function captureAndSendFrame() {
    // Prevent concurrent frame processing
    if (isProcessingFrame || !videoElement || !videoElement.videoWidth) {
        return;
    }
    
    const now = Date.now();
    if (now - lastFrameSentTime < frameSendInterval) {
        return; // Rate limiting
    }
    
    isProcessingFrame = true; // Set processing flag
    
    try {
        const srcW = videoElement.videoWidth;
        const srcH = videoElement.videoHeight;
        const maxW = isMobileDevice() ? 640 : srcW;
        const scale = Math.min(1, maxW / Math.max(1, srcW));
        const targetW = Math.max(1, Math.round(srcW * scale));
        const targetH = Math.max(1, Math.round(srcH * scale));
        
        if (processingCanvas.width !== targetW || processingCanvas.height !== targetH) {
            processingCanvas.width = targetW;
            processingCanvas.height = targetH;
        }
        
        // Draw current video frame to processing canvas
        processingCtx.imageSmoothingEnabled = false;
        processingCtx.drawImage(videoElement, 0, 0, targetW, targetH);
        
        // Convert to base64 with optimized quality for speed
        const jpegQuality = isMobileDevice() ? 0.6 : 0.7;
        const frameData = processingCanvas.toDataURL('image/jpeg', jpegQuality);
        
        // Send frame to processor
        const frameInfo = {
            frame: frameData,
            frame_id: `frame_${frameCounter}`,
            timestamp: now / 1000,
            roi_points: roiPoints
        };
        
        // Reduced logging for performance - only log every 10th frame
        if (frameCounter % 10 === 0) {
            console.log(`📤 Sending frame ${frameCounter} to processor`);
        }
        
        // Send via HTTP (more reliable than WebSocket for large data)
        sendFrameToProcessor(frameInfo)
            .then(response => {
                if (response.success) {
                    frameCounter++;
                    lastFrameSentTime = now;
                    
                    // Update frame counter in display (throttled)
                    if (frameCounter % 5 === 0) {
                        updateFrameCounter(frameCounter);
                    }
                    
                    // Reduced logging for performance
                    if (frameCounter % 10 === 0) {
                        console.log(`✅ Frame ${frameCounter} processed successfully`);
                    }
                    updateSegmentationStatus('Processing');
                } else {
                    console.warn('⚠️ Frame processing failed:', response.error);
                }
            })
            .catch(error => {
                console.warn('⚠️ Frame send error:', error);
                updateSegmentationStatus('Send Error');
                
                // Check if it's a CORS or network error
                if (error.message.includes('Failed to fetch') || error.message.includes('CORS')) {
                    updateSegmentationStatus('Connection Error - Check CORS');
                    const statusDiv = document.getElementById('segmentationStatus');
                    if (statusDiv) {
                        statusDiv.textContent = 'Connection failed - Check processor is running';
                    }
                }
            })
            .finally(() => {
                isProcessingFrame = false; // Reset processing flag
            });
            
    } catch (error) {
        console.error('❌ Frame capture error:', error);
        updateSegmentationStatus('Capture Error');
        isProcessingFrame = false; // Reset processing flag
    }
}

async function sendFrameToProcessor(frameInfo) {
    const response = await fetch(`${processorUrl}/api/process_frame`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        mode: 'cors',
        body: JSON.stringify(frameInfo)
    });
    
    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    return response.json();
}

function updateSegmentationStatus(status) {
    // Log status for debugging
    console.log('🔗 Segmentation status:', status);
    
    // Update main status if it's important
    if (status === 'Connected') {
        updateStatus('Segmentation processor connected');
    } else if (status.includes('Error') || status.includes('Failed')) {
        updateStatus('Segmentation error: ' + status);
    } else if (status === 'Offline') {
        updateStatus('Segmentation processor offline');
    }
}

function updateSegmentationDisplay(data) {
    // Update frame counter - always update for smooth feedback
    if (data.frame_counter) {
        frameCounter = data.frame_counter;
        
        // Update frame counter display in UI
        const frameCounterEl = document.getElementById('frameCounter');
        if (frameCounterEl) {
            frameCounterEl.textContent = frameCounter;
        }
    }
    
    // Only accept newer overlays to avoid showing old frames later (out-of-order loads)
    if (data.segmentation_overlay) {
        const infoCounter = (data.segmentation_info && typeof data.segmentation_info.frame_counter === 'number')
            ? data.segmentation_info.frame_counter
            : (typeof data.frame_counter === 'number' ? data.frame_counter : 0);
        if (infoCounter > latestOverlayFrameCounter) {
            latestOverlayFrameCounter = infoCounter;
            currentSegmentationOverlay = data.segmentation_overlay;
            currentSegmentationInfo = data.segmentation_info || null;
            
            if (segmentationDisplayEnabled) {
                drawSegmentationOverlay();
                if (frameCounter % 30 === 0 && currentSegmentationInfo) {
                    console.log('🔍 Segmentation updated (frame', latestOverlayFrameCounter, ')');
                }
            }
        } else {
            // Ignore stale overlays arriving late
        }
    }
    
    // Handle case when display is disabled
    if (!segmentationDisplayEnabled) {
        // Clear segmentation overlay when display is disabled (but processing continues)
        clearSegmentationOverlay();
    }
}

function updateFrameCounter(count) {
    // This function is kept for backward compatibility
    // Frame counter is now updated directly in updateSegmentationDisplay
}

/**
 * Draw segmentation overlay on the segmentation canvas
 * Now displays ONLY the segmentation data (like processor.py)
 */
function drawSegmentationOverlay() {
    if (!segmentationCanvas || !segmentationCtx || !currentSegmentationOverlay) {
        return;
    }
    
    try {
        const thisToken = ++drawToken;
        const thisCounter = latestOverlayFrameCounter;
        // Create an image element to load the base64 segmentation data
        const img = new Image();
        img.onload = function() {
            // Drop if a newer image was queued after this started loading
            if (thisToken !== drawToken || thisCounter !== latestOverlayFrameCounter) return;
            // Clear the segmentation canvas
            segmentationCtx.clearRect(0, 0, segmentationCanvas.width, segmentationCanvas.height);
            
            // Calculate scaling to fit the canvas while maintaining aspect ratio
            const canvasAspect = segmentationCanvas.width / segmentationCanvas.height;
            const imgAspect = img.width / img.height;
            
            let drawWidth, drawHeight, drawX, drawY;
            
            if (imgAspect > canvasAspect) {
                // Image is wider, fit to width
                drawWidth = segmentationCanvas.width;
                drawHeight = segmentationCanvas.width / imgAspect;
                drawX = 0;
                drawY = (segmentationCanvas.height - drawHeight) / 2;
            } else {
                // Image is taller, fit to height
                drawHeight = segmentationCanvas.height;
                drawWidth = segmentationCanvas.height * imgAspect;
                drawX = (segmentationCanvas.width - drawWidth) / 2;
                drawY = 0;
            }
            
            // Display the segmentation overlay at full opacity (not blended)
            // This shows ONLY the segmentation data, similar to processor.py
            segmentationCtx.globalAlpha = 1.0;
            segmentationCtx.globalCompositeOperation = 'source-over';
            
            // Draw the segmentation overlay
            segmentationCtx.imageSmoothingEnabled = false;
            segmentationCtx.drawImage(img, drawX, drawY, drawWidth, drawHeight);
        };
        
        img.onerror = function() {
            console.error('❌ Failed to load segmentation overlay image');
        };
        
        // Load the base64 image
        img.src = currentSegmentationOverlay;
        
    } catch (error) {
        console.error('❌ Error drawing segmentation overlay:', error);
    }
}

/**
 * Clear the segmentation overlay
 */
function clearSegmentationOverlay() {
    if (segmentationCanvas && segmentationCtx) {
        segmentationCtx.clearRect(0, 0, segmentationCanvas.width, segmentationCanvas.height);
    }
    currentSegmentationOverlay = null;
}

function toggleFrameProcessing() {
    console.log('🎯 toggleSegmentationDisplay called, current state:', segmentationDisplayEnabled);
    
    segmentationDisplayEnabled = !segmentationDisplayEnabled;
    
    console.log('🎯 New display state:', segmentationDisplayEnabled);
    
    // Update button state immediately
    updateSegmentationButtonState();
    
    // Show/hide elements based on segmentation display state
    const videoElement = document.getElementById('videoElement');
    const roiCanvas = document.getElementById('roiCanvas');
    const segCanvas = document.getElementById('segmentationCanvas');
    const instructions = document.querySelector('.instructions');
    
    if (segmentationDisplayEnabled) {
        // Show segmentation overlay with ROI (hide video but keep ROI visible)
        if (videoElement) videoElement.style.display = 'none';
        if (roiCanvas) roiCanvas.style.display = 'block'; // Keep ROI visible
        if (segCanvas) {
            segCanvas.style.display = 'block';
            segCanvas.style.pointerEvents = 'none'; // Keep it non-interactive
        }
        if (instructions) {
            instructions.textContent = 'Displaying AI Segmentation Overlay with ROI • Drag points to adjust ROI • Toggle off to return to original video';
        }
        
        updateStatus('Showing segmentation overlay with ROI (processing continues)');
        console.log('✅ Segmentation DISPLAY: Overlay with ROI (background processing continues)');
        
        // Immediately draw any existing overlay data
        if (currentSegmentationOverlay) {
            console.log('🎯 Drawing existing segmentation overlay immediately');
            drawSegmentationOverlay();
        } else {
            console.log('⚠️ No segmentation overlay data available yet - waiting for next update');
            // Request immediate update instead of showing loading message
            if (segmentationSocket && segmentationSocket.connected) {
                requestImmediateUpdate();
            }
        }
        
        // Ensure processor connection (processing was already running)
        if (!segmentationSocket || !segmentationSocket.connected) {
            connectToProcessor();
        } else {
            // Request immediate update to get latest segmentation data
            requestImmediateUpdate();
        }
    } else {
        // Show original video with ROI (hide segmentation display, but processing continues)
        if (videoElement) videoElement.style.display = 'block';
        if (roiCanvas) roiCanvas.style.display = 'block';
        if (segCanvas) segCanvas.style.display = 'none';
        if (instructions) {
            instructions.textContent = 'Drag green points to adjust ROI corners • Drag red points to control edge curves (segmentation processing continues in background)';
        }
        
        updateStatus('Showing original video with ROI (segmentation processing continues in background)');
        console.log('❌ Segmentation DISPLAY: Hidden (background processing continues)');
        
        // Clear segmentation overlay display but keep processing
        clearSegmentationOverlay();
    }
}

function updateSegmentationButtonState() {
    const button = document.getElementById('toggleSegmentationBtn');
    
    if (button) {
        console.log('🎯 Updating button state. segmentationDisplayEnabled:', segmentationDisplayEnabled);
        
        if (segmentationDisplayEnabled) {
            button.textContent = '🔍 Segmentation View: ON';
            button.style.background = '#00ff88';
            button.style.color = 'black';
        } else {
            button.textContent = '🔍 Segmentation View: OFF';
            button.style.background = '#4a4a4a';
            button.style.color = 'white';
        }
        
        console.log('🎯 Button updated to:', button.textContent);
        
        // Force a DOM update to ensure the change sticks
        button.offsetHeight; // This forces a repaint
        
    } else {
        console.error('❌ Segmentation button not found!');
    }
}

// Add a periodic check to ensure button state stays correct
let buttonStateCheckInterval;

function startButtonStateMonitoring() {
    // Clear any existing interval
    if (buttonStateCheckInterval) {
        clearInterval(buttonStateCheckInterval);
    }
    
    // Check button state every 100ms to ensure it doesn't revert
    buttonStateCheckInterval = setInterval(() => {
        const button = document.getElementById('toggleSegmentationBtn');
        if (button) {
            const expectedText = segmentationDisplayEnabled ? '🔍 Segmentation View: ON' : '🔍 Segmentation View: OFF';
            if (button.textContent !== expectedText) {
                console.log('🔄 Button state reverted! Fixing it. Expected:', expectedText, 'Actual:', button.textContent);
                updateSegmentationButtonState();
            }
        }
    }, 100);
}

function stopButtonStateMonitoring() {
    if (buttonStateCheckInterval) {
        clearInterval(buttonStateCheckInterval);
        buttonStateCheckInterval = null;
    }
}

/**
 * Input Source Selection
 */
function showInputSelection() {
    const modal = document.getElementById('inputModal');
    modal.style.display = 'flex'; // Use flex instead of block for centering
    modal.classList.add('show'); // Add show class for better styling
    
    // Add mobile-specific event listeners for input buttons
    if (isMobileDevice()) {
        const inputButtons = document.querySelectorAll('.input-btn');
        inputButtons.forEach((button, index) => {
            // Remove any existing listeners
            button.removeEventListener('touchend', handleInputButtonTouch);
            
            // Add touch event listener
            button.addEventListener('touchend', handleInputButtonTouch, { passive: false });
        });
    }
}

function handleInputButtonTouch(event) {
    event.preventDefault();
    event.stopPropagation();
    
    const button = event.target;
    const onclick = button.getAttribute('onclick');
    
    if (onclick) {
        // Extract the source type from onclick attribute
        const match = onclick.match(/selectInputSource\('([^']+)'\)/);
        if (match) {
            const sourceType = match[1];
            selectInputSource(sourceType);
        }
    }
}

function selectInputSource(source) {
    inputSource = source;
    const modal = document.getElementById('inputModal');
    modal.style.display = 'none';
    modal.classList.remove('show');
    
    if (source === 'video_file') {
        // Clear any previous file selection to ensure change event fires
        const fileInput = document.getElementById('videoFileInput');
        fileInput.value = '';
        
        // Add a one-time event listener to handle file selection
        const handleFileSelection = (event) => {
            fileInput.removeEventListener('change', handleFileSelection);
            
            if (event.target.files.length === 0) {
                // User cancelled file selection, show input selection again
                console.log('File selection cancelled, showing input selection again');
                inputSource = null;
                showInputSelection();
            } else {
                // File was selected, proceed with normal handling
                handleVideoFile(event);
            }
        };
        
        fileInput.addEventListener('change', handleFileSelection);
        fileInput.click();
    } else if (source === 'network_stream') {
        showUrlInput();
    } else {
        setupMainInterface();
    }
}

/**
 * URL Input Modal Functions
 */
function showUrlInput() {
    document.getElementById('urlModal').style.display = 'flex';
}

function confirmUrl() {
    const url = document.getElementById('streamUrl').value.trim();
    if (url) {
        document.getElementById('urlModal').style.display = 'none';
        setupMainInterface();
    } else {
        alert('Please enter a valid URL');
    }
}

function cancelUrl() {
    inputSource = null;
    document.getElementById('urlModal').style.display = 'none';
    showInputSelection();
}

/**
 * Video File Handling
 */
function handleVideoFile(event) {
    const file = event.target.files[0];
    if (file) {
        console.log('Video file selected:', file.name);
        const url = URL.createObjectURL(file);
        
        // Setup main interface first
        setupMainInterface();
        
        // Then set the video source after interface is ready
        setTimeout(() => {
            videoElement = document.getElementById('videoElement');
            
            // Clear any existing camera stream
            if (videoElement.srcObject) {
                videoElement.srcObject.getTracks().forEach(track => track.stop());
                videoElement.srcObject = null;
            }
            
            videoElement.src = url;
            videoElement.load(); // Force video to load
            
            console.log('Video source set to:', url);
        }, 100);
    } else {
        console.log('No file selected in handleVideoFile');
        // Don't automatically show input selection here - it's handled in selectInputSource
    }
}

/**
 * Main Interface Setup
 */
function setupMainInterface() {
    // Show main interface
    document.getElementById('menuFrame').style.display = 'flex';
    document.getElementById('displayFrame').style.display = 'flex';
    
    // Update source info in the Change Source button
    const sourceNames = {
        'phone_camera': 'Camera',
        'video_file': 'Video File',
        'screen_record': 'Screen Record',
        'network_stream': 'Network Stream'
    };
    
    const sourceDisplayName = sourceNames[inputSource] || 'Unknown';
    document.getElementById('changeSourceBtn').textContent = `📂 Change Source (${sourceDisplayName})`;
    
    // Setup video display
    setupVideoDisplay();
    
    // Setup ROI canvas
    setupRoiCanvas();
    
    // Setup mobile-specific button handling (with delay to ensure DOM is ready)
    setTimeout(setupSegmentationButtonMobile, 100);
    
    // Start video capture
    startVideoCapture();
    
    updateStatus('Ready');
}

/**
 * Video Display Setup
 */
function setupVideoDisplay() {
    videoElement = document.getElementById('videoElement');
    
    // Mobile-specific video settings
    videoElement.setAttribute('playsinline', 'true');
    videoElement.setAttribute('webkit-playsinline', 'true');
    
    // Handle video load
    videoElement.addEventListener('loadedmetadata', function() {
        updateStatus('Video loaded successfully');
        console.log('Video dimensions:', videoElement.videoWidth, 'x', videoElement.videoHeight);
        
        // Reinitialize ROI points based on actual video dimensions
        initializeRoiPoints();
        
        updateVideoFeed();
    });
    
    videoElement.addEventListener('loadeddata', function() {
        console.log('Video data loaded');
        drawRoi();
    });
    
    videoElement.addEventListener('canplay', function() {
        console.log('Video can start playing');
        if (inputSource === 'video_file') {
            videoElement.play();
        }
    });
    
    videoElement.addEventListener('error', function(e) {
        console.error('Video error details:', e);
        updateStatus('Error loading video source');
    });
    
    // Add mobile debugging
    videoElement.addEventListener('loadstart', function() {
        console.log('Video load started');
        updateStatus('Starting video...');
    });
    
    videoElement.addEventListener('progress', function() {
        console.log('Video loading progress');
    });
}

/**
 * ROI Canvas Setup
 */
function setupRoiCanvas() {
    canvas = document.getElementById('roiCanvas');
    ctx = canvas.getContext('2d');
    
    // Setup segmentation canvas
    segmentationCanvas = document.getElementById('segmentationCanvas');
    segmentationCtx = segmentationCanvas.getContext('2d');
    
    // Set canvas size to match container
    const container = document.getElementById('videoContainer');
    canvas.width = container.offsetWidth;
    canvas.height = container.offsetHeight;
    segmentationCanvas.width = container.offsetWidth;
    segmentationCanvas.height = container.offsetHeight;
    
    // Initialize ROI points based on video/canvas dimensions
    initializeRoiPoints();
    
    // Start drawing ROI
    drawRoi();
    
    // Update segmentation button state
    updateSegmentationButtonState();
}

/**
 * ROI Point Initialization
 */
function initializeControlPoints() {
    // Create control points for each edge (2 control points per edge for quadratic Bézier curves)
    controlPoints = [];
    for (let i = 0; i < roiPoints.length; i++) {
        const current = roiPoints[i];
        const next = roiPoints[(i + 1) % roiPoints.length];
        
        // Calculate control points for this edge
        const midX = (current[0] + next[0]) / 2;
        const midY = (current[1] + next[1]) / 2;
        
        // Offset control points slightly to create initial curve
        const offset = 20;
        const perpX = -(next[1] - current[1]) / Math.sqrt((next[0] - current[0])**2 + (next[1] - current[1])**2) * offset;
        const perpY = (next[0] - current[0]) / Math.sqrt((next[0] - current[0])**2 + (next[1] - current[1])**2) * offset;
        
        controlPoints.push([midX + perpX, midY + perpY]);
    }
}

function initializeRoiPoints() {
    // Get video dimensions, fallback to canvas dimensions if video not loaded yet
    const videoWidth = videoElement.videoWidth || canvas.width || 640;
    const videoHeight = videoElement.videoHeight || canvas.height || 480;
    
    // Calculate ROI points as percentages of video dimensions
    // Create a rectangle that's 60% of the video size, centered
    const roiWidth = videoWidth * 0.6;
    const roiHeight = videoHeight * 0.6;
    const offsetX = (videoWidth - roiWidth) / 2;
    const offsetY = (videoHeight - roiHeight) / 2;
    
    roiPoints = [
        [offsetX, offsetY], // Top-left
        [offsetX + roiWidth, offsetY], // Top-right
        [offsetX + roiWidth, offsetY + roiHeight], // Bottom-right
        [offsetX, offsetY + roiHeight] // Bottom-left
    ];
    
    // Initialize control points after setting ROI points
    initializeControlPoints();
}

/**
 * Video Capture Management
 */
function startVideoCapture() {
    // Clear any existing sources first
    if (videoElement.srcObject) {
        videoElement.srcObject.getTracks().forEach(track => track.stop());
        videoElement.srcObject = null;
    }
    if (videoElement.src) {
        videoElement.src = '';
    }
    
    if (inputSource === 'phone_camera') {
        // Enhanced mobile camera constraints
        const constraints = {
            video: {
                facingMode: 'environment', // Use back camera on mobile
                width: { ideal: 1280, max: 1920 },
                height: { ideal: 720, max: 1080 },
                frameRate: { ideal: 30, max: 60 }
            }
        };
        
        navigator.mediaDevices.getUserMedia(constraints)
            .then(stream => {
                videoElement.srcObject = stream;
                updateStatus('Connected - Receiving camera feed');
                console.log('Camera stream started successfully');
            })
            .catch(err => {
                console.error('Camera error details:', err);
                updateStatus(`Error: Could not access camera - ${err.message}`);
                
                // Fallback: try with basic constraints
                navigator.mediaDevices.getUserMedia({ video: true })
                    .then(stream => {
                        videoElement.srcObject = stream;
                        updateStatus('Connected - Using fallback camera settings');
                    })
                    .catch(fallbackErr => {
                        console.error('Fallback camera error:', fallbackErr);
                        updateStatus('Error: Camera not available on this device');
                    });
            });
    } else if (inputSource === 'network_stream') {
        const url = document.getElementById('streamUrl').value;
        videoElement.src = url;
        updateStatus('Connecting to network stream...');
    } else if (inputSource === 'screen_record') {
        // For screen recording, fall back to camera for now
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                videoElement.srcObject = stream;
                updateStatus('Connected - Using camera as fallback');
            })
            .catch(err => {
                updateStatus('Error: Could not access camera');
            });
    } else if (inputSource === 'video_file') {
        // Video file source is handled in handleVideoFile function
        updateStatus('Loading video file...');
    }
}

/**
 * Video Feed Update Loop
 */
function updateVideoFeed() {
    if (!isPaused) {
        drawRoi();
        updateRoiInfo();
        
        // Capture and send frame for processing
        captureAndSendFrame();
    }
    requestAnimationFrame(updateVideoFeed);
}

/**
 * ROI Drawing Functions
 */
function drawRoi() {
    if (!canvas || !ctx) return;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Calculate scaling factors
    const videoRect = videoElement.getBoundingClientRect();
    const containerRect = canvas.getBoundingClientRect();
    
    if (videoElement.videoWidth && videoElement.videoHeight) {
        const videoAspect = videoElement.videoWidth / videoElement.videoHeight;
        const containerAspect = canvas.width / canvas.height;
        
        let displayWidth, displayHeight;
        // Use 'cover' behavior - fill the container completely
        if (videoAspect > containerAspect) {
            displayHeight = canvas.height;
            displayWidth = canvas.height * videoAspect;
        } else {
            displayWidth = canvas.width;
            displayHeight = canvas.width / videoAspect;
        }
        
        scale.x = displayWidth / videoElement.videoWidth;
        scale.y = displayHeight / videoElement.videoHeight;
        offset.x = (canvas.width - displayWidth) / 2;
        offset.y = (canvas.height - displayHeight) / 2;
    }
    
    // Convert ROI points to canvas coordinates
    const canvasPoints = roiPoints.map(point => ({
        x: point[0] * scale.x + offset.x,
        y: point[1] * scale.y + offset.y
    }));
    
    // Convert control points to canvas coordinates
    const canvasControlPoints = controlPoints.map(point => ({
        x: point[0] * scale.x + offset.x,
        y: point[1] * scale.y + offset.y
    }));
    
    // Draw ROI with curved edges using Bézier curves
    if (canvasPoints.length >= 3) {
        ctx.strokeStyle = colors.accent;
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        // Start from the first point
        ctx.moveTo(canvasPoints[0].x, canvasPoints[0].y);
        
        // Draw curved edges
        for (let i = 0; i < canvasPoints.length; i++) {
            const current = canvasPoints[i];
            const next = canvasPoints[(i + 1) % canvasPoints.length];
            const control = canvasControlPoints[i];
            
            // Draw quadratic Bézier curve
            ctx.quadraticCurveTo(control.x, control.y, next.x, next.y);
        }
        
        ctx.closePath();
        ctx.stroke();
        
        // Fill with semi-transparent color
        ctx.fillStyle = colors.accent + '20'; // Add transparency
        ctx.fill();
    }
    
    // Draw ROI corner points
    const isMobile = isMobileDevice();
    const cornerRadius = isMobile ? 12 : 8; // Larger on mobile
    
    canvasPoints.forEach((point, index) => {
        ctx.fillStyle = colors.accent;
        ctx.beginPath();
        ctx.arc(point.x, point.y, cornerRadius, 0, 2 * Math.PI);
        ctx.fill();
        
        ctx.strokeStyle = 'white';
        ctx.lineWidth = isMobile ? 3 : 2; // Thicker border on mobile
        ctx.stroke();
        
        // Draw point numbers
        ctx.fillStyle = 'white';
        ctx.font = `bold ${isMobile ? 14 : 12}px Arial`; // Larger font on mobile
        ctx.textAlign = 'center';
        ctx.fillText((index + 1).toString(), point.x, point.y - (isMobile ? 18 : 15));
    });
    
    // Draw control points for curve adjustment
    if (showControlPoints) {
        const controlRadius = isMobile ? 10 : 6; // Larger on mobile
        
        canvasControlPoints.forEach((control, index) => {
            ctx.fillStyle = '#00ffff'; // Cyan color for control points
            ctx.beginPath();
            ctx.arc(control.x, control.y, controlRadius, 0, 2 * Math.PI);
            ctx.fill();
            
            ctx.strokeStyle = 'white';
            ctx.lineWidth = isMobile ? 2 : 1; // Thicker border on mobile
            ctx.stroke();
            
            // Draw connection lines to show which edge this control point affects
            const current = canvasPoints[index];
            const next = canvasPoints[(index + 1) % canvasPoints.length];
            
            ctx.strokeStyle = '#00ffff60'; // Semi-transparent cyan
            ctx.lineWidth = 1;
            ctx.setLineDash([3, 3]);
            ctx.beginPath();
            ctx.moveTo(current.x, current.y);
            ctx.lineTo(control.x, control.y);
            ctx.lineTo(next.x, next.y);
            ctx.stroke();
            ctx.setLineDash([]); // Reset line dash
        });
    }
}

/**
 * Mouse Event Handlers
 */
function onCanvasClick(event) {
    if (event.target !== canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;
    
    // Check if click is near any control point first (smaller targets)
    for (let i = 0; i < controlPoints.length; i++) {
        const canvasX = controlPoints[i][0] * scale.x + offset.x;
        const canvasY = controlPoints[i][1] * scale.y + offset.y;
        
        if (Math.abs(mouseX - canvasX) < 10 && Math.abs(mouseY - canvasY) < 10) {
            draggingControl = i;
            canvas.style.cursor = 'grab';
            return;
        }
    }
    
    // Check if click is near any ROI corner point
    for (let i = 0; i < roiPoints.length; i++) {
        const canvasX = roiPoints[i][0] * scale.x + offset.x;
        const canvasY = roiPoints[i][1] * scale.y + offset.y;
        
        if (Math.abs(mouseX - canvasX) < 12 && Math.abs(mouseY - canvasY) < 12) {
            draggingPoint = i;
            canvas.style.cursor = 'grab';
            break;
        }
    }
}

function onCanvasMove(event) {
    if (event.target !== canvas) return;
    
    const rect = canvas.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;
    
    if (draggingControl !== null) {
        // Convert canvas coordinates back to frame coordinates for control point
        const frameX = (mouseX - offset.x) / scale.x;
        const frameY = (mouseY - offset.y) / scale.y;
        
        // Clamp to frame boundaries
        const maxX = videoElement.videoWidth || 640;
        const maxY = videoElement.videoHeight || 480;
        
        controlPoints[draggingControl][0] = Math.max(0, Math.min(frameX, maxX));
        controlPoints[draggingControl][1] = Math.max(0, Math.min(frameY, maxY));
        
        drawRoi();
        updateRoiInfo();
    } else if (draggingPoint !== null) {
        // Convert canvas coordinates back to frame coordinates for corner point
        const frameX = (mouseX - offset.x) / scale.x;
        const frameY = (mouseY - offset.y) / scale.y;
        
        // Clamp to frame boundaries
        const maxX = videoElement.videoWidth || 640;
        const maxY = videoElement.videoHeight || 480;
        
        roiPoints[draggingPoint][0] = Math.max(0, Math.min(frameX, maxX));
        roiPoints[draggingPoint][1] = Math.max(0, Math.min(frameY, maxY));
        
        // Update control points when corner points move
        updateControlPointsForCornerChange(draggingPoint);
        
        drawRoi();
        updateRoiInfo();
    } else {
        // Check if mouse is over any point for cursor change
        let overPoint = false;
        
        // Check control points first
        for (let i = 0; i < controlPoints.length; i++) {
            const canvasX = controlPoints[i][0] * scale.x + offset.x;
            const canvasY = controlPoints[i][1] * scale.y + offset.y;
            
            if (Math.abs(mouseX - canvasX) < 10 && Math.abs(mouseY - canvasY) < 10) {
                overPoint = true;
                break;
            }
        }
        
        // Check corner points
        if (!overPoint) {
            for (let i = 0; i < roiPoints.length; i++) {
                const canvasX = roiPoints[i][0] * scale.x + offset.x;
                const canvasY = roiPoints[i][1] * scale.y + offset.y;
                
                if (Math.abs(mouseX - canvasX) < 12 && Math.abs(mouseY - canvasY) < 12) {
                    overPoint = true;
                    break;
                }
            }
        }
        
        canvas.style.cursor = overPoint ? 'pointer' : 'crosshair';
    }
}

function updateControlPointsForCornerChange(cornerIndex) {
    // When a corner point moves, adjust the adjacent control points proportionally
    const prevControlIndex = (cornerIndex - 1 + controlPoints.length) % controlPoints.length;
    const currentControlIndex = cornerIndex;
    
    // Update the control point for the edge ending at this corner
    if (prevControlIndex >= 0) {
        const prevCorner = roiPoints[(cornerIndex - 1 + roiPoints.length) % roiPoints.length];
        const currentCorner = roiPoints[cornerIndex];
        
        const midX = (prevCorner[0] + currentCorner[0]) / 2;
        const midY = (prevCorner[1] + currentCorner[1]) / 2;
        
        // Keep the control point proportionally positioned
        const currentControl = controlPoints[prevControlIndex];
        const oldMidX = (prevCorner[0] + currentCorner[0]) / 2;
        const oldMidY = (prevCorner[1] + currentCorner[1]) / 2;
        
        // Adjust control point position
        controlPoints[prevControlIndex][0] = midX + (currentControl[0] - oldMidX);
        controlPoints[prevControlIndex][1] = midY + (currentControl[1] - oldMidY);
    }
    
    // Update the control point for the edge starting from this corner
    if (currentControlIndex < controlPoints.length) {
        const currentCorner = roiPoints[cornerIndex];
        const nextCorner = roiPoints[(cornerIndex + 1) % roiPoints.length];
        
        const midX = (currentCorner[0] + nextCorner[0]) / 2;
        const midY = (currentCorner[1] + nextCorner[1]) / 2;
        
        // Keep the control point proportionally positioned
        const currentControl = controlPoints[currentControlIndex];
        const oldMidX = (currentCorner[0] + nextCorner[0]) / 2;
        const oldMidY = (currentCorner[1] + nextCorner[1]) / 2;
        
        // Adjust control point position
        controlPoints[currentControlIndex][0] = midX + (currentControl[0] - oldMidX);
        controlPoints[currentControlIndex][1] = midY + (currentControl[1] - oldMidY);
    }
}

function onCanvasRelease(event) {
    draggingPoint = null;
    draggingControl = null;
    if (canvas) {
        canvas.style.cursor = 'crosshair';
    }
}

/**
 * Touch Event Handlers for Mobile
 */
function onCanvasTouch(event) {
    event.preventDefault(); // Prevent scrolling
    if (event.target !== canvas) return;
    
    const touch = event.touches[0];
    const rect = canvas.getBoundingClientRect();
    const touchX = touch.clientX - rect.left;
    const touchY = touch.clientY - rect.top;
    
    // Check if touch is near any control point first (smaller targets, larger touch area)
    for (let i = 0; i < controlPoints.length; i++) {
        const canvasX = controlPoints[i][0] * scale.x + offset.x;
        const canvasY = controlPoints[i][1] * scale.y + offset.y;
        
        if (Math.abs(touchX - canvasX) < 20 && Math.abs(touchY - canvasY) < 20) { // Larger touch area
            draggingControl = i;
            return;
        }
    }
    
    // Check if touch is near any ROI corner point
    for (let i = 0; i < roiPoints.length; i++) {
        const canvasX = roiPoints[i][0] * scale.x + offset.x;
        const canvasY = roiPoints[i][1] * scale.y + offset.y;
        
        if (Math.abs(touchX - canvasX) < 25 && Math.abs(touchY - canvasY) < 25) { // Larger touch area
            draggingPoint = i;
            break;
        }
    }
}

function onCanvasTouchMove(event) {
    event.preventDefault(); // Prevent scrolling
    if (event.target !== canvas) return;
    
    const touch = event.touches[0];
    const rect = canvas.getBoundingClientRect();
    const touchX = touch.clientX - rect.left;
    const touchY = touch.clientY - rect.top;
    
    if (draggingControl !== null) {
        // Convert canvas coordinates back to frame coordinates for control point
        const frameX = (touchX - offset.x) / scale.x;
        const frameY = (touchY - offset.y) / scale.y;
        
        // Clamp to frame boundaries
        const maxX = videoElement.videoWidth || 640;
        const maxY = videoElement.videoHeight || 480;
        
        controlPoints[draggingControl][0] = Math.max(0, Math.min(frameX, maxX));
        controlPoints[draggingControl][1] = Math.max(0, Math.min(frameY, maxY));
        
        drawRoi();
        updateRoiInfo();
    } else if (draggingPoint !== null) {
        // Convert canvas coordinates back to frame coordinates for corner point
        const frameX = (touchX - offset.x) / scale.x;
        const frameY = (touchY - offset.y) / scale.y;
        
        // Clamp to frame boundaries
        const maxX = videoElement.videoWidth || 640;
        const maxY = videoElement.videoHeight || 480;
        
        roiPoints[draggingPoint][0] = Math.max(0, Math.min(frameX, maxX));
        roiPoints[draggingPoint][1] = Math.max(0, Math.min(frameY, maxY));
        
        // Update control points when corner points move
        updateControlPointsForCornerChange(draggingPoint);
        
        drawRoi();
        updateRoiInfo();
    }
}

function onCanvasTouchEnd(event) {
    event.preventDefault();
    draggingPoint = null;
    draggingControl = null;
}

/**
 * Window Event Handlers
 */
function onWindowResize() {
    if (canvas) {
        const container = document.getElementById('videoContainer');
        canvas.width = container.offsetWidth;
        canvas.height = container.offsetHeight;
        
        // Also resize segmentation canvas
        if (segmentationCanvas) {
            segmentationCanvas.width = container.offsetWidth;
            segmentationCanvas.height = container.offsetHeight;
            
            // Redraw segmentation overlay if it exists
            if (currentSegmentationOverlay) {
                drawSegmentationOverlay();
            }
        }
        
        drawRoi();
    }
}

/**
 * UI Update Functions
 */
function updateRoiInfo() {
    let roiText = '<table style="width: 100%; border-collapse: collapse; font-size: 10px;">';
    roiText += '<tr><th style="border: 1px solid #555; padding: 4px; background-color: #444;">Corner Points</th>';
    
    if (showControlPoints) {
        roiText += '<th style="border: 1px solid #555; padding: 4px; background-color: #444;">Curve Controls</th>';
    }
    roiText += '</tr>';
    
    const maxRows = Math.max(roiPoints.length, showControlPoints ? controlPoints.length : 0);
    
    for (let i = 0; i < maxRows; i++) {
        roiText += '<tr>';
        
        // Corner Points column
        if (i < roiPoints.length) {
            const point = roiPoints[i];
            roiText += `<td style="border: 1px solid #555; padding: 4px;">P${i + 1}: (${Math.round(point[0])}, ${Math.round(point[1])})</td>`;
        } else {
            roiText += '<td style="border: 1px solid #555; padding: 4px;"></td>';
        }
        
        // Curve Controls column (only if showControlPoints is true)
        if (showControlPoints) {
            if (i < controlPoints.length) {
                const point = controlPoints[i];
                roiText += `<td style="border: 1px solid #555; padding: 4px;">C${i + 1}: (${Math.round(point[0])}, ${Math.round(point[1])})</td>`;
            } else {
                roiText += '<td style="border: 1px solid #555; padding: 4px;"></td>';
            }
        }
        
        roiText += '</tr>';
    }
    
    roiText += '</table>';
    
    document.getElementById('roiInfo').innerHTML = roiText;
}

function updateStatus(message) {
    document.getElementById('statusText').textContent = message;
}

/**
 * Menu Button Functions
 */
function changeInputSource() {
    // Stop current video
    if (videoElement) {
        if (videoElement.srcObject) {
            videoElement.srcObject.getTracks().forEach(track => track.stop());
        }
        videoElement.src = '';
    }
    
    // Hide main interface
    document.getElementById('menuFrame').style.display = 'none';
    document.getElementById('displayFrame').style.display = 'none';
    
    // Reset variables
    inputSource = null;
    
    // Show input selection
    showInputSelection();
}

function toggleControlPoints() {
    showControlPoints = !showControlPoints;
    const button = event.target;
    button.textContent = showControlPoints ? '🎛️ Hide Curves' : '🎛️ Show Curves';
    drawRoi();
    updateRoiInfo();
    updateStatus(showControlPoints ? 'Curve controls visible' : 'Curve controls hidden');
}

function resetRoi() {
    // Reset ROI based on current video dimensions
    initializeRoiPoints();
    drawRoi();
    updateRoiInfo();
    updateStatus('ROI reset to default');
}

function startMusicGeneration() {
    if (!audioContext) {
        initializeAudioSystem();
    }
    
    // Resume audio context if suspended (required by browser)
    if (audioContext && audioContext.state === 'suspended') {
        audioContext.resume().then(() => {
            console.log('🎵 Audio context resumed');
        });
    }
    
    if (isMusicGenerationActive) {
        stopMusicGeneration();
    } else {
        isMusicGenerationActive = true;
        updateMusicButton();
        updateStatus('🎵 Music generation started - listening for events...');
        
        // Request music generation from server
        if (segmentationSocket && segmentationSocket.connected) {
            segmentationSocket.emit('toggle_music', { enabled: true });
        } else {
            console.warn('⚠️ Socket not connected, music will start when connection is established');
        }
    }
}

function stopMusicGeneration() {
    isMusicGenerationActive = false;
    updateMusicButton();
    updateStatus('🎵 Music generation stopped');
    
    // Stop any currently playing notes
    stopAllActiveNotes();
    
    // Disable music generation on server
    if (segmentationSocket && segmentationSocket.connected) {
        segmentationSocket.emit('toggle_music', { enabled: false });
    }
}

function updateMusicButton() {
    const musicBtn = document.querySelector('.music-gen-btn');
    if (musicBtn) {
        if (isMusicGenerationActive) {
            musicBtn.textContent = '🔇 Stop Music';
            musicBtn.style.backgroundColor = '#ff4444';
            musicBtn.classList.add('playing');
        } else {
            musicBtn.textContent = '🎵 Generate Music';
            musicBtn.style.backgroundColor = '#4a9eff';
            musicBtn.classList.remove('playing');
        }
    }
}

function togglePause() {
    isPaused = !isPaused;
    updateStatus(isPaused ? 'Paused' : 'Resumed');
    
    // Update button text and icon
    const pauseBtn = document.getElementById('pauseBtn');
    if (pauseBtn) {
        pauseBtn.textContent = isPaused ? '▶️ Play' : '⏸️ Pause';
    }
    
    if (videoElement) {
        if (isPaused) {
            videoElement.pause();
        } else {
            videoElement.play();
        }
    }
}

function takeScreenshot() {
    if (videoElement && videoElement.videoWidth && videoElement.videoHeight) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        
        ctx.drawImage(videoElement, 0, 0);
        
        // Convert to blob and download
        canvas.toBlob(function(blob) {
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `screenshot_${new Date().toISOString().replace(/[:.]/g, '-')}.jpg`;
            link.click();
            URL.revokeObjectURL(url);
            
            updateStatus('Screenshot saved');
        }, 'image/jpeg', 0.95);
    } else {
        updateStatus('No video frame available for screenshot');
    }
}

/**
 * Mobile Viewport Height Fix
 * Handles the mobile browser navigation bar issue
 */
function handleMobileViewportHeight() {
    // Set CSS custom properties for viewport height handling
    const setViewportHeight = () => {
        const vh = window.innerHeight * 0.01;
        document.documentElement.style.setProperty('--vh', `${vh}px`);
        
        // Also set dvh fallback for older browsers
        document.documentElement.style.setProperty('--dvh', `${window.innerHeight}px`);
    };
    
    // Set initial height
    setViewportHeight();
    
    // Update on resize and orientation change
    window.addEventListener('resize', setViewportHeight);
    window.addEventListener('orientationchange', () => {
        // Add delay for orientation change to complete
        setTimeout(setViewportHeight, 300);
    });
    
    // Handle iOS Safari specifically
    if (/iPad|iPhone|iPod/.test(navigator.userAgent)) {
        // Listen for scroll events to detect when address bar hides/shows
        let initialViewportHeight = window.innerHeight;
        
        window.addEventListener('scroll', () => {
            if (window.innerHeight !== initialViewportHeight) {
                setViewportHeight();
                initialViewportHeight = window.innerHeight;
            }
        });
        
        // Force layout recalculation on iOS
        document.addEventListener('touchstart', () => {
            setTimeout(setViewportHeight, 100);
        });
    }
}

// Call the mobile viewport fix on page load
document.addEventListener('DOMContentLoaded', handleMobileViewportHeight);
