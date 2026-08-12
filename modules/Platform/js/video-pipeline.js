/**
 * RagbaarNet AI Platform — video-pipeline.js
 * ===========================================
 * Everything about getting frames INTO the system and segmentation frames
 * back OUT: input-source selection (camera/file/screen/network), Socket.IO
 * connection to processor.py, frame capture/send loop, and segmentation
 * overlay display. Depends on core.js (detectProcessorUrl, isMobileDevice,
 * updateStatus, ...). Requires roi.js's drawRoi()/setupRoiCanvas() to already
 * be defined by the time the video/frame loop actually runs (load roi.js
 * before this file, or after — see note in core.js; execution order at
 * runtime is what matters, not declaration order, so either load order works
 * as long as both are loaded before window.onload fires).
 */

let videoElement = null;

// Reference to the <video> tag
let streamElement = null;

// Reference to the <img> tag for MJPEG streams
let activeSource = null;

let segmentationCanvas = null;

let segmentationCtx = null;

let isPaused = false;

// Frame processing variables
let frameProcessingEnabled = true;

// Always keep processing enabled
let segmentationDisplayEnabled = false;

// Only control display - Start with segmentation display OFF

// Dynamic processor URL detection for mobile/desktop compatibility
let processorUrl = detectProcessorUrl();

let frameCounter = 0;

let lastFrameSentTime = 0;

// Adapt frame send rate on mobile to reduce bandwidth/CPU contention
let frameSendInterval = isMobileDevice() ? 250 : 150;

// ms
let processingCanvas = null;

let processingCtx = null;

let segmentationSocket = null;

let currentSegmentationOverlay = null;

let currentSegmentationInfo = null;

// Prevent stale/out-of-order overlays from replacing newer ones on mobile
let latestOverlayFrameCounter = -1;

let drawToken = 0;

// Performance optimization variables
let isProcessingFrame = false;

// Prevent concurrent frame processing
let lastUpdateTime = 0;

let updateThrottleInterval = 50;

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
                console.trace('🔌 Creating new socket connection');
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
            
            // Sync the currently active musician (in case it differs from our default guess)
            segmentationSocket.emit('get_available_musicians');
            segmentationSocket.emit('get_music_status');
            
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
        
        segmentationSocket.on('music_status', function(data) {
            if (data && Number.isInteger(data.tempo)) {
                currentTempo = clampTempoValue(data.tempo);
                pendingTempo = currentTempo;
                currentSpeedKmh = speedFromTempo(currentTempo);
                pendingSpeedKmh = currentSpeedKmh;
                lastMusicStatus.tempo = currentTempo;
                updateTempoControls(currentSpeedKmh);
                updateMusicStatusDisplay();

                if (Tone.Transport.state === 'started') {
                    syncTransportBpm(currentTempo);
                }
            }
            if (data && data.instrument) {
                currentInstrument = data.instrument;
                pendingInstrument = currentInstrument;
                updateInstrumentControls();
            }
            if (data && data.key_signature) {
                lastMusicStatus.keySignature = data.key_signature;
                updateMusicStatusDisplay();
            }
            if (data && data.time_signature) {
                lastMusicStatus.timeSignature = data.time_signature;
                updateMusicStatusDisplay();
            }
        });

        // Pushed by the processor whenever it receives fresh telemetry
        segmentationSocket.on('telemetry_update', function(data) {
            if (!data) return;

            latestTelemetry = {
                speed_kmh: Number.isFinite(data.speed_kmh) ? data.speed_kmh : latestTelemetry.speed_kmh,
                accel: Number.isFinite(data.accel) ? data.accel : latestTelemetry.accel,
                rpm: Number.isFinite(data.rpm) ? data.rpm : latestTelemetry.rpm
            };

            currentSpeedKmh = latestTelemetry.speed_kmh != null ? clampSpeedValue(latestTelemetry.speed_kmh) : currentSpeedKmh;
            currentTempo = calculateAutoTempoFromSpeed(currentSpeedKmh);
            if (Tone.Transport.state === 'started') {
                syncTransportBpm(currentTempo);
            }
            pendingSpeedKmh = currentSpeedKmh;
            pendingTempo = currentTempo;
            lastMusicStatus.tempo = currentTempo;

            updateTempoControls(currentSpeedKmh);
            updateMusicStatusDisplay();

            const telemetryStatusEl = document.getElementById('telemetryStatus');
            if (telemetryStatusEl) {
                telemetryStatusEl.textContent = `📡 Live: ${Math.round(latestTelemetry.speed_kmh ?? 0)} km/h`
                    + (latestTelemetry.rpm != null ? ` • ${Math.round(latestTelemetry.rpm)} RPM` : '');
            }
        });
        
        segmentationSocket.on('musicians_list', function(data) {
            if (data && Array.isArray(data.musicians) && data.musicians.length > 0) {
                availableMusicians = data.musicians;
            }
            if (data && data.current) {
                currentMusicianType = data.current;
            }
            if (data && data.instrument) {
                currentInstrument = data.instrument;
                pendingInstrument = currentInstrument;
            }
            renderMusicianList();
            updateInstrumentControls();
        });
        
        segmentationSocket.on('music_settings_updated', function(data) {
            clearTimeout(musicianSwitchTimeoutId);
            isSwitchingMusician = false;
            setMusicianListInteractive(true);

            if (data && data.success) {
                currentMusicianType = data.musician_type;
                currentInstrument = data.instrument || currentInstrument;
                currentTempo = clampTempoValue(data.tempo);
                currentSpeedKmh = speedFromTempo(currentTempo);
                pendingMusicianSelection = currentMusicianType;
                pendingInstrument = currentInstrument;
                pendingTempo = currentTempo;
                pendingSpeedKmh = currentSpeedKmh;
                lastMusicStatus.tempo = currentTempo;
                updateMusicStatusDisplay();
                updateStatus(`🎵 Music settings updated • ${getMusicianLabel(currentMusicianType)} • ${currentTempo} BPM`);
                closeMusicianModal();
            } else {
                const errorMessage = (data && data.error) || 'Unknown error';
                setMusicianModalStatus(`❌ Failed to update music settings: ${errorMessage}`);
                updateMusicianApplyButton();
            }
        });

        segmentationSocket.on('musician_switched', function(data) {
            clearTimeout(musicianSwitchTimeoutId);
            isSwitchingMusician = false;
            setMusicianListInteractive(true);
            
            if (data && data.success) {
                currentMusicianType = data.musician_type;
                renderMusicianList();
                const label = getMusicianLabel(currentMusicianType);
                setMusicianModalStatus(`✅ Now using: ${label}`);
                updateStatus(`🎭 Musician switched to ${label}`);
            } else {
                const errorMessage = (data && data.error) || 'Unknown error';
                setMusicianModalStatus(`❌ Failed to switch musician: ${errorMessage}`);
                console.error('❌ Musician switch failed:', errorMessage);
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
                        console.trace('🔌 Creating new socket connection');
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
 * Frame Processing and Transmission
*/
function captureAndSendFrame() {
    // Prevent concurrent frame processing
    if (isProcessingFrame || !activeSource) {
        return;
    }

    const srcW = getSourceWidth();
    const srcH = getSourceHeight();

    if (!srcW || !srcH) return;

    const now = Date.now();
    if (now - lastFrameSentTime < frameSendInterval) {
        return; // Rate limiting
    }
    
    const currentFrameId = frameCounter++;
    lastFrameSentTime = now;
    isProcessingFrame = true; // Set processing flag
    
    try {
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
        processingCtx.drawImage(activeSource, 0, 0, targetW, targetH);
        
        // Convert to base64 with optimized quality for speed
        const jpegQuality = isMobileDevice() ? 0.6 : 0.7;
        const frameData = processingCanvas.toDataURL('image/jpeg', jpegQuality);
        
        // Send frame to processor
        const frameInfo = {
            frame: frameData,
            frame_id: `frame_${currentFrameId}`,
            timestamp: now / 1000,
            // ROI coordinates must use the same dimensions as the transmitted frame.
            roi_points: roiPoints.map(point => [
                point[0] * targetW / srcW,
                point[1] * targetH / srcH
            ]),
            roi_controls: controlPoints.map(point => [
                point[0] * targetW / srcW,
                point[1] * targetH / srcH
            ])
        };
        
        // Reduced logging for performance - only log every 10th frame
        if (currentFrameId  % 10 === 0) {
            console.log(`📤 Sending frame ${currentFrameId} to processor`);
        }
        
        // Send via HTTP (more reliable than WebSocket for large data)
        sendFrameToProcessor(frameInfo)
            .then(response => {
                if (response.success) {
                    
                    // Update frame counter in display (throttled)
                    if (currentFrameId  % 5 === 0) {
                        updateFrameCounter(currentFrameId);
                    }
                    
                    // Reduced logging for performance
                    if (currentFrameId % 10 === 0) {
                        console.log(`✅ Frame ${currentFrameId} processed successfully`);
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

function updateFrameCounter(count) {
    // This function is kept for backward compatibility
    // Frame counter is now updated directly in updateSegmentationDisplay
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
    
    // When display is OFF, keep the latest overlay cached so toggling ON can show it immediately.
    // (Canvas may be hidden anyway, so no need to clear+drop cached data here.)
}

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

function clearSegmentationOverlay() {
    if (segmentationCanvas && segmentationCtx) {
        segmentationCtx.clearRect(0, 0, segmentationCanvas.width, segmentationCanvas.height);
    }
    // Keep `currentSegmentationOverlay` cached; only the visible canvas is cleared.
}

function toggleFrameProcessing() {
    console.log('🎯 toggleSegmentationDisplay called, current state:', segmentationDisplayEnabled);
    
    segmentationDisplayEnabled = !segmentationDisplayEnabled;
    
    console.log('🎯 New display state:', segmentationDisplayEnabled);
    
    // Update button state immediately
    updateSegmentationButtonState();
    
    // Show/hide elements based on segmentation display state
    const roiCanvas = document.getElementById('roiCanvas');
    const segCanvas = document.getElementById('segmentationCanvas');
    
    if (segmentationDisplayEnabled) {
        // Show segmentation overlay with ROI (hide source but keep ROI visible)
        if (activeSource) {
            // IMPORTANT: Using display:none can freeze frame updates in some browsers.
            // Keep the source in the render tree and hide it visually instead.
            activeSource.style.display = 'block';
            activeSource.style.visibility = 'hidden';
            activeSource.style.opacity = '0';
            activeSource.style.pointerEvents = 'none';
        }
        if (roiCanvas) roiCanvas.style.display = 'block'; // Keep ROI visible
        if (segCanvas) {
            segCanvas.style.display = 'block';
            segCanvas.style.pointerEvents = 'none'; // Keep it non-interactive
        }
        setInstructionsText('Displaying AI Segmentation Overlay with ROI • Drag points to adjust ROI • Toggle off to return to original video');
        
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
        // Show original source with ROI (hide segmentation display, but processing continues)
        if (activeSource) {
            activeSource.style.display = 'block';
            activeSource.style.visibility = 'visible';
            activeSource.style.opacity = '1';
            activeSource.style.pointerEvents = '';
        }
        if (roiCanvas) roiCanvas.style.display = 'block';
        if (segCanvas) segCanvas.style.display = 'none';
        setInstructionsText('Drag green points to adjust ROI corners • Drag cyan points to control edge curves');
        
        updateStatus('Showing original video with ROI (segmentation processing continues in background)');
        console.log('❌ Segmentation DISPLAY: Hidden (background processing continues)');
        
        // Clear segmentation overlay display but keep processing
        clearSegmentationOverlay();
    }
}

function updateSegmentationButtonState() {
    const button = document.getElementById('toggleSegmentationBtn');
    const img = document.getElementById('toggleSegmentationBtnImg');

    if (!button) return;

    button.dataset.active = segmentationDisplayEnabled.toString();
    button.setAttribute('aria-pressed', segmentationDisplayEnabled.toString());

    button.title = segmentationDisplayEnabled
        ? 'Segmentation View: ON (tap to return to original video)'
        : 'Segmentation View: OFF (tap to show segmentation overlay)';

    if (img) {
        img.src = segmentationDisplayEnabled
            ? '../../assets/icons/segment.png'
            : '../../assets/icons/segment.png';
    }
}

/**
 * Input Source Selection
 */
function showInputSelection() {
    const modal = document.getElementById('inputModal');
    modal.style.display = 'flex';       // Use flex instead of block for centering
    modal.classList.add('show');        // Add show class for better styling
    
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
            streamElement = document.getElementById('streamElement');

            // Clear any existing camera stream
            if (videoElement.srcObject) {
                videoElement.srcObject.getTracks().forEach(track => track.stop());
                videoElement.srcObject = null;
            }

            // Switch to video mode
            activeSource = videoElement;
            videoElement.style.display = 'block';
            streamElement.style.display = 'none';
            streamElement.src = '';

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
    
    // Start video capture
    startVideoCapture();
    
    updateStatus('Ready');
}

/**
 * Gets the current active source width (videoWidth for video, naturalWidth for images)
 */
function getSourceWidth() {
    if (!activeSource) return 0;
    return activeSource.tagName === 'VIDEO' ? activeSource.videoWidth : activeSource.naturalWidth;
}

/**
 * Gets the current active source height (videoHeight for video, naturalHeight for images)
 */
function getSourceHeight() {
    if (!activeSource) return 0;
    return activeSource.tagName === 'VIDEO' ? activeSource.videoHeight : activeSource.naturalHeight;
}

/**
 * Video Display Setup
 */
function setupVideoDisplay() {
    videoElement = document.getElementById('videoElement');
    streamElement = document.getElementById('streamElement');

    // Set crossOrigin to allow canvas capture of remote sources without security errors
    videoElement.crossOrigin = "anonymous";
    streamElement.crossOrigin = "anonymous";

    activeSource = videoElement; // Default to video

    // Mobile-specific video settings
    videoElement.setAttribute('playsinline', 'true');
    videoElement.setAttribute('webkit-playsinline', 'true');
    
    // Shared load handler
    const onSourceLoaded = function() {
        updateStatus('Source loaded successfully');
        console.log('Source dimensions:', getSourceWidth(), 'x', getSourceHeight());
        
        // Reinitialize ROI points based on actual dimensions
        initializeRoiPoints();
        updateVideoFeed();
    };

    // Handle video load
    videoElement.addEventListener('loadedmetadata', onSourceLoaded);
    
    // Handle MJPEG image load
    streamElement.addEventListener('load', onSourceLoaded);

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
        if (inputSource !== 'network_stream') updateStatus('Error loading video source');
    });

    streamElement.addEventListener('error', function(e) {
        console.error('Stream error details:', e);
        updateStatus('Error connecting to MJPEG stream');
    });
    
    // Add mobile debugging
    videoElement.addEventListener('loadstart', function() {
        console.log('Video load started');
        updateStatus('Starting video...');
    });
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
    videoElement.src = '';
    streamElement.src = '';

    if (inputSource === 'network_stream') {
        // Switch to MJPEG stream mode (IMG tag)
        activeSource = streamElement;
        videoElement.style.display = 'none';
        streamElement.style.display = 'block';
        
        const url = document.getElementById('streamUrl').value;
        streamElement.src = url;
        updateStatus('Connecting to MJPEG stream...');
        console.log('🔌 Switching to MJPEG real-time stream:', url);
    } else {
        // Standard Video mode (VIDEO tag)
        activeSource = videoElement;
        videoElement.style.display = 'block';
        streamElement.style.display = 'none';

        if (inputSource === 'phone_camera') {
            // Enhanced mobile camera constraints
            const constraints = {
                video: {
                    facingMode: 'environment',
                    width: { ideal: 1280, max: 1920 },
                    height: { ideal: 720, max: 1080 },
                    frameRate: { ideal: 30, max: 60 }
                }
            };

            navigator.mediaDevices.getUserMedia(constraints)
                .then(stream => {
                    videoElement.srcObject = stream;
                    const playPromise = videoElement.play();
                    if (playPromise && typeof playPromise.then === 'function') {
                        playPromise.catch(err => {
                            console.warn('Camera playback blocked:', err);
                        });
                    }
                })
                .catch(err => {
                    console.error('Camera error:', err);
                    updateStatus(`Error: Camera access failed`);
                });
        } else if (inputSource === 'screen_record') {
            if (navigator.mediaDevices && typeof navigator.mediaDevices.getDisplayMedia === 'function') {
                navigator.mediaDevices.getDisplayMedia({ video: true })
                    .then(stream => {
                        videoElement.srcObject = stream;
                        videoElement.play();
                    })
                    .catch(err => {
                        console.warn('Screen capture failed:', err);
                    });
            }
        }
        // video_file is handled by its own change listener
    }
}

/**
 * Video Feed Update Loop
 */
function updateVideoFeed() {
    if (!isPaused) {
        drawRoi();
        
        // Capture and send frame for processing
        captureAndSendFrame();
    }
    requestAnimationFrame(updateVideoFeed);
}

/**
 * Menu Button Functions
 */
function changeInputSource() {
    // Stop current video/stream
    if (videoElement) {
        if (videoElement.srcObject) {
            videoElement.srcObject.getTracks().forEach(track => track.stop());
        }
        videoElement.src = '';
    }
    if (streamElement) {
        streamElement.src = '';
    }
    
    // Hide main interface
    document.getElementById('menuFrame').style.display = 'none';
    document.getElementById('displayFrame').style.display = 'none';
    
    // Reset variables
    inputSource = null;
    
    // Show input selection
    showInputSelection();
}
