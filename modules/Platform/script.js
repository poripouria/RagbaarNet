/**
 * AI Music Generation Platform - Main JavaScript Module
 * Handles video input sources, ROI drawing, and music generation
 */

// Global variables
let inputSource = null;
let videoElement = null;
let canvas = null;
let ctx = null;
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
let frameProcessingEnabled = true;
let processorUrl = 'http://127.0.0.1:5000';
let frameCounter = 0;
let lastFrameSentTime = 0;
let frameSendInterval = 200; // Send frames every 200ms for better performance
let processingCanvas = null;
let processingCtx = null;
let segmentationSocket = null;
let segmentationDisplay = null;

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
        const segButton = document.getElementById('toggleProcessingBtn');
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
    
    // Try to connect to segmentation processor
    connectToProcessor();
    
    console.log('✅ Frame processing initialized');
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
        segmentationSocket = io(processorUrl);
        
        segmentationSocket.on('connect', function() {
            console.log('✅ Connected to segmentation processor');
            updateStatus('Processor connected - Ready for segmentation');
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

function startRequestingUpdates() {
    // Request updates every 100ms
    setInterval(() => {
        if (segmentationSocket && segmentationSocket.connected) {
            segmentationSocket.emit('request_update');
        }
    }, 100);
}

async function checkProcessorStatus() {
    const response = await fetch(`${processorUrl}/api/status`);
    if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
    }
    return response.json();
}

function captureAndSendFrame() {
    if (!frameProcessingEnabled || !videoElement || !videoElement.videoWidth) {
        return;
    }
    
    const now = Date.now();
    if (now - lastFrameSentTime < frameSendInterval) {
        return; // Rate limiting
    }
    
    try {
        // Set processing canvas size to match video
        processingCanvas.width = videoElement.videoWidth;
        processingCanvas.height = videoElement.videoHeight;
        
        // Draw current video frame to processing canvas
        processingCtx.drawImage(videoElement, 0, 0);
        
        // Convert to base64
        const frameData = processingCanvas.toDataURL('image/jpeg', 0.8);
        
        // Send frame to processor
        const frameInfo = {
            frame: frameData,
            frame_id: `frame_${frameCounter}`,
            timestamp: now / 1000,
            roi_points: roiPoints
        };
        
        // Log frame sending for debugging
        console.log(`📤 Sending frame ${frameCounter} to processor`);
        
        // Send via HTTP (more reliable than WebSocket for large data)
        sendFrameToProcessor(frameInfo)
            .then(response => {
                if (response.success) {
                    frameCounter++;
                    lastFrameSentTime = now;
                    
                    // Update frame counter in display
                    updateFrameCounter(frameCounter);
                    
                    console.log(`✅ Frame ${frameCounter} processed successfully`);
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
            });
            
    } catch (error) {
        console.error('❌ Frame capture error:', error);
        updateSegmentationStatus('Capture Error');
    }
}

async function sendFrameToProcessor(frameInfo) {
    const response = await fetch(`${processorUrl}/api/process_frame`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(frameInfo)
    });
    
    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    return response.json();
}

function updateSegmentationStatus(status) {
    const statusElement = document.getElementById('segProcessorStatus');
    if (statusElement) {
        statusElement.textContent = status;
    }
    
    // Update container border color based on status
    const container = document.getElementById('segmentationContainer');
    if (container) {
        if (status === 'Connected') {
            container.style.borderColor = '#00ff88';
        } else if (status.includes('Error') || status.includes('Failed')) {
            container.style.borderColor = '#ff4444';
        } else {
            container.style.borderColor = '#ffaa00';
        }
    }
}

function updateSegmentationDisplay(data) {
    // Update frame counter
    if (data.frame_counter !== undefined) {
        updateFrameCounter(data.frame_counter);
    }
    
    // Update segmentation image
    if (data.segmentation_overlay) {
        const segImg = document.getElementById('segmentationImage');
        const statusDiv = document.getElementById('segmentationStatus');
        
        if (segImg && statusDiv) {
            segImg.src = data.segmentation_overlay;
            segImg.style.display = 'block';
            statusDiv.style.display = 'none';
            
            // Update segmentation info
            if (data.segmentation_info) {
                const framesSince = data.segmentation_info.frames_since_segmentation || 0;
                const framesSinceElement = document.getElementById('segFramesSince');
                if (framesSinceElement) {
                    framesSinceElement.textContent = framesSince;
                }
                
                // Log successful segmentation
                console.log('🔍 Segmentation updated:', data.segmentation_info);
            }
            
            updateSegmentationStatus('Processing');
        }
    } else {
        // No segmentation available
        const segImg = document.getElementById('segmentationImage');
        const statusDiv = document.getElementById('segmentationStatus');
        
        if (segImg && statusDiv) {
            segImg.style.display = 'none';
            statusDiv.style.display = 'block';
            statusDiv.textContent = 'Processing...';
        }
    }
}

function updateFrameCounter(count) {
    const frameCountElement = document.getElementById('segFrameCount');
    if (frameCountElement) {
        frameCountElement.textContent = count;
    }
}

function toggleFrameProcessing() {
    console.log('🎯 toggleFrameProcessing called, current state:', frameProcessingEnabled);
    
    frameProcessingEnabled = !frameProcessingEnabled;
    
    console.log('🎯 New state:', frameProcessingEnabled);
    
    // Update button state immediately
    updateSegmentationButtonState();
    
    if (frameProcessingEnabled) {
        updateStatus('Frame processing enabled');
        console.log('✅ Segmentation turned ON');
        
        // Reconnect if disconnected
        if (!segmentationSocket || !segmentationSocket.connected) {
            connectToProcessor();
        }
    } else {
        updateStatus('Frame processing disabled');
        console.log('❌ Segmentation turned OFF');
        
        // Clear segmentation display
        const segImg = document.getElementById('segmentationImage');
        const statusDiv = document.getElementById('segmentationStatus');
        if (segImg && statusDiv) {
            segImg.style.display = 'none';
            statusDiv.style.display = 'block';
            statusDiv.textContent = 'Processing disabled';
        }
    }
}

function updateSegmentationButtonState() {
    const button = document.getElementById('toggleProcessingBtn');
    
    if (button) {
        console.log('🎯 Updating button state. frameProcessingEnabled:', frameProcessingEnabled);
        
        if (frameProcessingEnabled) {
            button.textContent = '🔍 Segmentation: ON';
            button.style.background = '#00ff88';
            button.style.color = 'black';
        } else {
            button.textContent = '🔍 Segmentation: OFF';
            button.style.background = '#666';
            button.style.color = 'white';
        }
        
        // Preserve mobile-friendly styles
        button.style.padding = '8px 16px';
        button.style.borderRadius = '6px';
        button.style.minHeight = '44px';
        button.style.minWidth = '120px';
        button.style.border = 'none';
        button.style.cursor = 'pointer';
        button.style.fontSize = '12px';
        button.style.fontWeight = 'bold';
        button.style.touchAction = 'manipulation';
        button.style.userSelect = 'none';
        button.style.webkitTapHighlightColor = 'transparent';
        
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
        const button = document.getElementById('toggleProcessingBtn');
        if (button) {
            const expectedText = frameProcessingEnabled ? '🔍 Segmentation: ON' : '🔍 Segmentation: OFF';
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
    document.getElementById('urlModal').style.display = 'block';
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
    
    // Set canvas size to match container
    const container = document.getElementById('videoContainer');
    canvas.width = container.offsetWidth;
    canvas.height = container.offsetHeight;
    
    // Initialize ROI points based on video/canvas dimensions
    initializeRoiPoints();
    
    // Start drawing ROI
    drawRoi();
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
    const message = `Music generation will be implemented here!\nROI Points: ${JSON.stringify(roiPoints)}\nInput Source: ${inputSource}`;
    alert(message);
    updateStatus('Music generation started');
}

function togglePause() {
    isPaused = !isPaused;
    updateStatus(isPaused ? 'Paused' : 'Resumed');
    
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
