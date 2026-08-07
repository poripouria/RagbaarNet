/**
 * RagbaarNet AI Platform — roi.js
 * ================================
 * The ROI (region of interest) system: the draggable Bézier-curved polygon
 * overlaid on the video, corner/control-point dragging (mouse + touch),
 * fill toggle, tooltips, and the ROI canvas setup. Self-contained aside from
 * relying on `videoElement` (video-pipeline.js) and `updateStatus`/
 * `escapeHtml` (core.js).
 */

// Current active DOM element (videoElement or streamElement)
let canvas = null;

let ctx = null;

let roiPoints = [];

// Will be initialized based on video dimensions
let controlPoints = [];

// Bézier control points for curves
let draggingPoint = null;

let draggingControl = null;

let scale = {x: 1, y: 1};

let offset = {x: 0, y: 0};

let showControlPoints = true;

let roiFillEnabled = true;

// When false, ROI area is transparent (outline + vertices still visible)
let roiFillHoldTimer = null;

let roiFillLongPressTriggered = false;

const ROI_RESET_HOLD_DURATION_MS = 600;

/**
 * ROI Canvas Setup
 */
function setupRoiCanvas() {
    canvas = document.getElementById('roiCanvas');
    ctx = canvas.getContext('2d');
    
    // Setup non-interactive result canvases.
    segmentationCanvas = document.getElementById('segmentationCanvas');
    segmentationCtx = segmentationCanvas.getContext('2d');
    
    // Set canvas size to match container
    const container = document.getElementById('videoContainer');
    canvas.width = container.offsetWidth;
    canvas.height = container.offsetHeight;
    segmentationCanvas.width = container.offsetWidth;
    segmentationCanvas.height = container.offsetHeight;
    
    // Hide the point tooltip whenever the mouse leaves the canvas
    canvas.addEventListener('mouseleave', hidePointTooltip);
    
    // Initialize ROI points based on video/canvas dimensions
    initializeRoiPoints();
    
    // Start drawing ROI
    drawRoi();
    
    // Update segmentation button state
    updateSegmentationButtonState();

    // Update ROI fill button state
    updateRoiFillButtonState();

    // Enable press-and-hold on the fill-toggle icon to reset the ROI
    setupRoiFillHoldToReset();
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
    // Get source dimensions, fallback to canvas dimensions if not loaded yet
    const sourceWidth = getSourceWidth() || canvas.width || 640;
    const sourceHeight = getSourceHeight() || canvas.height || 480;
    
    // Calculate ROI points as percentages of video dimensions
    // Create a rectangle that's 60% of the video size, centered
    const roiWidth = sourceWidth * 0.6;
    const roiHeight = sourceHeight * 0.6;
    const offsetX = (sourceWidth - roiWidth) / 2;
    const offsetY = (sourceHeight - roiHeight) / 2;
    
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
 * ROI Drawing Functions
 */
function drawRoi() {
    if (!canvas || !ctx || !activeSource) return;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Calculate scaling factors
    const srcW = getSourceWidth();
    const srcH = getSourceHeight();

    if (srcW && srcH) {
        const videoAspect = srcW / srcH;
        const containerAspect = canvas.width / canvas.height;
        
        let displayWidth, displayHeight;
        // Match the video's CSS object-fit: contain behavior.
        if (videoAspect > containerAspect) {
            displayWidth = canvas.width;
            displayHeight = canvas.width / videoAspect;
        } else {
            displayHeight = canvas.height;
            displayWidth = canvas.height * videoAspect;
        }
        
        scale.x = displayWidth / srcW;
        scale.y = displayHeight / srcH;
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

        // Fill with semi-transparent color (optional)
        if (roiFillEnabled) {
            ctx.fillStyle = colors.accent + '20'; // Add transparency
            ctx.fill();
        }
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
        const maxX = getSourceWidth() || 640;
        const maxY = getSourceHeight() || 480;
        
        controlPoints[draggingControl][0] = Math.max(0, Math.min(frameX, maxX));
        controlPoints[draggingControl][1] = Math.max(0, Math.min(frameY, maxY));
        
        drawRoi();
        showPointTooltip(
            controlPoints[draggingControl][0] * scale.x + offset.x,
            controlPoints[draggingControl][1] * scale.y + offset.y,
            controlPoints[draggingControl][0],
            controlPoints[draggingControl][1]
        );
    } else if (draggingPoint !== null) {
        // Convert canvas coordinates back to frame coordinates for corner point
        const frameX = (mouseX - offset.x) / scale.x;
        const frameY = (mouseY - offset.y) / scale.y;
        
        // Clamp to frame boundaries
        const maxX = getSourceWidth() || 640;
        const maxY = getSourceHeight() || 480;
        
        roiPoints[draggingPoint][0] = Math.max(0, Math.min(frameX, maxX));
        roiPoints[draggingPoint][1] = Math.max(0, Math.min(frameY, maxY));
        
        // Update control points when corner points move
        updateControlPointsForCornerChange(draggingPoint);
        
        drawRoi();
        showPointTooltip(
            roiPoints[draggingPoint][0] * scale.x + offset.x,
            roiPoints[draggingPoint][1] * scale.y + offset.y,
            roiPoints[draggingPoint][0],
            roiPoints[draggingPoint][1]
        );
    } else {
        // Check if mouse is over any point for cursor change / coordinate tooltip
        let overPoint = false;
        let hoveredCanvasX = null, hoveredCanvasY = null, hoveredFrameX = null, hoveredFrameY = null;
        
        // Check control points first
        for (let i = 0; i < controlPoints.length; i++) {
            const canvasX = controlPoints[i][0] * scale.x + offset.x;
            const canvasY = controlPoints[i][1] * scale.y + offset.y;
            
            if (Math.abs(mouseX - canvasX) < 10 && Math.abs(mouseY - canvasY) < 10) {
                overPoint = true;
                hoveredCanvasX = canvasX;
                hoveredCanvasY = canvasY;
                hoveredFrameX = controlPoints[i][0];
                hoveredFrameY = controlPoints[i][1];
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
                    hoveredCanvasX = canvasX;
                    hoveredCanvasY = canvasY;
                    hoveredFrameX = roiPoints[i][0];
                    hoveredFrameY = roiPoints[i][1];
                    break;
                }
            }
        }
        
        if (overPoint) {
            showPointTooltip(hoveredCanvasX, hoveredCanvasY, hoveredFrameX, hoveredFrameY);
        } else {
            hidePointTooltip();
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
    hidePointTooltip();
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
            showPointTooltip(canvasX, canvasY, controlPoints[i][0], controlPoints[i][1]);
            return;
        }
    }
    
    // Check if touch is near any ROI corner point
    for (let i = 0; i < roiPoints.length; i++) {
        const canvasX = roiPoints[i][0] * scale.x + offset.x;
        const canvasY = roiPoints[i][1] * scale.y + offset.y;
        
        if (Math.abs(touchX - canvasX) < 25 && Math.abs(touchY - canvasY) < 25) { // Larger touch area
            draggingPoint = i;
            showPointTooltip(canvasX, canvasY, roiPoints[i][0], roiPoints[i][1]);
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
        const maxX = getSourceWidth() || 640;
        const maxY = getSourceHeight() || 480;
        
        controlPoints[draggingControl][0] = Math.max(0, Math.min(frameX, maxX));
        controlPoints[draggingControl][1] = Math.max(0, Math.min(frameY, maxY));
        
        drawRoi();
        showPointTooltip(
            controlPoints[draggingControl][0] * scale.x + offset.x,
            controlPoints[draggingControl][1] * scale.y + offset.y,
            controlPoints[draggingControl][0],
            controlPoints[draggingControl][1]
        );
    } else if (draggingPoint !== null) {
        // Convert canvas coordinates back to frame coordinates for corner point
        const frameX = (touchX - offset.x) / scale.x;
        const frameY = (touchY - offset.y) / scale.y;
        
        // Clamp to frame boundaries
        const maxX = getSourceWidth() || 640;
        const maxY = getSourceHeight() || 480;
        
        roiPoints[draggingPoint][0] = Math.max(0, Math.min(frameX, maxX));
        roiPoints[draggingPoint][1] = Math.max(0, Math.min(frameY, maxY));
        
        // Update control points when corner points move
        updateControlPointsForCornerChange(draggingPoint);
        
        drawRoi();
        showPointTooltip(
            roiPoints[draggingPoint][0] * scale.x + offset.x,
            roiPoints[draggingPoint][1] * scale.y + offset.y,
            roiPoints[draggingPoint][0],
            roiPoints[draggingPoint][1]
        );
    }
}

function onCanvasTouchEnd(event) {
    event.preventDefault();
    draggingPoint = null;
    draggingControl = null;
    hidePointTooltip();
}

/**
 * Window Event Handlers
 */
function onWindowResize() {
    if (canvas) {
        const container = document.getElementById('videoContainer');
        canvas.width = container.offsetWidth;
        canvas.height = container.offsetHeight;
        
        // Also resize result canvases.
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

/**
 * ROI point coordinate tooltip
 */
function showPointTooltip(canvasX, canvasY, frameX, frameY) {
    const tooltip = document.getElementById('roiPointTooltip');
    if (!tooltip) return;
    tooltip.textContent = `(${Math.round(frameX)}, ${Math.round(frameY)})`;
    tooltip.style.left = `${canvasX}px`;
    tooltip.style.top = `${canvasY}px`;
    tooltip.style.display = 'block';
}

function hidePointTooltip() {
    const tooltip = document.getElementById('roiPointTooltip');
    if (tooltip) {
        tooltip.style.display = 'none';
    }
}

function toggleControlPoints() {
    showControlPoints = !showControlPoints;
    const button = event.target;
    button.textContent = showControlPoints ? '🎛️ Hide Curves' : '🎛️ Show Curves';
    drawRoi();
    updateStatus(showControlPoints ? 'Curve controls visible' : 'Curve controls hidden');
}

function resetRoi() {
    // Reset ROI based on current video dimensions
    initializeRoiPoints();
    drawRoi();
    hidePointTooltip();
    updateStatus('ROI reset to default');
}

function toggleRoiFill() {
    // A completed press-and-hold on this same button resets the ROI instead;
    // ignore the click/touchend that follows it.
    if (roiFillLongPressTriggered) {
        roiFillLongPressTriggered = false;
        return;
    }
    roiFillEnabled = !roiFillEnabled;
    updateRoiFillButtonState();
    drawRoi();
    updateStatus(roiFillEnabled ? 'ROI area fill enabled' : 'ROI area is transparent');
}

function updateRoiFillButtonState() {
    // Legacy menu button (if present)
    const legacyButton = document.getElementById('toggleRoiFillBtn');
    if (legacyButton) {
        legacyButton.textContent = roiFillEnabled ? '⬜ ROI Area: Filled' : '🔳 ROI Area: Transparent';
    }

    // New compact icon in the instructions pill
    const iconButton = document.getElementById('toggleRoiFillIcon');
    if (iconButton) {
        // State: transparent when roiFillEnabled === false
        const transparentEnabled = !roiFillEnabled;
        iconButton.dataset.active = transparentEnabled.toString();
        iconButton.setAttribute('aria-pressed', transparentEnabled.toString());
        iconButton.title = roiFillEnabled
            ? 'ROI area: Filled (tap for transparent, hold to reset ROI)'
            : 'ROI area: Transparent (tap for filled, hold to reset ROI)';
    }
}

/**
 * Pressing and holding the ROI fill-toggle icon resets the ROI to its default rectangle
 */
function setupRoiFillHoldToReset() {
    const button = document.getElementById('toggleRoiFillIcon');
    if (!button || button.dataset.holdToResetBound === 'true') return;
    button.dataset.holdToResetBound = 'true';

    const startHold = () => {
        roiFillLongPressTriggered = false;
        clearTimeout(roiFillHoldTimer);
        button.classList.add('roi-fill-toggle--holding');
        roiFillHoldTimer = setTimeout(() => {
            roiFillLongPressTriggered = true;
            button.classList.remove('roi-fill-toggle--holding');
            button.classList.add('roi-fill-toggle--reset');
            setTimeout(() => button.classList.remove('roi-fill-toggle--reset'), 200);
            if (navigator.vibrate) {
                navigator.vibrate(30);
            }
            resetRoi();
        }, ROI_RESET_HOLD_DURATION_MS);
    };

    const cancelHold = () => {
        clearTimeout(roiFillHoldTimer);
        button.classList.remove('roi-fill-toggle--holding');
    };

    button.addEventListener('mousedown', startHold);
    button.addEventListener('mouseup', cancelHold);
    button.addEventListener('mouseleave', cancelHold);

    button.addEventListener('touchstart', startHold, { passive: true });
    button.addEventListener('touchend', cancelHold);
    button.addEventListener('touchcancel', cancelHold);
}
