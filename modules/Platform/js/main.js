/**
 * RagbaarNet AI Platform — main.js
 * ==================================
 * Bootstrap (window.onload, setupEventListeners) and all direct UI-control
 * handlers: the musician/instrument/tempo settings modal, the volume slider,
 * the music start/stop/pause/screenshot buttons. Loads LAST — it's what
 * wires every other module's functions to DOM events and to `window` (so
 * inline onclick="..." handlers in UI.html keep working exactly as before).
 */

/**
 * Application Initialization
 */
window.onload = function() {
    setupEventListeners();
    showInputSelection();
    updateMusicButton();
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

    // Custom scrollbars are hidden via CSS; wire up drag-to-scroll (mouse)
    enableDragToScroll(document.querySelector('.menu-frame'));
    enableDragToScroll(document.querySelector('.musician-modal-body'));
    enableDragToScroll(document.getElementById('instrumentList'));
    enableWheelToHorizontalScroll(document.getElementById('instrumentList'));

    // Initialize frame processing
    initializeFrameProcessing();
}

/**
 * Custom-scrollbar replacement: lets mouse users click-and-drag to scroll a
 * container (touch users already get native drag/momentum scrolling once the
 * OS scrollbar is hidden via CSS). A short move-threshold distinguishes a
 * genuine drag from a plain click/tap, and the resulting click is swallowed
 * so buttons inside the container do not fire after a drag gesture.
 */
function enableDragToScroll(el) {
    if (!el) return;

    const DRAG_THRESHOLD_PX = 6;
    const NON_DRAG_SELECTOR = 'input, textarea, select, a[href]';

    let isPointerDown = false;
    let hasDragged = false;
    let startX = 0;
    let startY = 0;
    let startScrollLeft = 0;
    let startScrollTop = 0;

    const suppressNextClick = (event) => {
        event.preventDefault();
        event.stopPropagation();
        el.removeEventListener('click', suppressNextClick, true);
    };

    const endDrag = () => {
        if (!isPointerDown) return;
        isPointerDown = false;
        el.classList.remove('is-drag-scrolling');
        if (hasDragged) {
            el.addEventListener('click', suppressNextClick, true);
        }
        hasDragged = false;
    };

    el.addEventListener('pointerdown', (event) => {
        if (event.pointerType !== 'mouse') return;
        if (event.target.closest(NON_DRAG_SELECTOR)) return;

        isPointerDown = true;
        hasDragged = false;
        startX = event.clientX;
        startY = event.clientY;
        startScrollLeft = el.scrollLeft;
        startScrollTop = el.scrollTop;
    });

    el.addEventListener('pointermove', (event) => {
        if (!isPointerDown) return;

        const deltaX = event.clientX - startX;
        const deltaY = event.clientY - startY;

        if (!hasDragged && (Math.abs(deltaX) > DRAG_THRESHOLD_PX || Math.abs(deltaY) > DRAG_THRESHOLD_PX)) {
            hasDragged = true;
            el.classList.add('is-drag-scrolling');
        }

        if (hasDragged) {
            el.scrollLeft = startScrollLeft - deltaX;
            el.scrollTop = startScrollTop - deltaY;
            event.preventDefault();
        }
    });

    el.addEventListener('pointerup', endDrag);
    el.addEventListener('pointercancel', endDrag);
    el.addEventListener('pointerleave', (event) => {
        if (event.pointerType === 'mouse') endDrag();
    });
}

/**
 * Lets a plain mouse wheel (which only reports vertical delta by default)
 * scroll a horizontally-scrolling container, such as the instrument list.
 */
function enableWheelToHorizontalScroll(el) {
    if (!el) return;

    el.addEventListener('wheel', (event) => {
        if (event.deltaY === 0 || event.deltaX !== 0) return;
        event.preventDefault();
        el.scrollLeft += event.deltaY;
    }, { passive: false });
}

function getMusicianLabel(musicianId) {
    const found = availableMusicians.find(m => m.id === musicianId);
    return found ? found.label : (musicianId || 'Unknown');
}

function setMusicianModalStatus(text) {
    const statusEl = document.getElementById('musicianModalStatus');
    if (statusEl) {
        statusEl.textContent = text || '';
    }
}

function setMusicianListInteractive(interactive) {
    const container = document.getElementById('musicianList');
    if (container) {
        container.classList.toggle('musician-list--busy', !interactive);
    }
}

function renderMusicianList() {
    const container = document.getElementById('musicianList');
    if (!container) return;

    container.innerHTML = '';

    availableMusicians.forEach(musician => {
        const option = document.createElement('button');
        option.type = 'button';
        const isSelected = musician.id === pendingMusicianSelection;
        option.className = 'musician-option' + (isSelected ? ' selected' : '');
        option.setAttribute('aria-pressed', isSelected.toString());
        option.innerHTML = `
            <div class="musician-option-name">
                <span class="musician-option-label">${escapeHtml(musician.label)}</span>
                <span class="musician-option-name-right">
                    <span class="musician-option-badge"> ✓ </span>
                    <span class="musician-option-info" tabindex="0" role="button" aria-label="${escapeHtml(musician.label)} info">
                        <img class="musician-option-info-icon" src="../../assets/icons/round-information-outline-white-icon.png" alt="" aria-hidden="true" draggable="false">
                        <span class="musician-option-desc">${escapeHtml(musician.description || '')}</span>
                    </span>
                </span>
            </div>
        `;

        const selectHandler = (event) => {
            event.preventDefault();
            event.stopPropagation();
            if (isSwitchingMusician) return;
            pendingMusicianSelection = musician.id;
            renderMusicianList();
            updateInstrumentControls();
            setMusicianModalStatus(`Selected: ${getMusicianLabel(musician.id)}`);
            updateMusicianApplyButton();
        };

        option.addEventListener('click', selectHandler);
        option.addEventListener('touchend', selectHandler);

        // The info icon only reveals the description tooltip (via CSS :hover/:active) -
        // stop its clicks/touches from bubbling up and triggering musician selection.
        const infoIcon = option.querySelector('.musician-option-info');
        if (infoIcon) {
            const stopBubble = (event) => event.stopPropagation();
            infoIcon.addEventListener('click', stopBubble);
            infoIcon.addEventListener('touchstart', stopBubble, { passive: true });
            infoIcon.addEventListener('touchend', stopBubble);
        }

        container.appendChild(option);
    });
}

function updateMusicianApplyButton() {
    const applyBtn = document.getElementById('musicianApplyBtn');
    if (!applyBtn) return;

    const hasSelection = !!pendingMusicianSelection;
    applyBtn.disabled = !hasSelection || isSwitchingMusician;
    applyBtn.classList.toggle('is-disabled', applyBtn.disabled);
}

function openMusicianModal() {
    const modal = document.getElementById('musicianModal');
    if (!modal) return;

    pendingMusicianSelection = currentMusicianType;
    pendingInstrument = currentInstrument;
    pendingSpeedKmh = currentSpeedKmh;
    renderMusicianList();
    updateInstrumentControls();
    updateTempoControls(pendingSpeedKmh);
    setMusicianModalStatus('Adjust the settings and tap Apply.');
    setMusicianListInteractive(!isSwitchingMusician);
    updateMusicianApplyButton();
    modal.style.display = 'flex';

    // Refresh from the server in case the list or current selection changed elsewhere
    if (segmentationSocket && segmentationSocket.connected) {
        segmentationSocket.emit('get_available_musicians');
    }
}

function closeMusicianModal() {
    const modal = document.getElementById('musicianModal');
    if (modal) {
        pendingMusicianSelection = currentMusicianType;
        pendingInstrument = currentInstrument;
        pendingSpeedKmh = currentSpeedKmh;
        updateMusicianApplyButton();
        modal.style.display = 'none';
    }
}

function applyMusicSettings() {
    if (!pendingMusicianSelection) {
        setMusicianModalStatus('Please select a musician first.');
        return;
    }

    if (!segmentationSocket || !segmentationSocket.connected) {
        setMusicianModalStatus('⚠️ Not connected to processor - cannot update music settings');
        return;
    }

    pendingSpeedKmh = latestTelemetry.speed_kmh != null ? clampSpeedValue(latestTelemetry.speed_kmh) : pendingSpeedKmh;
    pendingTempo = calculateAutoTempoFromSpeed(pendingSpeedKmh);

    isSwitchingMusician = true;
    setMusicianListInteractive(false);
    updateMusicianApplyButton();
    setMusicianModalStatus('Applying music settings...');
    segmentationSocket.emit('set_music_settings', {
        musician_type: pendingMusicianSelection,
        instrument: pendingInstrument,
        tempo: pendingTempo
    });

    clearTimeout(musicianSwitchTimeoutId);
    musicianSwitchTimeoutId = setTimeout(() => {
        if (!isSwitchingMusician) return;
        isSwitchingMusician = false;
        setMusicianListInteractive(true);
        updateMusicianApplyButton();
        setMusicianModalStatus('⚠️ No response from processor - please try again');
    }, MUSICIAN_SWITCH_TIMEOUT_MS);
}

function selectMusician(musicianId) {
    if (isSwitchingMusician) return;
    pendingMusicianSelection = musicianId;
    renderMusicianList();
    updateInstrumentControls();
    setMusicianModalStatus(`Selected: ${getMusicianLabel(musicianId)}`);
    updateMusicianApplyButton();
}

function renderInstrumentList() {
    const container = document.getElementById('instrumentList');
    if (!container) return;

    container.innerHTML = '';

    instrumentOptions.forEach(instrument => {
        const chip = document.createElement('button');
        chip.type = 'button';
        const isSelected = instrument.id === pendingInstrument;
        chip.className = 'instrument-chip' + (isSelected ? ' selected' : '');
        chip.setAttribute('role', 'option');
        chip.setAttribute('aria-selected', isSelected.toString());
        chip.innerHTML = `
            <img class="instrument-chip-icon" src="${instrument.icon}" alt="" aria-hidden="true" draggable="false">
            <span class="instrument-chip-label">${escapeHtml(instrument.label)}</span>
        `;

        const selectHandler = (event) => {
            event.preventDefault();
            pendingInstrument = instrument.id;
            renderInstrumentList();
            updateMusicianApplyButton();
        };

        chip.addEventListener('click', selectHandler);
        chip.addEventListener('touchend', selectHandler);
        container.appendChild(chip);
    });
}

function updateInstrumentControls() {
    const settings = document.getElementById('instrumentSettings');
    const showInstrument = pendingMusicianSelection === 'lstm-onessen';

    if (settings) settings.hidden = !showInstrument;
    renderInstrumentList();
}

function updateTempoControls(speedValue) {
    pendingSpeedKmh = clampSpeedValue(speedValue);
    pendingTempo = calculateAutoTempoFromSpeed(pendingSpeedKmh);

    const tempoValueEl = document.getElementById('tempoDerivedValue');
    const speedValueEl = document.getElementById('speedDerivedValue');

    if (tempoValueEl) tempoValueEl.textContent = pendingTempo;
    if (speedValueEl) speedValueEl.textContent = pendingSpeedKmh;
}

function updateVolumeControls(value) {
    currentVolume = clampVolumeValue(value);

    const slider = document.getElementById('volumeSlider');
    if (slider) {
        slider.value = currentVolume;
    }

    if (masterGain) {
        masterGain.gain.value = currentVolume / 100;
    }
}

function handleVolumeSliderInput() {
    updateVolumeControls(this.value);
}

async function startMusicGeneration() {
    // Unlock/resume the underlying (Tone.js) audio context — required by browsers
    try {
        await Tone.start();
        console.log('✅ AudioContext resumed by user gesture');
    } catch (err) {
        console.warn('⚠️ Unable to resume audio context:', err);
        return;
    }

    if (!masterGain) {
        initializeAudioSystem();
    }

    if (isMusicGenerationActive) {
        stopMusicGeneration();
    } else {
        // Central clock
        Tone.Transport.bpm.value = currentTempo;
        Tone.Transport.start();
        initBeatGenerator();

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
    stopBeatGenerator();

    // Clear all scheduled events on the Transport and stop the clock
    Tone.Transport.cancel();
    Tone.Transport.stop();
    
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
            musicBtn.style.backgroundColor = '';
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
    
    if (activeSource && activeSource.tagName === 'VIDEO') {
        if (isPaused) {
            activeSource.pause();
        } else {
            activeSource.play();
        }
    }
    // Note: MJPEG streams (IMG) cannot be paused via DOM API, they just continue in background
}

function takeScreenshot() {
    const srcW = getSourceWidth();
    const srcH = getSourceHeight();

    if (activeSource && srcW && srcH) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = srcW;
        canvas.height = srcH;
        
        ctx.drawImage(activeSource, 0, 0);
        
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
