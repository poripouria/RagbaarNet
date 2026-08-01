/**
 * AI Music Generation Platform - Main JavaScript Module
 * Handles video input sources, ROI drawing, and music generation
 */

// Global variables
let inputSource = null;
let videoElement = null;
let canvas = null;
let ctx = null;
let segmentationCanvas = null;
let segmentationCtx = null;
let roiPoints = [];                     // Will be initialized based on video dimensions
let controlPoints = [];                 // Bézier control points for curves
let draggingPoint = null;
let draggingControl = null;
let scale = {x: 1, y: 1};
let offset = {x: 0, y: 0};
let isPaused = false;
let settings = {};
let showControlPoints = true;
let roiFillEnabled = true;              // When false, ROI area is transparent (outline + vertices still visible)
let roiFillHoldTimer = null;
let roiFillLongPressTriggered = false;
const ROI_RESET_HOLD_DURATION_MS = 600;

// Frame processing variables
let frameProcessingEnabled = true;      // Always keep processing enabled
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
let isProcessingFrame = false;      // Prevent concurrent frame processing
let lastUpdateTime = 0;
let updateThrottleInterval = 50;    // Throttle updates to 50ms (20 FPS)

// Audio system variables
let audioContext = null;
let masterGain = null;
let isMusicGenerationActive = false;
let activeNotes = new Map();        // Track currently playing (sustained, tonal) notes
let instrumentVoices = {};          // Store instrument voice settings
let musicEventQueue = [];           // Queue for scheduling music events
let recentPercussion = new Map();   // Track short-lived drum hits: key -> expiry timestamp
let lastMusicEventTime = 0;

let instrumentFactories = {};       // instrument name -> () => fresh { synth, nodes, release, isPluck }
let reverbBus = null;               // the reverb "tank" itself (100% wet — mix is handled via sends)
let reverbPreFilter = null;         // high-pass before the tank, so bass frequencies stay out of the reverb
let masterBusIn = null;             // everything (dry + wet) sums here before mastering
let drumsBus = null;                // small submix bus so percussion isn't swallowed by tonal dynamics
let masterEQ = null;
let masterCompressor = null;
let masterLimiter = null;

// Music settings variables
let availableMusicians = [
    { id: 'rule-based', label: 'Rule-based Musician', description: 'Rule-based multi-instrument demo mapping (drums, bass, strings, etc.).' },
    { id: 'continuous_pianist', label: 'Continuous Pianist', description: 'Piano musician with sustained/continuous note playback.' },
    { id: 'lstm-onessen', label: 'LSTM (Essen Folk Song)', description: 'Neural LSTM model trained on the Essen folk song collection.' },
    { id: 'lstm-onessen-orchestral', label: 'LSTM (Orchestral)', description: 'Just like the LSTM musician, but with orchestral instruments.' }
];
let currentMusicianType = 'lstm-onessen'; // Matches the processor's default musician on startup
let pendingMusicianSelection = null;
const instrumentOptions = [
    { id: 'piano', label: 'Piano', icon: '../../assets/icons/instruments/piano.png' },
    { id: 'electric_piano', label: 'Electric Piano', icon: '../../assets/icons/instruments/elec-piano.png' },
    { id: 'strings', label: 'Strings', icon: '../../assets/icons/instruments/violin.png' },
    { id: 'bass', label: 'Bass', icon: '../../assets/icons/instruments/bass.png' },
    { id: 'electric_guitar', label: 'Electric Guitar', icon: '../../assets/icons/instruments/elec-guitar.png' },
    { id: 'acoustic_guitar', label: 'Acoustic Guitar', icon: '../../assets/icons/instruments/guitar.png' },
    { id: 'pad', label: 'Pad', icon: '../../assets/icons/instruments/pad.png' },
    { id: 'synth', label: 'Synth', icon: '../../assets/icons/instruments/synth.png' }
];
let currentInstrument = 'piano';
let pendingInstrument = currentInstrument;
let pendingTempo = 120;
let isSwitchingMusician = false;
let musicianSwitchTimeoutId = null;
const MUSICIAN_SWITCH_TIMEOUT_MS = 8000;
let currentTempo = 120;
const TEMPO_MIN = 60;
const TEMPO_MAX = 300;
const SPEED_MIN = 0;
const SPEED_MAX = 160;
// Controlable, The higher the value, the slower the curve at low speeds. A value of 1.0 would be linear.
const N = 1.7;
let pendingSpeedKmh = speedFromTempo(120);
let currentSpeedKmh = pendingSpeedKmh;
//Latest telemetry pushed by the processor
let latestTelemetry = { speed_kmh: null, accel: null, rpm: null };
const VOLUME_MIN = 0;
const VOLUME_MAX = 100;
const DEFAULT_VOLUME = 40;
let currentVolume = DEFAULT_VOLUME;
let lastMusicStatus = {
    eventCount: 0,
    tempo: currentTempo,
    keySignature: 'C_major',
    instruments: []
};
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

function setInstructionsText(text) {
    const instructionsTextEl = document.getElementById('instructionsText');
    if (instructionsTextEl) {
        instructionsTextEl.textContent = text;
        return;
    }

    // Fallback for older markup
    const instructions = document.querySelector('.instructions');
    if (instructions) {
        instructions.textContent = text;
    }
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

    // Custom scrollbars are hidden via CSS; wire up drag-to-scroll (mouse)
    enableDragToScroll(document.querySelector('.menu-frame'));
    enableDragToScroll(document.querySelector('.musician-modal-body'));
    enableDragToScroll(document.getElementById('instrumentList'));
    enableWheelToHorizontalScroll(document.getElementById('instrumentList'));

    // Initialize frame processing
    initializeFrameProcessing();
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
 * Audio System Functions (Tone.js synthesis engine)
 */

function initializeAudioSystem() {
    try {
        // Tone.js manages its own internal AudioContext.
        const initialVolume = clampVolumeValue(document.getElementById('volumeSlider')?.value ?? DEFAULT_VOLUME);
        masterGain = new Tone.Gain(initialVolume / 100).toDestination();

        // Volume slider control
        const volumeSlider = document.getElementById('volumeSlider');
        if (volumeSlider) {
            volumeSlider.addEventListener('input', handleVolumeSliderInput);
        }

        updateVolumeControls(initialVolume);

        // --- Mastering chain (sits right before the final volume stage) ---
        // EQ: shave a touch of low-mud, add a little "air" on top.
        // Compressor: gently glues everything together so quiet/loud events feel cohesive.
        // Limiter: safety net so nothing ever clips, even with several instruments stacked.
        masterLimiter = new Tone.Limiter(-1).connect(masterGain);
        masterCompressor = new Tone.Compressor({
            threshold: -8,
            ratio: 2,
            attack: 0.03,
            release: 0.15
        }).connect(masterLimiter);
        masterEQ = new Tone.EQ3({ low: -1, mid: 0, high: 1.5 }).connect(masterCompressor);
        masterBusIn = new Tone.Gain(1).connect(masterEQ);

        // Initialize instrument voices
        initializeInstrumentVoices();

        console.log('🎵 Audio system initialized successfully (Tone.js)');
        updateStatus('Audio system ready');

    } catch (error) {
        console.error('❌ Failed to initialize audio system:', error);
        updateStatus('Audio initialization failed');
    }
}


/**
 * Connects a voice's final node to the mix as a proper AUX SEND: the dry signal goes straight to 
 the master bus at full level, and a separate, independently-controlled copy is sent into the shared 
 reverb tank at `sendAmount` (0-1).
 * This is the standard mixing-console approach — it lets every instrument have its own reverb amount 
 (bass stays tight and dry, pads/strings get washed in space) instead of one fixed wet% for everything.
 * Returns the send Gain node (if any) so callers can add it to their disposable `nodes` list.
 */
function connectWithReverbSend(node, sendAmount, velocity = 1) {
    node.connect(masterBusIn);
    if (sendAmount > 0 && reverbPreFilter) {
        const send = new Tone.Gain(sendAmount * (0.5 + velocity * 0.5)).connect(reverbPreFilter);
        node.connect(send);
        return send;
    }
    return null;
}

// Per-instrument output trim (0-1 multiplier applied on top of velocity, right before triggerAttack)
const INSTRUMENT_OUTPUT_TRIM = {
    piano: 1.0,
    electric_piano: 0.95,
    strings: 0.6,
    bass: 1.05,
    electric_guitar: 0.95,
    acoustic_guitar: 1.0,
    pad: 0.85,
    synth: 0.9
};

// Hard cap on simultaneous voices PER INSTRUMENT. This is a safety net independent of the shared-effects
const MAX_POLYPHONY_PER_INSTRUMENT = 6;

function enforcePolyphonyLimit(instrument) {
    let count = 0;
    let oldestKey = null;
    for (const [key, data] of activeNotes) {
        if (data.instrument === instrument) {
            count++;
            if (oldestKey === null) oldestKey = key;
        }
    }
    if (count >= MAX_POLYPHONY_PER_INSTRUMENT && oldestKey !== null) {
        const sepIndex = oldestKey.indexOf('-');
        const oldChannel = Number(oldestKey.slice(0, sepIndex));
        const oldNote = Number(oldestKey.slice(sepIndex + 1));
        stopNote(oldNote, oldChannel);
    }
}

function initializeInstrumentVoices() {
    // The reverb "tank": pre-delay -> high-pass-> Freeverb itself
    const reverbPredelay = new Tone.Delay(0.03);
    reverbPreFilter = new Tone.Filter(250, 'highpass').connect(reverbPredelay);
    reverbBus = new Tone.Freeverb({ roomSize: 0.6, dampening: 2500 });
    reverbBus.wet.value = 1;
    reverbPredelay.connect(reverbBus);
    reverbBus.connect(masterBusIn);

    // --- Shared FX buses for the instruments that use an LFO-based effect (Chorus/Tremolo) ---
    // These used to be built FRESH inside every factory call only the (cheap) per-note Tone.Synth
    // itself is created and disposed per note now.
    const pianoFilter = new Tone.Filter(2600, 'lowpass');
    connectWithReverbSend(pianoFilter, 0.22);
    const pianoChorus = new Tone.Chorus(4, 2.5, 0.25).connect(pianoFilter).start();

    const epFilter = new Tone.Filter(1800, 'lowpass');
    connectWithReverbSend(epFilter, 0.15);
    const epTremolo = new Tone.Tremolo(4, 0.3).connect(epFilter).start();

    const stringsFilter = new Tone.Filter(2800, 'lowpass');
    connectWithReverbSend(stringsFilter, 0.24);
    const stringsChorus = new Tone.Chorus(3.2, 3.5, 0.4).connect(stringsFilter).start();

    const padFilter = new Tone.Filter(1400, 'lowpass');
    connectWithReverbSend(padFilter, 0.4);
    const padChorus = new Tone.Chorus(2.2, 4, 0.5).connect(padFilter).start();

    // Each tonal instrument is a FACTORY that builds a small, self-contained per-note voice
    // (just the synth itself for piano/electric_piano/strings/pad, since their filter/chorus/
    // reverb-send are now shared buses above; synth + filter [+ distortion] + send for the
    // MonoSynth-based instruments below, which don't use an LFO effect and are cheap enough
    // to keep fully per-note). Building a fresh synth per note-on (instead of sharing one
    // Tone.PolySynth across every note of that instrument) means note-off always calls
    // triggerRelease() on the *exact* instance that was triggered — there is no shared
    // "which internal voice is this note?" bookkeeping for Tone to get wrong.

    // `velocity` (0-1) is passed into every factory so envelope shape itself — not just
    // loudness — reacts to how "hard" the note was hit
    instrumentFactories = {
        piano: (velocity = 1) => {
            const release = 0.9 + velocity * 0.6;
            const synth = new Tone.Synth({
                oscillator: { type: 'fatsawtooth4' },
                envelope: {
                    attack: 0.006,
                    decay: 0.25 + velocity * 0.25,   // louder = longer decay
                    sustain: 0.18 + velocity * 0.25,
                    release: release
                }
            }).connect(pianoChorus);
            return { synth, nodes: [synth], release, isSharedBus: true };
        },
        electric_piano: (velocity = 1) => {
            const release = 0.55 + velocity * 0.45;
            const synth = new Tone.Synth({
                oscillator: { type: 'fmsquare' },
                envelope: {
                    attack: 0.006,
                    decay: 0.14 + velocity * 0.14,
                    sustain: 0.22 + velocity * 0.25,
                    release: release
                }
            }).connect(epTremolo);
            return { synth, nodes: [synth], release, isSharedBus: true };
        },
        strings: (velocity = 1) => {
            // Bow feel
            const release = 1.3 + velocity * 0.5;
            const synth = new Tone.Synth({
                oscillator: { type: 'fatsawtooth', count: 3, spread: 30 },
                envelope: {
                    attack: 0.28 - velocity * 0.12,
                    decay: 0.2,
                    sustain: 0.5 + velocity * 0.2,   // was a flat 0.8 — see stringsFilter note above
                    release: release
                }
            }).connect(stringsChorus);
            return { synth, nodes: [synth], release, isSharedBus: true };
        },
        bass: (velocity = 1) => {
            // MonoSynth's filterEnvelope gives the punchy "pluck then settle" character
            // real basses have — a plain oscillator+lowpass (the old design) sounds flat.
            const filter = new Tone.Filter(700, 'lowpass');
            const send = connectWithReverbSend(filter, 0.03, velocity);
            const synth = new Tone.MonoSynth({
                oscillator: { type: 'fmsine' },
                envelope: {
                    attack: 0.02,
                    decay: 0.2 + velocity * 0.15,
                    sustain: 0.45 + velocity * 0.25,
                    release: 0.5
                },
                filterEnvelope: {
                    attack: 0.008,
                    decay: 0.18 + velocity * 0.25,
                    sustain: 0.25 + velocity * 0.4,
                    release: 0.45,
                    baseFrequency: 70,
                    octaves: 2.8 + velocity * 1.2   // louder = brighter filter sweep
                }
            }).connect(filter);
            return { synth, nodes: [synth, filter, send].filter(Boolean), release: 0.6 };
        },
        electric_guitar: (velocity = 1) => {
            // Harder picking = more grit (distortion amount scales with velocity) and a
            // brighter filter sweep, mimicking how a real amp reacts to pick attack dynamics.
            const dist = new Tone.Distortion(0.25 + velocity * 0.35);
            const send = connectWithReverbSend(dist, 0.12, velocity);
            const filter = new Tone.Filter(1800 + velocity * 1800, 'lowpass').connect(dist);
            const synth = new Tone.MonoSynth({
                oscillator: { type: 'fatsawtooth', count: 3, spread: 25 },
                envelope: {
                    attack: 0.003,
                    decay: 0.1 + velocity * 0.08,
                    sustain: 0.28 + velocity * 0.22,
                    release: 0.3 + velocity * 0.25
                },
                filterEnvelope: {
                    attack: 0.001,
                    decay: 0.15,
                    sustain: 0.35,
                    release: 0.3,
                    baseFrequency: 300 + velocity * 400,
                    octaves: 3.2
                }
            }).connect(filter);
            return { synth, nodes: [synth, filter, dist, send].filter(Boolean), release: 0.4 };
        },
        acoustic_guitar: (velocity = 1) => {
            const bodyShelf = new Tone.Filter({ type: 'lowshelf', frequency: 180, gain: 3 });
            const highpass = new Tone.Filter(75, 'highpass');
            bodyShelf.connect(highpass);
            const send = connectWithReverbSend(highpass, 0.2, velocity);
            const synth = new Tone.PluckSynth({
                attackNoise: 0.8 + velocity * 0.6,   // 0.8 -> 1.4 (Tone's own default is 1)
                dampening: 3500 + velocity * 2000,   // 3500 -> 5500: harder pluck = brighter
                resonance: 0.82 + velocity * 0.12    // 0.82 -> 0.94: real ring/sustain, still stable
            }).connect(bodyShelf);
            return { synth, nodes: [synth, bodyShelf, highpass, send].filter(Boolean), release: 1.2, isPluck: true };
        },
        pad: (velocity = 1) => {
            const release = 2.0 + velocity * 0.8;
            const synth = new Tone.Synth({
                oscillator: { type: 'fatsine', count: 3, spread: 40 },
                envelope: {
                    attack: 0.5 + (1 - velocity) * 0.3,  // softer hits bloom in more slowly
                    decay: 0.6,
                    sustain: 0.65 + velocity * 0.2,
                    release: release
                }
            }).connect(padChorus);
            return { synth, nodes: [synth], release, isSharedBus: true };
        },
        synth: (velocity = 1) => {
            const filter = new Tone.Filter(2200, 'lowpass');
            const send = connectWithReverbSend(filter, 0.15, velocity);
            const synth = new Tone.MonoSynth({
                oscillator: { type: 'fatsquare', count: 2, spread: 25 },
                envelope: {
                    attack: 0.01,
                    decay: 0.15 + velocity * 0.15,
                    sustain: 0.22 + velocity * 0.2,
                    release: 0.4 + velocity * 0.3
                },
                filterEnvelope: {
                    attack: 0.01,
                    decay: 0.2 + velocity * 0.15,
                    sustain: 0.25 + velocity * 0.2,
                    release: 0.5,
                    baseFrequency: 400 + velocity * 400,
                    octaves: 2.5
                }
            }).connect(filter);
            return { synth, nodes: [synth, filter, send].filter(Boolean), release: 0.5 };
        }
    };

    // --- Drums ---
    // Drums get their OWN submix bus (drumsBus) instead of hitting masterBusIn directly.
    drumsBus = new Tone.Gain(1.35).connect(masterBusIn);

    function connectDrumWithReverbSend(node, sendAmount) {
        node.connect(drumsBus);
        if (sendAmount > 0 && reverbPreFilter) {
            const send = new Tone.Gain(sendAmount).connect(reverbPreFilter);
            node.connect(send);
            return send;
        }
        return null;
    }

    const snareFilter = new Tone.Filter(1800, 'highpass');
    connectDrumWithReverbSend(snareFilter, 0.14);
    const genericFilter = new Tone.Filter(1000, 'bandpass');
    connectDrumWithReverbSend(genericFilter, 0.1);
    const crashFilter = new Tone.Filter(6000, 'highpass');
    connectDrumWithReverbSend(crashFilter, 0.28); // crashes love room/reverb, unlike kick

    instrumentVoices = {
        drums: {
            kick: (() => {
                const node = new Tone.MembraneSynth({
                    pitchDecay: 0.045,
                    octaves: 6,
                    envelope: { attack: 0.001, decay: 0.35, sustain: 0, release: 0.4 }
                });
                connectDrumWithReverbSend(node, 0.03);
                return node;
            })(),
            snare: new Tone.NoiseSynth({
                noise: { type: 'white' },
                envelope: { attack: 0.001, decay: 0.18, sustain: 0 }
            }).connect(snareFilter),
            hihat: (() => {
                const hihatFilter = new Tone.Filter(3500, 'highpass');
                connectDrumWithReverbSend(hihatFilter, 0.08);
                const node = new Tone.MetalSynth({
                    envelope: { attack: 0.001, decay: 0.16, release: 0.05 },
                    harmonicity: 5.1,
                    modulationIndex: 32,
                    resonance: 5000,
                    octaves: 1.5
                }).connect(hihatFilter);
                node.volume.value = 2;
                return node;
            })(),
            crash: (() => {
                const node = new Tone.MetalSynth({
                    envelope: { attack: 0.001, decay: 1.4, release: 0.4 },
                    harmonicity: 3.1,
                    modulationIndex: 16,
                    resonance: 3000,
                    octaves: 2.5
                }).connect(crashFilter);
                node.volume.value = 1;
                return node;
            })(),
            generic: new Tone.NoiseSynth({
                noise: { type: 'pink' },
                envelope: { attack: 0.001, decay: 0.2, sustain: 0 }
            }).connect(genericFilter)
        }
    };
}

function handleMusicEvents(musicData) {
    try {
        if (!musicData || !musicData.events) {
            return;
        }

        console.log(`🎵 Received ${musicData.events.length} music events for frame ${musicData.frame_counter}`);

        // Slight delay between events to avoid overwhelming
        const scheduleTime = Tone.now() + 0.02;
        // // human feel
        // const jitter = (Math.random() - 0.5) * 0.008;
        // Schedule each music event
        musicData.events.forEach((event, index) => {
            playMusicEvent(event, scheduleTime); // + jitter
        });

        // Update UI with music info
        updateMusicInfo(musicData);

    } catch (error) {
        console.error('❌ Error handling music events:', error);
    }
}

function playMusicEvent(event, scheduleTime) {

    const type = event.event_type || event.type;
    const channel = event.channel !== undefined ? event.channel : 0;

    let instrument = event.instrument || currentInstrument;

    if (type === "note_off") {
        stopNote(event.note, channel);
        return;
    }
    if (!isMusicGenerationActive) return;

    if (channel === 9 || instrument === "drums") {
        playDrumSound(event, scheduleTime);
    } else {
        playTonalInstrument(event, instrument, channel, scheduleTime);
    }
}

function disposeVoiceSoon(voice) {
    // Give the release tail (or, for plucks, the natural decay) time to finish before
    // tearing down the nodes, so we don't clip/click the tail off.
    setTimeout(() => {
        try {
            voice.nodes.forEach(n => n.dispose && n.dispose());
        } catch (e) { /* already disposed, ignore */ }
    }, (voice.release + 0.3) * 1000);
}

function playTonalInstrument(event, instrument, channel, scheduleTime) {

    // Never have two voices fighting over the same pitch.
    const voiceKey = `${channel ?? 0}-${event.note}`;
    stopNote(event.note, channel);

    // Use the sent instrument name, but fall back to piano if it's unknown/invalid.
    const factoryName = normalizeInstrumentName(instrument, 'piano');

    // Voice-stealing safety net: caps how many notes of this instrument can ring at once
    enforcePolyphonyLimit(factoryName);

    const factory = instrumentFactories[factoryName] || instrumentFactories.piano;

    const noteName = Tone.Frequency(event.note, "midi").toNote();

    // Velocity mapping: humans hear loudness LOGARITHMICALLY, but Tone's triggerAttack velocity
    // multiplies gain LINEARLY.
    const rawVelocity = Math.min(1, Math.max(0, (event.velocity ?? 100) / 127));
    const velocity = Math.max(0.15, Math.pow(rawVelocity, 0.6));

    // Envelope shape (decay/sustain/release/brightness) reacts to the RAW velocity curve above;
    const trim = INSTRUMENT_OUTPUT_TRIM[factoryName] ?? 1;
    const ampVelocity = Math.min(1, Math.max(0.01, velocity * trim));

    const voice = factory(velocity);

    try {
        voice.synth.triggerAttack(noteName, scheduleTime, ampVelocity);
    } catch (e) {
        console.warn('⚠️ Tone.js triggerAttack error:', e);
        disposeVoiceSoon(voice);
        return;
    }

    activeNotes.set(voiceKey, { voice, instrument: factoryName, channel });

    // Safety timeout (in case a NoteOff never arrives from the backend)
    const timeout = (voice.release + 4) * 1000;
    setTimeout(() => {
        const current = activeNotes.get(voiceKey);
        if (current && current.voice === voice) {
            stopNote(event.note, channel);
        }
    }, timeout);
}

function playDrumSound(event, scheduleTime) {

    const rawDrumVelocity = Math.min(1, Math.max(0, (event.velocity ?? 100) / 127));
    const velocity = Math.max(0.2, Math.pow(rawDrumVelocity, 0.6));
    const drumType = getDrumType(event.note);
    const drumVoices = instrumentVoices.drums || {};
    const voice = drumVoices[drumType] || drumVoices.generic;

    if (!voice)
        return;

    try {
        if (typeof Tone !== 'undefined' && voice instanceof Tone.MembraneSynth) {
            const noteName = Tone.Frequency(48, "midi").toNote();
            voice.triggerAttackRelease(noteName, "8n", scheduleTime, velocity);
        } else if (typeof Tone !== 'undefined' && voice instanceof Tone.MetalSynth) {
            voice.triggerAttackRelease(200, "8n", scheduleTime, velocity);
        } else {
            voice.triggerAttackRelease("16n", scheduleTime, velocity);
        }
        const PERCUSSION_VISIBILITY_MS = 600;
        recentPercussion.set(`${drumType}-${Date.now()}`, Date.now() + PERCUSSION_VISIBILITY_MS);
    } catch (e) {
        console.warn('⚠️ Tone.js drum trigger error:', e);
    }
}

function stopNote(note, channel = 0) {

    const voiceKey = `${channel}-${note}`;
    const voiceData = activeNotes.get(voiceKey);

    if (!voiceData) return;

    const { voice } = voiceData;

    try {
        if (!voice.isPluck) {
            voice.synth.triggerRelease(Tone.now());
        }
        // PluckSynth has no triggerRelease — it just rings out and gets disposed below.
    } catch(e){}

    disposeVoiceSoon(voice);
    activeNotes.delete(voiceKey);
}

function hardStopAllAudio() {
    // A "panic" stop
    activeNotes.forEach(({ voice }) => {
        try {
            voice.nodes.forEach(n => n.dispose && n.dispose());
        } catch (e) { /* ignore */ }
    });
    activeNotes.clear();
    recentPercussion.clear();

    if (masterGain && typeof Tone !== 'undefined') {
        const now = Tone.now();
        masterGain.gain.cancelScheduledValues(now);
        masterGain.gain.setValueAtTime(0, now);
        masterGain.gain.linearRampToValueAtTime(masterGain.gain.value || 0.3, now + 0.05);
    }

    console.log('🛑 Hard stop: all audio silenced immediately');
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

    activeNotes.forEach((voiceData, key) => {
        const sepIndex = key.indexOf('-');
        const channel = Number(key.slice(0, sepIndex));
        const note = Number(key.slice(sepIndex + 1));
        stopNote(note, channel);
    });
    activeNotes.clear();
    recentPercussion.clear();

    console.log("🔇 All notes stopped");
}

function normalizeInstrumentName(instrument, fallback = 'piano') {

    const raw = String(instrument || '').trim().toLowerCase();
    if (!raw || raw === 'unknown' || raw === 'none' || raw === 'null') {
        return fallback;
    }
    if (raw === 'piano_only') {
        return 'piano';
    }
    return raw;
}

function formatInstrumentName(instrument) {
    const normalized = normalizeInstrumentName(instrument);
    return normalized
        .replace(/_/g, ' ')
        .replace(/\b\w/g, char => char.toUpperCase());
}

function updateMusicStatusDisplay() {
    const parts = [];
    if (lastMusicStatus.eventCount > 0) {
        parts.push(`${lastMusicStatus.eventCount} events`);
    }
    parts.push(`${clampTempoValue(lastMusicStatus.tempo)} BPM`);

    const keyLabel = String(lastMusicStatus.keySignature || 'C_major').replace(/_/g, ' ');
    if (keyLabel) {
        parts.push(keyLabel);
    }

    if (lastMusicStatus.instruments.length > 0) {
        const instrumentLabel = lastMusicStatus.instruments.join(', ');
        parts.push(`Instruments: ${instrumentLabel}`);
    }

    const message = parts.length > 0 ? `🎵 ${parts.join(' • ')}` : `🎵 Tempo ${lastMusicStatus.tempo} BPM`;
    updateStatus(message);
}

function getCurrentlyPlayingInstruments() {
    const now = Date.now();
    const instruments = {};

    // Sustained tonal notes that are still actually ringing
    activeNotes.forEach(voiceData => {
        const instr = normalizeInstrumentName(voiceData.instrument, 'piano');
        instruments[instr] = (instruments[instr] || 0) + 1;
    });

    // Drum hits have no sustain to track, so keep them visible for a short window
    recentPercussion.forEach((expiresAt, key) => {
        if (expiresAt <= now) {
            recentPercussion.delete(key);
        } else {
            instruments['drums'] = (instruments['drums'] || 0) + 1;
        }
    });

    return instruments;
}

function updateMusicInfo(musicData) {
    const eventCount = (musicData && Array.isArray(musicData.events)) ? musicData.events.length : 0;
    const key = (musicData && musicData.key_signature) ? musicData.key_signature : lastMusicStatus.keySignature;

    const instruments = getCurrentlyPlayingInstruments();
    const instrumentSummary = Object.entries(instruments)
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([instr, count]) => `${formatInstrumentName(instr)} (${count})`);

    lastMusicStatus = {
        eventCount,
        tempo: currentTempo,
        keySignature: key,
        instruments: instrumentSummary
    };

    updateMusicStatusDisplay();

    const roiInfo = document.getElementById('roiInfo');
    if (roiInfo) {
        roiInfo.innerHTML = '';
    }
}

/**
 * Frame Processing and Transmission
*/
function captureAndSendFrame() {
    // Prevent concurrent frame processing
    if (isProcessingFrame || !videoElement || !videoElement.videoWidth) {
        return;
    }
    
    const now = Date.now();
    if (now - lastFrameSentTime < frameSendInterval) {
        return; // Rate limiting
    }
    
    const currentFrameId = frameCounter++;
    lastFrameSentTime = now;
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
    const videoElement = document.getElementById('videoElement');
    const roiCanvas = document.getElementById('roiCanvas');
    const segCanvas = document.getElementById('segmentationCanvas');
    
    if (segmentationDisplayEnabled) {
        // Show segmentation overlay with ROI (hide video but keep ROI visible)
        if (videoElement) {
            // IMPORTANT: Using display:none can freeze frame updates in some browsers.
            // Keep the video in the render tree and hide it visually instead.
            videoElement.style.display = 'block';
            videoElement.style.visibility = 'hidden';
            videoElement.style.opacity = '0';
            videoElement.style.pointerEvents = 'none';
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
        // Show original video with ROI (hide segmentation display, but processing continues)
        if (videoElement) {
            videoElement.style.display = 'block';
            videoElement.style.visibility = 'visible';
            videoElement.style.opacity = '1';
            videoElement.style.pointerEvents = '';
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
                // Ensure playback starts (some browsers require an explicit play() call)
                const playPromise = videoElement.play();
                if (playPromise && typeof playPromise.then === 'function') {
                    playPromise.then(() => {
                        updateStatus('Connected - Receiving camera feed');
                        console.log('Camera stream started and playing');
                    }).catch(err => {
                        // Playback may be blocked by autoplay policies; still keep stream attached
                        console.warn('Camera stream attached but playback blocked:', err);
                        updateStatus('Connected - Camera attached (tap play to start)');
                    });
                } else {
                    updateStatus('Connected - Receiving camera feed');
                    console.log('Camera stream started (play() not required)');
                }
            })
            .catch(err => {
                console.error('Camera error details:', err);
                updateStatus(`Error: Could not access camera - ${err.message}`);
                
                // Fallback: try with basic constraints
                navigator.mediaDevices.getUserMedia({ video: true })
                    .then(stream => {
                        videoElement.srcObject = stream;
                        const playPromise = videoElement.play();
                        if (playPromise && typeof playPromise.then === 'function') {
                            playPromise.then(() => {
                                updateStatus('Connected - Using fallback camera settings');
                            }).catch(err => {
                                console.warn('Fallback camera attached but playback blocked:', err);
                                updateStatus('Connected - Camera attached (tap play to start)');
                            });
                        } else {
                            updateStatus('Connected - Using fallback camera settings');
                        }
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
        // For screen recording on desktop, prefer getDisplayMedia (screen capture)
        if (navigator.mediaDevices && typeof navigator.mediaDevices.getDisplayMedia === 'function') {
            navigator.mediaDevices.getDisplayMedia({ video: true })
                .then(stream => {
                    // Some browsers provide a MediaStream with a single video track for the screen
                    videoElement.srcObject = stream;
                    const playPromise = videoElement.play();
                    if (playPromise && typeof playPromise.then === 'function') {
                        playPromise.then(() => {
                            updateStatus('Connected - Screen capture started');
                            console.log('Screen capture started and playing');
                        }).catch(err => {
                            console.warn('Screen capture attached but playback blocked:', err);
                            updateStatus('Connected - Screen attached (tap play to start)');
                        });
                    } else {
                        updateStatus('Connected - Screen capture started');
                    }
                })
                .catch(err => {
                    console.warn('Screen capture failed, falling back to camera:', err);
                    updateStatus('Screen capture denied or unavailable - falling back to camera');

                    // Fallback to camera if screen capture is denied or unsupported at runtime
                    navigator.mediaDevices.getUserMedia({ video: true })
                        .then(camStream => {
                            videoElement.srcObject = camStream;
                            const playPromise = videoElement.play();
                            if (playPromise && typeof playPromise.then === 'function') {
                                playPromise.then(() => {
                                    updateStatus('Connected - Using camera as fallback');
                                }).catch(playErr => {
                                    console.warn('Fallback camera attached but playback blocked:', playErr);
                                    updateStatus('Connected - Camera attached (tap play to start)');
                                });
                            } else {
                                updateStatus('Connected - Using camera as fallback');
                            }
                        })
                        .catch(camErr => {
                            console.error('Fallback camera error:', camErr);
                            updateStatus('Error: Could not access screen or camera');
                        });
                });
        } else {
            // getDisplayMedia not supported; use camera as fallback
            console.warn('getDisplayMedia not supported in this browser - using camera fallback');
            updateStatus('Screen capture not supported - using camera fallback');
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(stream => {
                    videoElement.srcObject = stream;
                    const playPromise = videoElement.play();
                    if (playPromise && typeof playPromise.then === 'function') {
                        playPromise.then(() => {
                            updateStatus('Connected - Using camera as fallback');
                        }).catch(err => {
                            console.warn('Fallback camera attached but playback blocked:', err);
                            updateStatus('Connected - Camera attached (tap play to start)');
                        });
                    } else {
                        updateStatus('Connected - Using camera as fallback');
                    }
                })
                .catch(err => {
                    console.error('Fallback camera error:', err);
                    updateStatus('Error: Could not access camera');
                });
        }
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
        // Match the video's CSS object-fit: contain behavior.
        if (videoAspect > containerAspect) {
            displayWidth = canvas.width;
            displayHeight = canvas.width / videoAspect;
        } else {
            displayHeight = canvas.height;
            displayWidth = canvas.height * videoAspect;
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
        const maxX = videoElement.videoWidth || 640;
        const maxY = videoElement.videoHeight || 480;
        
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
        const maxX = videoElement.videoWidth || 640;
        const maxY = videoElement.videoHeight || 480;
        
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
        const maxX = videoElement.videoWidth || 640;
        const maxY = videoElement.videoHeight || 480;
        
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
        const maxX = videoElement.videoWidth || 640;
        const maxY = videoElement.videoHeight || 480;
        
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


/**
 * Musician Selection Modal
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text == null ? '' : String(text);
    return div.innerHTML;
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

function clampTempoValue(value) {
    const parsedValue = Number.parseInt(value, 10);
    if (Number.isNaN(parsedValue)) {
        return currentTempo;
    }
    return Math.max(TEMPO_MIN, Math.min(TEMPO_MAX, parsedValue));
}

function clampSpeedValue(value) {
    const parsedValue = Number.parseInt(value, 10);
    if (Number.isNaN(parsedValue)) {
        return pendingSpeedKmh;
    }
    return Math.max(SPEED_MIN, Math.min(SPEED_MAX, parsedValue));
}

// Inverse of calculateAutoTempoFromSpeed(); used to position the speed
// slider to match a known tempo value (e.g. coming from the processor).
function calculateAutoTempoFromSpeed(speedKmh) {
    const v = Number(speedKmh);
    if (!Number.isFinite(v) || v <= 0) return TEMPO_MIN;

    const ratio = Math.pow(Math.min(v, SPEED_MAX) / SPEED_MAX, N);
    const bpm = TEMPO_MIN + (TEMPO_MAX - TEMPO_MIN) * ratio;
    return clampTempoValue(Math.round(bpm));
}

// The exact inverse of the above formula — no approximation involved
function speedFromTempo(bpm) {
    const clampedBpm = Math.max(TEMPO_MIN, Math.min(TEMPO_MAX, Number(bpm) || TEMPO_MIN));
    const ratio = (clampedBpm - TEMPO_MIN) / (TEMPO_MAX - TEMPO_MIN);
    const v = SPEED_MAX * Math.pow(ratio, 1 / N);
    return Math.max(SPEED_MIN, Math.min(SPEED_MAX, Math.round(v)));
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


function clampVolumeValue(value) {
    const parsedValue = Number.parseInt(value, 10);
    if (Number.isNaN(parsedValue)) {
        return currentVolume;
    }
    return Math.max(VOLUME_MIN, Math.min(VOLUME_MAX, parsedValue));
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
