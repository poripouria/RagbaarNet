/**
 * RagbaarNet AI Platform — core.js
 * ================================
 * Shared constants, generic device/UI utilities, and cross-cutting app state
 * (musician/instrument/tempo/speed settings). Load this FIRST — every other
 * module relies on globals declared here (e.g. detectProcessorUrl(),
 * isMobileDevice(), the musician/tempo/volume state).
 *
 * NOTE ON MODULE STYLE: these files are loaded as plain classic <script> tags
 * (NOT type="module"). Classic scripts on the same page share one top-level
 * lexical scope, so every `let`/`const`/`function` declared in any of these
 * files is a normal global visible to all the others, exactly as if this was
 * still one big file — this split is purely organizational, the runtime
 * behavior is unchanged. Load order (core -> video-pipeline -> roi ->
 * audio-engine -> main) matters only because a couple of top-level variable
 * initializers call a function immediately (e.g. `let processorUrl =
 * detectProcessorUrl();` in video-pipeline.js), so the function's file must
 * load first.
 */

/**
 * AI Music Generation Platform - Main JavaScript Module
 * Handles video input sources, ROI drawing, and music generation
 */

// Global variables
let inputSource = null;

let settings = {};

// Music settings variables
let availableMusicians = [
    { id: 'rule-based', label: 'Rule-based Musician', description: 'Rule-based multi-instrument demo mapping (drums, bass, strings, etc.).' },
    { id: 'lstm-onessen', label: 'LSTM (Essen Folk Song)', description: 'Neural LSTM model trained on the Essen folk song collection.' },
    { id: 'lstm-onessen-orchestral', label: 'LSTM (Orchestral)', description: 'Just like the LSTM musician, but with orchestral instruments.' }
];

let currentMusicianType = 'lstm-onessen';

// Matches the processor's default musician on startup
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

let isDrumsEnabled = true;

let pendingDrumsEnabled = isDrumsEnabled;

// const timeSignatureOptions = [
//     { id: '4/4', label: '4/4', value: [4, 4] },
//     { id: '3/4', label: '3/4', value: [3, 4] },
//     { id: '6/8', label: '6/8', value: [6, 8] }
// ];

// let currentTimeSignature = [4, 4];

let pendingTempo = 120;

let isSwitchingMusician = false;

let musicianSwitchTimeoutId = null;

const MUSICIAN_SWITCH_TIMEOUT_MS = 8000;

let currentTempo = pendingTempo;

const TEMPO_MIN = 60;

const TEMPO_MAX = 300;

const SPEED_MIN = 0;

const SPEED_MAX = 160;

// Controlable, The higher the value, the slower the curve at low speeds. A value of 1.0 would be linear.
const N = 1.7;

let pendingSpeedKmh = speedFromTempo(pendingTempo);

let currentSpeedKmh = pendingSpeedKmh;

//Latest telemetry pushed by the processor
let latestTelemetry = { speed_kmh: null, accel: null, rpm: null };

const VOLUME_MIN = 0;

const VOLUME_MAX = 100;

const DEFAULT_VOLUME = 40;

let currentVolume = DEFAULT_VOLUME;

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

function updateStatus(message) {
    document.getElementById('statusText').textContent = message;
}

/**
 * Musician Selection Modal
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text == null ? '' : String(text);
    return div.innerHTML;
}

let lastBpmRampTime = 0;
const BPM_RAMP_MIN_INTERVAL_MS = 1000; // New ramp each 1s maximum, to avoid too many ramp calls in a short time

function syncTransportBpm(newTempo) {
    if (Tone.Transport.state !== 'started') return;
    const now = performance.now();
    if (now - lastBpmRampTime < BPM_RAMP_MIN_INTERVAL_MS) return;
    lastBpmRampTime = now;
    Tone.Transport.bpm.rampTo(newTempo, 1); // Ramp to new tempo over 1 second
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

function clampVolumeValue(value) {
    const parsedValue = Number.parseInt(value, 10);
    if (Number.isNaN(parsedValue)) {
        return currentVolume;
    }
    return Math.max(VOLUME_MIN, Math.min(VOLUME_MAX, parsedValue));
}

// --- Drum / Beat engine: RPM-driven hi-hat density ---
let currentHihatLevel = 'low'; // 'low' | 'mid' | 'high' | 'max'
const RPM_LOW_THRESHOLD = 1500;
const RPM_MID_THRESHOLD = 3000;
const RPM_HIGH_THRESHOLD = 4000;
const RPM_HYSTERESIS = 200; // Prevents rapid oscillation near borders

function calculateDrumDensityFromRpm(rpm) {
    if (!Number.isFinite(rpm)) return currentHihatLevel;

    if (currentHihatLevel === 'low' && rpm > RPM_LOW_THRESHOLD + RPM_HYSTERESIS) {
        currentHihatLevel = 'mid';
    } else if (currentHihatLevel === 'mid' && rpm > RPM_MID_THRESHOLD + RPM_HYSTERESIS) {
        currentHihatLevel = 'high';
    } else if (currentHihatLevel === 'high' && rpm > RPM_HIGH_THRESHOLD + RPM_HYSTERESIS) {
        currentHihatLevel = 'max';
    } else if (currentHihatLevel === 'max' && rpm < RPM_HIGH_THRESHOLD - RPM_HYSTERESIS) {
        currentHihatLevel = 'high';
    } else if (currentHihatLevel === 'high' && rpm < RPM_MID_THRESHOLD - RPM_HYSTERESIS) {
        currentHihatLevel = 'mid';
    } else if (currentHihatLevel === 'mid' && rpm < RPM_LOW_THRESHOLD - RPM_HYSTERESIS) {
        currentHihatLevel = 'low';
    }
    return currentHihatLevel;
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
