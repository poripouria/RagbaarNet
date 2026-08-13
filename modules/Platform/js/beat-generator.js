/**
 * RagbaarNet AI Platform — beat-generator.js
 * ============================================
 * Independent rhythmic layer that runs on the same Tone.Transport (central clock).
 * It ticks every "16n" — this rate is fixed and never changes.
 * Hihat density (currentHihatLevel, from core.js) only determines which hihat
 * layer is played, not the tick rate itself — so there is no need to cancel/reschedule
 * when the engine speed changes.
 * Depends on: core.js (currentHihatLevel), audio-engine.js (playDrumSound,
 * lastMusicStatus). Must be loaded after audio-engine.js.
 */

// Each array is indexed at "16n" resolution (sixteenth note). steps = measure length
// in sixteenth notes for the corresponding meter.
const METER_PATTERNS = {
    "4/4": {
        steps: 16,
        kick:  [1,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0], // Beats 1 and 3
        snare: [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0], // Beats 2 and 4 (standard backbeat)
        hihat: {
            low:  [1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0], // Quarter-note pulse
            mid:  [1,0,1,0, 1,0,1,0, 1,0,1,0, 1,0,1,0], // Eighth-note pulse
            high: [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1], // Sixteenth-note pulse
            max:  [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1], // Sixteenth-note pulse
        }
    },
    "3/4": {
        steps: 12,
        kick:  [1,0,0,0, 0,0,0,0, 0,0,0,0], // Beat 1 only (waltz feel)
        snare: [0,0,0,0, 1,0,0,0, 1,0,0,0], // Beats 2 and 3
        hihat: {
            low:  [1,0,0,0, 1,0,0,0, 1,0,0,0],
            mid:  [1,0,1,0, 1,0,1,0, 1,0,1,0],
            high: [1,1,1,1, 1,1,1,1, 1,1,1,1],
            max:  [1,1,1,1, 1,1,1,1, 1,1,1,1],
        }
    },
    "6/8": {
        // Compound meter: 6 eighth notes = 12 sixteenth-note steps, with two main pulses
        // (dotted quarter notes on eighth notes 1 and 4, i.e. steps 0 and 6), not six equal ticks.
        steps: 12,
        kick:  [1,0,0,0,0,0, 1,0,0,0,0,0], // Two main compound pulses
        snare: [0,0,0,0,0,0, 0,0,1,0,0,0], // Decorative accent on the fifth eighth note
        hihat: {
            low:  [1,0,0,0,0,0, 1,0,0,0,0,0], // Dotted-quarter pulse — 2 hits per measure
            mid:  [1,0,1,0,1,0, 1,0,1,0,1,0], // Eighth-note pulse — natural 6/8 feel (6 hits)
            high: [1,1,1,1,1,1, 1,1,1,1,1,1], // Sixteenth-note pulse — busy "gallop" feel
            max:  [1,1,1,1,1,1, 1,1,1,1,1,1], // Sixteenth-note pulse — busy "gallop" feel
        }
    },
};

function timeSignatureKey(timeSignature) {
    if (!Array.isArray(timeSignature) || timeSignature.length < 2) return "4/4";
    const key = `${timeSignature[0]}/${timeSignature[1]}`;
    return METER_PATTERNS[key] ? key : "4/4"; // Unknown meter → safe default to 4/4
}

let beatGeneratorEventId = null;
let beatStepIndex = 0;

function setDrumsEnabled(enabled) {
    isDrumsEnabled = enabled;

    if (!isMusicGenerationActive) return;
    if (enabled) {
        initBeatGenerator();
    } else {
        stopBeatGenerator();
    }
}

function initBeatGenerator() {
    if (beatGeneratorEventId !== null) return; // Already running

    beatStepIndex = 0;

    beatGeneratorEventId = Tone.Transport.scheduleRepeat((time) => {
        const meterKey = timeSignatureKey(lastMusicStatus.timeSignature);
        const pattern = METER_PATTERNS[meterKey];
        const step = beatStepIndex % pattern.steps;

        if (pattern.kick[step]) {
            playDrumSound({ note: 36, velocity: 75 }, time); // kick
        }
        if (pattern.snare[step]) {
            playDrumSound({ note: 38, velocity: 65 }, time); // snare
        }
        const hihatPattern = pattern.hihat[currentHihatLevel] || pattern.hihat.low;
        if (hihatPattern[step]) {
            playDrumSound({ note: 42, velocity: 45 }, time); // hihat
        }

        beatStepIndex++;
    }, "16n");

    console.log('🥁 BeatGenerator started (synced to Tone.Transport, "16n" grid)');
}

function stopBeatGenerator() {
    if (beatGeneratorEventId !== null) {
        Tone.Transport.clear(beatGeneratorEventId);
        beatGeneratorEventId = null;
    }
    beatStepIndex = 0;
    console.log('🥁 BeatGenerator stopped');
}
