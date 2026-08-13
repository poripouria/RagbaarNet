/**
* RagbaarNet AI Platform — beat-generator.js
* ============================================
* Independent rhythmic layer that runs on the same Tone.Transport (central clock).
* It ticks every "16n" — this rate is fixed and never changes.
* Hihat density (currentHihatLevel, from core.js) only determines which hihat
* layer is played, not the tick rate itself — so there is no need to cancel/reschedule
* when the engine speed changes.
* Hihat levels:
  low  = quarter-note pulse
  mid  = eighth-note pulse
  high = sixteenth-note pulse
  max  = sixteenth-note pulse with accents and phrase variation
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
            max:  [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1]  // Sixteenth-note pulse; energy comes from dynamics
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
            max:  [1,1,1,1, 1,1,1,1, 1,1,1,1]
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
            max:  [1,1,1,1,1,1, 1,1,1,1,1,1]  // Sixteenth-note pulse — busy "gallop" feel
        }
    }
};

function timeSignatureKey(timeSignature) {
    if (!Array.isArray(timeSignature) || timeSignature.length < 2) return "4/4";
    const key = `${timeSignature[0]}/${timeSignature[1]}`;
    return METER_PATTERNS[key] ? key : "4/4"; // Unknown meter → safe default to 4/4
}

let beatGeneratorEventId = null;
let beatStepIndex = 0;
let beatBarIndex = 0; // Number of completed bars, used for phrase-level variation

function getHihatVelocity(level, step, barIndex, meterKey) {
    // LOW: quarter-note pulse, stronger on the first step of the bar.
    if (level === "low") {
        if (step === 0) return 70;
        return 52;
    }

    // MID: eighth-note pulse, downbeats stronger than offbeats.
    if (level === "mid") {
        if (step % 4 === 0) return 68;
        return 48;
    }

    // HIGH: sixteenth-note pulse, layered accents.
    if (level === "high") {
        if (step % 4 === 0) return 65;
        if (step % 2 === 0) return 50;
        return 38;
    }

    // MAX: same 16n density as HIGH, but stronger dynamics and phrase ending.
    if (level === "max") {
        const isDownbeat = step % 4 === 0;
        const isOffbeat = step % 2 === 1;
        const isFillBar = barIndex % 4 === 3;

        if (isFillBar && step >= 12) return 82;
        if (isDownbeat) return 78;
        if (isOffbeat) return 58;
        return 46;
    }

    return 45;
}

function shouldPlayKick(pattern, step, barIndex) {
    // Base kick pattern.
    if (pattern.kick[step]) return true;

    // Every fourth bar: add a pickup kick on the final step.
    if (barIndex % 4 === 3 && step === pattern.steps - 1) return true;

    return false;
}

function getKickVelocity(step, barIndex, pattern) {
    // Strong downbeat.
    if (step === 0) return 105;

    // Phrase-ending pickup.
    if (barIndex % 4 === 3 && step === pattern.steps - 1) return 72;

    return 88;
}

function shouldPlaySnare(pattern, step, barIndex) {
    // Base snare pattern.
    if (pattern.snare[step]) return true;

    // Every fourth bar: subtle 4/4 pickup before the final downbeat.
    if (barIndex % 4 === 3 && pattern.steps === 16 && step === 14) return true;

    return false;
}

function getSnareVelocity(step, barIndex, pattern) {
    // Main backbeat accents.
    if (
        (pattern.steps === 16 && (step === 4 || step === 12)) ||
        (pattern.steps === 12 && (step === 4 || step === 8))
    ) {
        return 92;
    }

    // Fill/pickup hit.
    if (barIndex % 4 === 3) return 58;

    return 70;
}

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
    beatBarIndex = 0;

    beatGeneratorEventId = Tone.Transport.scheduleRepeat((time) => {
        const meterKey = timeSignatureKey(lastMusicStatus.timeSignature);
        const pattern = METER_PATTERNS[meterKey];
        const step = beatStepIndex % pattern.steps;

        // Kick
        if (shouldPlayKick(pattern, step, beatBarIndex)) {
            playDrumSound({ note: 36, velocity: getKickVelocity(step, beatBarIndex, pattern) }, time);
        }

        // Snare
        if (shouldPlaySnare(pattern, step, beatBarIndex)) {
            playDrumSound({ note: 38, velocity: getSnareVelocity(step, beatBarIndex, pattern) }, time);
        }

        // Hihat
        const hihatPattern = pattern.hihat[currentHihatLevel] ?? pattern.hihat.high;
        if (hihatPattern[step]) {
            const velocity = getHihatVelocity(currentHihatLevel, step, beatBarIndex, meterKey);
            playDrumSound({ note: 42, velocity }, time);
        }

        beatStepIndex++;

        // When the measure ends, move to the next bar.
        if (beatStepIndex % pattern.steps === 0) {
            beatBarIndex++;
        }
    }, "16n");

    console.log('🥁 BeatGenerator started (synced to Tone.Transport, "16n" grid)');
}

function stopBeatGenerator() {
    if (beatGeneratorEventId !== null) {
        Tone.Transport.clear(beatGeneratorEventId);
        beatGeneratorEventId = null;
    }

    beatStepIndex = 0;
    beatBarIndex = 0;
    console.log('🥁 BeatGenerator stopped');
}
