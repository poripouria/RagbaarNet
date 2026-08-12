/**
 * RagbaarNet AI Platform — audio-engine.js
 * ==========================================
 * Everything Tone.js: the master bus (EQ/compressor/limiter), the reverb
 * send system, every instrument factory (piano/strings/bass/guitars/pad/
 * synth), the drum kit, and note-on/note-off handling (playMusicEvent,
 * playTonalInstrument, playDrumSound, stopNote, the polyphony limiter).
 * Depends on core.js for currentInstrument/currentVolume/etc.
 */

// Throttle updates to 50ms (20 FPS)

// Audio system variables
let audioContext = null;

let masterGain = null;

let isMusicGenerationActive = false;

let activeNotes = new Map();

// Track currently playing (sustained, tonal) notes
let instrumentVoices = {};

// Store instrument voice settings
let musicEventQueue = [];

// Queue for scheduling music events
let recentPercussion = new Map();

// Track short-lived drum hits: key -> expiry timestamp
let lastMusicEventTime = 0;

let instrumentFactories = {};

// instrument name -> () => fresh { synth, nodes, release, isPluck }
let reverbBus = null;

// the reverb "tank" itself (100% wet — mix is handled via sends)
let reverbPreFilter = null;

// high-pass before the tank, so bass frequencies stay out of the reverb
let masterBusIn = null;

// everything (dry + wet) sums here before mastering
let drumsBus = null;

// small submix bus so percussion isn't swallowed by tonal dynamics
let masterEQ = null;

let masterCompressor = null;

let masterLimiter = null;

let lastMusicStatus = {
    eventCount: 0,
    tempo: currentTempo,
    keySignature: 'C_major',
    timeSignature: [4, 4],
    instruments: []
};

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

const QUANTIZE_GRID = "16n";

function handleMusicEvents(musicData) {
    try {
        if (!musicData || !musicData.events) {
            return;
        }

        console.log(`🎵 Received ${musicData.events.length} music events for frame ${musicData.frame_counter}`);

        // Nearest grid point on the central clock
        const scheduleTime = Tone.Transport.state === 'started'
        ? Tone.Transport.nextSubdivision(QUANTIZE_GRID)
        : Tone.now() + 0.02;    // Fallback

        // Schedule each music event
        musicData.events.forEach((event, index) => {
            playMusicEvent(event, scheduleTime);
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

    const timeSign = lastMusicStatus.timeSignature;
    if (timeSign && timeSign[0] && timeSign[1]) {
        parts.push(`${timeSign[0]}/${timeSign[1]}`);
    }

    if (lastMusicStatus.instruments.length > 0) {
        const instrumentLabel = lastMusicStatus.instruments.join(', ');
        parts.push(`Instruments: ${instrumentLabel}`);
    }

    const message = parts.length > 0 ? `🎵 ${parts.join(' • ')}` : `🎵 Tempo ${lastMusicStatus.tempo} BPM, ${lastMusicStatus.timeSignature[1]/lastMusicStatus.timeSignature[2]}`;
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
    const timesign = (musicData && musicData.time_signature) ? musicData.time_signature : lastMusicStatus.timeSignature;

    const instruments = getCurrentlyPlayingInstruments();
    const instrumentSummary = Object.entries(instruments)
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([instr, count]) => `${formatInstrumentName(instr)} (${count})`);

    lastMusicStatus = {
        eventCount,
        tempo: currentTempo,
        keySignature: key,
        timeSignature: timesign,
        instruments: instrumentSummary
    };

    updateMusicStatusDisplay();

    const roiInfo = document.getElementById('roiInfo');
    if (roiInfo) {
        roiInfo.innerHTML = '';
    }
}
