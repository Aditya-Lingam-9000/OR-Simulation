# Phase 2 Explanation — Audio Capture & Streaming

## What Was Done

Phase 2 implemented the **complete audio capture pipeline** — from raw microphone input through voice activity detection to neatly packaged speech chunks ready for ASR inference. The system works both in real-time (live microphone) and offline (pre-recorded / synthetic audio), so the entire pipeline can be tested in CI without a physical microphone.

---

## Why This Architecture?

### The Problem
OR-Symphony needs to listen to surgical team conversations in real-time. But a surgical OR is noisy — ventilators humming, monitors beeping, instruments clattering. We can't just send continuous audio to ASR; we need to:

1. **Detect when someone is speaking** (Voice Activity Detection)
2. **Cut the speech into manageable chunks** (not too short, not too long)
3. **Deliver chunks fast enough** for real-time processing (< 50ms latency)
4. **Bridge short pauses** like "um" or breathing mid-sentence
5. **Discard silence** to avoid wasting ASR compute

### The Solution: Three-Layer Pipeline

**Layer 1 — VADProcessor**: WebRTC VAD is the gold standard for speech detection. It works on tiny 30ms frames of raw PCM audio and returns a binary speech/silence verdict. We use aggressiveness level 2 (moderate) — aggressive enough to filter OR background noise, but not so aggressive we miss quiet speech.

**Layer 2 — ChunkBuilder**: Raw VAD frames are too small for ASR (30ms each). The ChunkBuilder accumulates speech frames into chunks of 0.5–2.0 seconds. Key intelligence:
- **Pause bridging**: Up to 300ms of silence within speech is kept (people pause naturally)
- **Min duration gate**: Chunks under 0.5s are discarded (too short for useful transcription)  
- **Max duration cap**: At 2.0s the chunk is force-cut (keeps latency bounded)
- **Overlap**: 200ms of audio carries over to the next chunk (prevents cutting words)

**Layer 3 — MicStream**: Ties sounddevice (audio capture), VADProcessor, and ChunkBuilder together. The sounddevice callback runs in a dedicated audio thread; chunks are transferred to the main asyncio event loop via `call_soon_threadsafe()`.

---

## Module Details

### `src/ingest/mic_stream.py`

The heart of Phase 2. Three classes and one convenience function:

**`VADProcessor`** — Wraps `webrtcvad.Vad`. Validates WebRTC's strict requirements:
- Only 8/16/32/48 kHz sample rates
- Only 10/20/30 ms frame durations
- Input must be exact byte count (frame_size × 2 for 16-bit)

Also provides `process_audio(numpy_array)` → `List[VADFrame]` for batch processing.

**`ChunkBuilder`** — Finite state machine with three states:
1. **Idle** (no speech seen yet) — silence frames are discarded
2. **In speech** — speech and bridging-silence frames accumulate in buffer
3. **End of speech** — when silence exceeds 10 consecutive frames (300ms), chunk is finalized

On finalization: trailing silence is stripped, PCM bytes are concatenated, converted to float32, duration is checked against min/max bounds, overlap is saved for next chunk.

**`MicStream`** — Real-time microphone capture class:
- Opens `sounddevice.InputStream` with `blocksize=frame_size` so each callback gets exactly one VAD frame
- Callback runs VAD classification and feeds ChunkBuilder
- When a chunk is ready, uses `loop.call_soon_threadsafe(asyncio.ensure_future, ...)` to safely cross from audio thread to async world
- Queue has drop-oldest policy when full (100 chunks max)
- Optional WAV file saving per chunk for debugging

**`process_audio_buffer()`** — Sync function that processes a numpy array through the same VAD + ChunkBuilder pipeline. Used in tests and for re-processing recorded audio.

### `src/ingest/audio_utils.py`

**WAV I/O**: `load_wav()` and `save_wav()` with automatic format conversion (int16 ↔ float32), mono mixdown, and directory creation.

**Validation**: `validate_wav()` checks sample rate, channels, bit depth, and duration bounds. Returns a structured `AudioValidationResult` with errors, warnings, and metadata. `validate_audio_array()` does the same for in-memory arrays (checks NaN, Inf, amplitude range).

**Synthetic audio generators** (crucial for testing without a microphone):
- `generate_sine(freq, duration)` — pure tone
- `generate_silence(duration)` — zeros
- `generate_white_noise(duration)` — random noise, seeded for reproducibility
- `generate_speech_like(duration)` — multi-harmonic signal (150/300/450/600 Hz + noise) that mimics voice spectral characteristics enough to trigger VAD
- `generate_speech_silence_pattern(speech_s, silence_s, reps)` — alternating segments for testing chunk boundaries

### `src/ingest/mic_server.py` (updated)

The high-level `MicrophoneCapture` class now:
- **Live mode**: `await start()` creates a MicStream and captures from mic
- **Offline mode**: `feed_audio(numpy_array)` processes pre-recorded audio synchronously via `process_audio_buffer()`
- Lazy imports MicStream to avoid sounddevice dependency issues at import time

### Scripts

**`scripts/generate_test_audio.py`** — Generates 5 WAV files in `tmp/test_audio/`, validates each, reports results. Run once to populate test data.

**`scripts/test_mic_client.py`** — Interactive mic test. Records for N seconds, shows chunk events live, saves WAVs, validates output. Command line: `--duration 30 --no-save`.

---

## Testing Strategy

45 new tests organized in 8 test classes:

| Class | Tests | What It Covers |
|-------|-------|---------------|
| `TestGenerateSyntheticAudio` | 8 | Shape, dtype, amplitude bounds, reproducibility |
| `TestWavIO` | 4 | Save/load roundtrip, directory creation, error handling |
| `TestValidateWav` | 4 | Valid/invalid files, sample rate, duration bounds |
| `TestValidateAudioArray` | 4 | Valid arrays, NaN/Inf detection, amplitude warnings |
| `TestVADProcessor` | 8 | Init validation, speech/silence classification, frame processing |
| `TestChunkBuilder` | 8 | Empty/silence/speech, min/max duration, flush, IDs, dtype |
| `TestProcessAudioBuffer` | 5 | Silence→0 chunks, speech→chunks, duration bounds, timestamps |
| `TestMicrophoneCaptureFeedAudio` | 3 | Offline processing via `feed_audio()` |
| `TestVADFalsePositive` | 1 | Master plan requirement: 30s silence < 2 chunks |

Key design decision: All tests use synthetic audio (no microphone needed), making them fully CI-compatible.

---

## Dependency Note: webrtcvad-wheels

The original `webrtcvad` package requires Microsoft Visual C++ 14.0+ to compile from source on Windows. We switched to `webrtcvad-wheels` which ships pre-built wheels — same API, no build tools needed. Updated in `requirements.txt`.

---

## How This Feeds Phase 3

Phase 3 will implement **MedASR ONNX inference**. The connection points:

1. `AudioChunk.audio` (float32, 16kHz, mono) → direct input to `OnnxASRRunner.transcribe()`
2. `process_audio_buffer()` → used in ASR tests to generate chunks from synthetic audio
3. The async `MicStream.get_chunk()` → consumed by `ASRWorker` in the live pipeline
4. WAV files in `tmp/test_audio/` → used for ASR accuracy testing with known inputs
