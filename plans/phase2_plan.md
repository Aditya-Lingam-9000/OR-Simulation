# Phase 2 Plan — Audio Capture & Streaming

**Date:** 2026-02-22  
**Depends on:** Phase 1 PASS ✅  
**Goal:** Robust mic capture, VAD, chunking policy, audio queue, replay files.

---

## Deliverables

1. **`src/ingest/mic_stream.py`** — Full mic capture + VAD + chunking engine
   - `sounddevice` InputStream for real-time mic audio
   - `webrtcvad` for speech/silence detection
   - Produces 0.5–2.0s speech chunks with configurable overlap
   - Async queue output for ASR worker consumption
   - Chunk save to `tmp/audio_chunks/` for replay/debugging

2. **`src/ingest/mic_server.py`** — Update existing skeleton with real integration
   - Wire `MicrophoneCapture` to use the new `MicStream` engine

3. **`src/ingest/audio_utils.py`** — Audio utility functions
   - WAV file I/O (load/save)
   - Audio format validation (16kHz, mono, 16-bit)
   - Synthetic audio generation for testing

4. **`scripts/test_mic_client.py`** — Local mic test script
   - Records 10s from mic, runs VAD, produces chunks
   - Saves chunks as WAV files for inspection

5. **`scripts/generate_test_audio.py`** — Generate synthetic test audio
   - Creates test WAV files (sine waves, silence, mixed)
   - Used by tests without requiring a real microphone

6. **`tests/test_audio.py`** — Comprehensive audio/VAD tests
   - WAV validation tests
   - VAD silence rejection
   - Chunk duration bounds
   - Queue semantics
   - Latency measurement

## Architecture

```
Microphone → sounddevice.InputStream
    ↓ (30ms frames)
WebRTC VAD → speech/silence classification
    ↓ (speech frames accumulated)
Chunk Builder → 0.5–2.0s AudioChunk objects
    ↓
asyncio.Queue → ASR Worker (Phase 3)
    ↓ (optional)
tmp/audio_chunks/*.wav → replay files
```

## Key Settings (from constants.py)
- AUDIO_SAMPLE_RATE = 16000 Hz
- AUDIO_CHANNELS = 1 (mono)
- AUDIO_BIT_DEPTH = 16
- VAD_FRAME_DURATION_MS = 30
- VAD_AGGRESSIVENESS = 2
- CHUNK_MIN_DURATION_S = 0.5
- CHUNK_MAX_DURATION_S = 2.0
- CHUNK_OVERLAP_MS = 200

## Dependency Notes
- `webrtcvad` needs `webrtcvad-wheels` on Windows (no MSVC required)
- Updated `requirements.txt`: `webrtcvad-wheels>=2.0.10` replaces `webrtcvad>=2.0.10`
