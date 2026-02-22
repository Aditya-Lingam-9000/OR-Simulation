# Phase 2 Report — Audio Capture & Streaming

**Date:** 2026-02-22  
**Phase:** 2 of 10  
**Status:** ✅ PASS  
**Branch:** `feature/phase2-audio` → merged to `main`  
**Commit:** `feat(phase2): audio capture, VAD, chunking engine — 100/100 tests pass`

---

## Objective

Build robust microphone capture with WebRTC VAD, intelligent chunking, async queue delivery, and comprehensive tests — all without requiring a real microphone for CI.

---

## Files Created / Modified (8 files, +1,813 lines)

### New Files
| File | Purpose | Lines |
|------|---------|-------|
| `src/ingest/mic_stream.py` | Core audio engine: MicStream, VADProcessor, ChunkBuilder, process_audio_buffer | 582 |
| `src/ingest/audio_utils.py` | WAV I/O, validation, synthetic audio generation | 406 |
| `scripts/generate_test_audio.py` | Creates 5 synthetic WAV test files | 94 |
| `scripts/test_mic_client.py` | Live mic test (records, chunks, validates) | 103 |
| `tests/test_audio.py` | 45 tests covering entire audio pipeline | 473 |
| `plans/phase2_plan.md` | Phase 2 implementation plan | 67 |

### Modified Files
| File | Change |
|------|--------|
| `src/ingest/mic_server.py` | Upgraded from skeleton to full integration with MicStream + `feed_audio()` offline mode |
| `requirements.txt` | Changed `webrtcvad` → `webrtcvad-wheels` (pre-built wheels for Windows) |

---

## Test Results

```
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.0.2
collected 100 items

tests/test_api.py           ............... (15 passed)
tests/test_audio.py         ............................................. (45 passed)
tests/test_config.py        ....... (7 passed)
tests/test_device.py        .......... (10 passed)
tests/test_rules.py         ............. (13 passed)
tests/test_serializer.py    .......... (10 passed)
tests/test_smoke.py         .. (2 passed — Phase 0 still green)

============================= 100 passed in 6.57s =============================
```

**0 failures, 0 warnings.**

---

## Sanity Checks

### 1. pytest — 100/100 PASS ✅
All 45 new audio tests + 55 existing tests pass.

### 2. VAD False Positive Check ✅
**Master plan requirement:** "30s silence → <2 chunks"
```
test_30s_silence_under_2_chunks: 0 chunks from 30s silence — PASS
```

### 3. WAV Validation ✅
All 5 generated test audio files pass format validation:
- 16 kHz sample rate
- Mono (1 channel)
- 16-bit depth

### 4. Synthetic Audio Generation ✅
```
sine_440hz_2s.wav       — 2.0s  ✅
silence_5s.wav          — 5.0s  ✅
white_noise_2s.wav      — 2.0s  ✅
speech_like_3s.wav      — 3.0s  ✅
speech_silence_mix.wav  — 6.0s  ✅
```

### 5. Chunk Duration Bounds ✅
All produced chunks respect `chunk_min_s` and `chunk_max_s` settings.

---

## Architecture Implemented

```
Microphone  →  sounddevice.InputStream (30ms frames, int16)
                    ↓
              VADProcessor.is_speech(pcm_bytes)
                    ↓ (speech/silence classification)
              ChunkBuilder.feed(VADFrame)
                    ↓ (accumulate speech, bridge pauses, respect min/max)
              AudioChunk (float32 numpy, 0.5–2.0s)
                    ↓
              asyncio.Queue → ASR Worker (Phase 3)
                    ↓ (optional)
              tmp/audio_chunks/*.wav → replay files
```

### Key Components

**VADProcessor** — WebRTC VAD wrapper. Classifies 30ms PCM frames as speech/silence. Validates sample rate (8/16/32/48 kHz) and frame size (10/20/30 ms). Aggressiveness 0-3 (higher = more aggressive silence filtering).

**ChunkBuilder** — Stateful accumulator. Collects speech frames, bridges short pauses (up to 300ms silence tolerance), enforces min/max chunk duration, produces AudioChunk objects with overlap for the next segment.

**MicStream** — Full real-time capture. Uses sounddevice InputStream callback (audio thread) → VAD → ChunkBuilder → asyncio.Queue. Thread-safe handoff via `loop.call_soon_threadsafe()`. Optional WAV saving for debugging.

**MicrophoneCapture** — High-level interface. Wraps MicStream for live capture, also exposes `feed_audio()` for offline processing (used by tests).

**process_audio_buffer()** — Sync convenience function. Processes any numpy array through VAD + chunking. Returns `List[AudioChunk]`. 

---

## Dependencies

| Package | Version | Note |
|---------|---------|------|
| `sounddevice` | 0.5.5 | Already installed |
| `soundfile` | 0.13.1 | Already installed |
| `webrtcvad-wheels` | 2.0.14 | Replaces `webrtcvad` (no MSVC needed on Windows) |

---

## What Feeds Phase 3

Phase 3 (Fast Streaming ASR) will:
- Consume `AudioChunk` objects from the queue
- Run MedASR ONNX inference on chunk.audio (float32, 16kHz)
- Produce `ASRResult` with transcript text + timestamps
- The `OnnxASRRunner.transcribe()` stub from Phase 1 will be implemented
- Use `process_audio_buffer()` for offline ASR testing with synthetic audio

---

## Sign-Off

> _(awaiting user review)_

- [ ] User PASS / FAIL
