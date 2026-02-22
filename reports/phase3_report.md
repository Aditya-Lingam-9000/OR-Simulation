# Phase 3 Report: Fast Streaming ASR (MedASR on CPU via ONNX)

**Date:** 2026-02-22  
**Status:** PASS ✅  
**Tests:** 142 total (42 new + 100 prior), all passing  
**Commit:** `9dc6972` on `main`

---

## Objective

Implement the complete MedASR ONNX inference pipeline: raw audio → fbank features → ONNX model → CTC decode → text. All within the 400ms latency target on CPU.

---

## Deliverables

| Item | File | Status |
|------|------|--------|
| Feature Extractor | `src/asr/feature_extractor.py` | ✅ Complete |
| CTC Decoder | `src/asr/ctc_decoder.py` | ✅ Complete |
| ONNX Runner (rewritten) | `src/asr/onnx_runner.py` | ✅ Complete |
| Test Suite | `tests/test_asr.py` | ✅ 42 tests |
| Inference Script | `scripts/test_asr_inference.py` | ✅ Complete |
| Phase Plan | `plans/phase3_plan.md` | ✅ Complete |

---

## Architecture

```
Audio (float32, 16kHz)
  │
  ▼
FeatureExtractor
  │  torchaudio.compliance.kaldi.fbank
  │  128 mel bins, 25ms window, 10ms hop, Hamming
  │  Utterance-level CMVN normalization
  │  Output: (1, T, 128) float32 + (1, T) int64 mask
  │
  ▼
ONNX Runtime (CPUExecutionProvider)
  │  Input: x (N, T, 128), mask (N, T)
  │  Output: logits (N, T/4, 512), logits_len (N)
  │  INT8 quantized, graph optimization enabled
  │  4 intra-op threads, 2 inter-op threads
  │
  ▼
CTCDecoder
  │  Greedy argmax → remove blanks → deduplicate
  │  Skip special tokens (BOS, EOS, UNK)
  │  SentencePiece: ▁ → space boundaries
  │  Confidence: geometric mean of softmax probs
  │  Per-token timestamps via frame positions
  │  Output: text, confidence, token_ids, timestamps
  │
  ▼
ASRResult
  │  segments: [TranscriptSegment(text, confidence, timestamps)]
  │  processing_time_ms, model_name, is_partial
```

---

## Model Specification (MedASR)

| Property | Value |
|----------|-------|
| Format | ONNX (INT8 quantized) |
| Size | 154 MB |
| Type | CTC (medasr_ctc) |
| Maintainer | k2-fsa/sherpa |
| Vocab | 512 SentencePiece tokens |
| Subsampling | 4x |
| Input | (N, T, 128) fbank + (N, T) mask |
| Output | (N, T/4, 512) CTC logits |

---

## Latency Benchmarks (CPU)

| Signal | Duration | Latency | RTF |
|--------|----------|---------|-----|
| Silence | 1.0s | 237ms | 0.237x |
| White noise | 1.0s | 162ms | 0.162x |
| Speech harmonics | 2.0s | 243ms | 0.122x |
| Tone bursts | 3.0s | 310ms | 0.103x |
| **Average** | - | **238ms** | **0.136x** |

**Target: <400ms → PASS** (238ms average, 40% margin)

---

## Test Summary

### New Tests (42)

**FeatureExtractor (13 tests):**
- Shape validation (1s, 3s audio)
- Float32 output dtype
- Batch + mask shapes
- Mask all-ones verification
- CMVN normalization check
- No-normalization mode
- Audio too short error
- compute_output_length accuracy
- Auto int16→float32 conversion
- Info property
- Deterministic output

**CTCDecoder (14 tests):**
- Vocab size = 512
- Blank token = <blk>
- Special tokens at expected positions
- All-blanks → empty text
- Single token repeated → decoded once
- Confidence ∈ [0.0, 1.0]
- Special tokens skipped
- Batch decode
- decode_single method
- SentencePiece space handling
- Timestamp structure
- Timestamp monotonicity
- Decode without lengths
- Softmax normalization

**OnnxASRRunner (15 tests):**
- is_loaded status
- Silence transcription
- Return type verification
- Latency < 400ms (1s audio)
- Latency < 500ms (3s audio)
- Streaming mode (partial flag)
- Segment field validation
- Model info keys
- Latency stats tracking
- full_text property
- Not-loaded error
- Missing model error
- Missing tokens error
- Load/unload lifecycle
- Default paths + custom providers

---

## Files Changed

| File | Lines | Action |
|------|-------|--------|
| `src/asr/feature_extractor.py` | 200 | New |
| `src/asr/ctc_decoder.py` | 266 | New |
| `src/asr/onnx_runner.py` | 286 | Rewritten |
| `tests/test_asr.py` | 351 | New |
| `scripts/test_asr_inference.py` | 160 | New |
| `plans/phase3_plan.md` | 147 | New |
| **Total** | **1410** | |

---

## Key Decisions

1. **torchaudio.compliance.kaldi.fbank** for feature extraction — matches the k2-fsa/sherpa training pipeline exactly (128 mel, 25ms/10ms, Hamming, dither=0).

2. **Utterance-level CMVN** (mean/variance normalization per utterance) — standard for CTC models, applied per-feature-dim with min-std clamping at 1e-5.

3. **Greedy CTC decode** (no beam search) — sufficient for real-time streaming; beam search adds latency with marginal accuracy gain for this use case.

4. **Geometric mean confidence** — uses log-space mean of softmax probabilities for decoded tokens, avoiding arithmetic mean's upward bias.

5. **Default paths from constants** — `OnnxASRRunner()` works zero-config with model at `onnx_models/medasr/`.

---

## Next Phase

Phase 4: ASR Worker Integration — connects the ONNX runner to the audio pipeline via an async worker queue (mic chunks → ASR → transcript buffer).
