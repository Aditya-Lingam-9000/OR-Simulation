# Phase 3 Plan — Fast Streaming ASR (MedASR via ONNX)

**Date:** 2026-02-22  
**Depends on:** Phase 2 PASS ✅  
**Goal:** Real-time transcripts from MedASR INT8 ONNX model with CTC decoding.

---

## Model Analysis (completed during investigation)

- **Model**: `onnx_models/medasr/model.int8.onnx` (147 MB, INT8 quantized)
- **Type**: CTC (Connectionist Temporal Classification), k2-fsa/sherpa format
- **Input**: `x` (N, T, 128) = 128-dim log-mel fbank features; `mask` (N, T) = length mask
- **Output**: `logits` (N, T/4, 512) = CTC logits; `logits_len` (N) = valid lengths
- **Vocab**: 512 SentencePiece tokens, token 0 = `<blk>` (CTC blank)
- **Subsampling**: Factor 4 (T frames → T/4 output frames)
- **Feature extraction**: 128-dim fbank, 25ms window, 10ms hop, Hamming window
- **Latency**: ~300ms for 3s audio on CPU → RTF ≈ 0.1x (well under 400ms target)

## Deliverables

1. **`src/asr/feature_extractor.py`** — Audio → 128-dim fbank features
   - Uses torchaudio.compliance.kaldi.fbank
   - Global CMVN normalization
   - Handles variable-length audio

2. **`src/asr/ctc_decoder.py`** — CTC logits → text
   - Greedy decode (remove blanks + deduplicate)
   - SentencePiece detokenization (▁ → space)
   - Confidence estimation from softmax probabilities

3. **`src/asr/onnx_runner.py`** — Update with real inference
   - Wire feature_extractor → ONNX session → CTC decoder
   - Latency tracking + logging

4. **`scripts/test_asr_inference.py`** — Test ASR on WAV files
   - Transcribe test audio files
   - Report latency and output

5. **`tests/test_asr.py`** — ASR unit tests
   - Feature extractor shape/dtype
   - CTC decoder correctness
   - End-to-end inference pipeline
   - Latency bounds

## Architecture

```
AudioChunk.audio (float32, 16kHz)
    ↓
FeatureExtractor.extract(audio)
    ↓ (N, T, 128) fbank features
OnnxASRRunner._session.run(x, mask)
    ↓ (N, T/4, 512) CTC logits
CTCDecoder.decode(logits, lengths)
    ↓ text + confidence + timestamps
ASRResult
```
