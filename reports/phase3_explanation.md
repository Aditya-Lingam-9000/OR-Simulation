# Phase 3 Explanation: Fast Streaming ASR — How It Works

## What Phase 3 Does

Phase 3 builds the **speech-to-text inference pipeline** — the core ASR engine that converts raw surgical audio into text. This is the "ears" of OR-Symphony. Phase 2 captured and chunked the audio; Phase 3 understands it.

---

## The Three-Stage Pipeline

### Stage 1: Feature Extraction (`feature_extractor.py`)

Raw audio waveforms can't be fed directly into a neural network. They need to be converted into a compact representation that captures the spectral content of speech — specifically, **log-mel filterbank (fbank) features**.

**How it works:**
1. Take raw audio (float32, 16kHz mono)
2. Apply a sliding window (25ms wide, sliding 10ms at a time)
3. For each window, compute the FFT to get frequency content
4. Apply 128 triangular mel-scale filters to group frequencies the way human hearing works
5. Take the log — this compresses the dynamic range
6. Apply CMVN (Cepstral Mean and Variance Normalization) — subtracts the mean and divides by std of each feature dimension across the utterance, so the model sees normalized input regardless of volume/microphone differences

**Output:** A matrix of shape (T, 128), where T is the number of time frames. For 1 second of audio at 10ms hop, T ≈ 98 frames.

We use `torchaudio.compliance.kaldi.fbank` because the MedASR model was trained with Kaldi-compatible features (confirmed by the k2-fsa/sherpa maintainers). Using a different feature extraction would degrade accuracy.

### Stage 2: ONNX Inference (`onnx_runner.py`)

The features go into the MedASR neural network, which runs in ONNX Runtime. This is a CTC (Connectionist Temporal Classification) model:

**Input:**
- `x`: (1, T, 128) — the fbank features, batched
- `mask`: (1, T) — all-ones mask indicating all frames are valid

**Output:**
- `logits`: (1, T/4, 512) — probability distribution over 512 tokens at each time step. The T/4 comes from 4x temporal subsampling inside the model (convolutional layers reduce time resolution for efficiency)
- `logits_len`: (1,) — how many output frames are valid

The model is INT8 quantized (154MB, reduced from ~600MB fp32) and runs on CPU with ONNX graph optimization. Session options use 4 intra-op threads for parallelism within operators and 2 inter-op threads for parallelism between operators.

### Stage 3: CTC Decoding (`ctc_decoder.py`)

CTC logits need special decoding because the model outputs one prediction per time step, and the same token may span multiple steps.

**Greedy decode algorithm:**
1. At each time step, take the token with highest logit value (argmax)
2. Remove all "blank" tokens (token 0 = `<blk>`, the CTC silence/filler token)
3. Collapse consecutive duplicates — if the model outputs `[h, h, h, e, e, l, l, l, o]`, collapse to `[h, e, l, o]`
4. Skip special tokens: BOS (`<s>`), EOS (`</s>`), UNK (`<unk>`)
5. Join remaining tokens using SentencePiece rules

**SentencePiece detokenization:**
The vocabulary uses SentencePiece encoding where `▁` (U+2581, Lower One Eighth Block) marks word boundaries:
- `▁the` → ` the`
- `s` → `s` (continuation of previous word)
- `▁doctor` → ` doctor`
- Result: `"the doctors"`

**Confidence score:** Computed as the geometric mean of softmax probabilities of all decoded (non-blank) tokens. This gives a single 0.0–1.0 value indicating how certain the model is about the entire transcription.

**Timestamps:** Each decoded token gets a start/end time computed from its frame position × frame duration (10ms hop × 4x subsampling = 40ms per CTC output frame).

---

## Why These Design Choices?

### Why CTC instead of attention-based (e.g., Whisper)?

CTC models are **auto-regressive-free** — they produce output in a single forward pass without generating tokens one at a time. This means:
- Deterministic latency (always one forward pass)
- No risk of decoder loops or hallucination
- Natural for streaming (can decode partial outputs)
- Lower computational cost per token

The trade-off is slightly lower accuracy than attention models on complex sentences, but for surgical commands (short, structured phrases like "activate the ventilator" or "clamp the hepatic artery"), CTC is more than sufficient.

### Why greedy decode instead of beam search?

Beam search explores multiple hypotheses and can improve word error rate by 1-3%. But it adds 50-100ms of latency. Since we're targeting <400ms total and already at 238ms average, we have margin, but greedy keeps us well within budget.

### Why CMVN normalization?

Different microphones, rooms, and speaker volumes produce very different raw feature magnitudes. CMVN normalizes each utterance to zero mean and unit variance, making the model robust to these variations. This is critical in an OR where the microphone type and placement may vary.

### Why 128 mel bins (not 80)?

The MedASR model was trained with 128 mel bins (verified by inspecting the model's input shape). Using 80 (which is more common in open-source ASR) would produce features the model wasn't trained on, causing garbage output.

---

## Latency Breakdown

For a typical 2-second audio chunk:

| Stage | Time | % |
|-------|------|---|
| Feature extraction | ~20ms | 8% |
| ONNX inference | ~210ms | 86% |
| CTC decode | ~2ms | 1% |
| Overhead | ~11ms | 5% |
| **Total** | **~243ms** | 100% |

ONNX inference dominates. The INT8 quantization already reduces this by ~2x versus fp32. Further optimization options (for future phases) include:
- ONNX execution provider tuning
- TensorRT if GPU is available
- Reducing model size via knowledge distillation

---

## How It Connects to the Pipeline

```
Phase 2 (Audio)                    Phase 3 (ASR)
                                   
MicStream → VAD → ChunkBuilder → FeatureExtractor → ONNX → CTCDecoder
                                                                  │
                                                            ASRResult
                                                            (text, conf,
                                                             timestamps)
                                                                  │
                                                                  ▼
                                                   Phase 4: ASR Worker Queue
                                                   Phase 5: Rolling Buffer
                                                   Phase 6: Rule Engine / LLM
```

Phase 4 will create the async worker that connects Phase 2's chunk output to Phase 3's inference, running transcription in a background thread so the audio capture never blocks.
