"""
OR-Symphony: ONNX ASR Runner

Concrete ASR implementation using ONNX Runtime for MedASR CTC model.
Full pipeline: audio → fbank features → ONNX inference → CTC decode → text.

Model specification:
  - Input:  x (N, T, 128) fbank features, mask (N, T) int64
  - Output: logits (N, T/4, 512) CTC logits, logits_len (N) int64
  - Vocab:  512 SentencePiece tokens, token 0 = CTC blank
  - Quant:  INT8

Usage:
    runner = OnnxASRRunner()
    runner.load_model()
    result = runner.transcribe(audio_float32)
    print(result.full_text)

Sanity check:
    python -m src.asr.onnx_runner
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.asr.ctc_decoder import CTCDecoder
from src.asr.feature_extractor import FeatureExtractor
from src.asr.runner import ASRResult, BaseASRRunner, TranscriptSegment
from src.utils.constants import (
    ASR_CHUNK_LATENCY_TARGET_MS,
    AUDIO_SAMPLE_RATE,
    MEDASR_MODEL_PATH,
    MEDASR_TOKENS_PATH,
)
from src.utils.device import get_onnx_providers

logger = logging.getLogger(__name__)


class OnnxASRRunner(BaseASRRunner):
    """
    ONNX Runtime-based ASR runner for MedASR CTC model.

    Full pipeline:
        audio (float32, 16kHz)
          → FeatureExtractor → 128-dim fbank features
          → ONNX Session → CTC logits
          → CTCDecoder → text + confidence + timestamps
          → ASRResult
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        tokens_path: Optional[str | Path] = None,
        providers: Optional[List[str]] = None,
        latency_target_ms: int = ASR_CHUNK_LATENCY_TARGET_MS,
    ) -> None:
        """
        Initialize the ONNX ASR runner.

        Args:
            model_path: Path to ONNX model. Defaults to MEDASR_MODEL_PATH.
            tokens_path: Path to tokens.txt. Defaults to MEDASR_TOKENS_PATH.
            providers: ONNX providers. Auto-detected if None.
            latency_target_ms: Warn if inference exceeds this latency.
        """
        self.model_path = Path(model_path or MEDASR_MODEL_PATH)
        self.tokens_path = Path(tokens_path or MEDASR_TOKENS_PATH)
        self.providers = providers or get_onnx_providers()
        self.latency_target_ms = latency_target_ms

        self._session = None
        self._feature_extractor: Optional[FeatureExtractor] = None
        self._decoder: Optional[CTCDecoder] = None
        self._loaded = False

        # Latency tracking
        self._inference_count = 0
        self._total_inference_ms = 0.0
        self._total_audio_s = 0.0

    def load_model(self) -> None:
        """Load the ONNX model, feature extractor, and CTC decoder."""
        import onnxruntime as ort

        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        if not self.tokens_path.exists():
            raise FileNotFoundError(f"Tokens file not found: {self.tokens_path}")

        logger.info("Loading ONNX ASR model: %s", self.model_path)
        logger.info("Using providers: %s", self.providers)

        # Session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 2

        # Create inference session
        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=self.providers,
        )

        # Read model metadata
        meta = self._session.get_modelmeta()
        metadata = meta.custom_metadata_map
        subsampling = int(metadata.get("subsampling_factor", "4"))

        # Initialize feature extractor
        self._feature_extractor = FeatureExtractor(
            sample_rate=AUDIO_SAMPLE_RATE,
            num_mel_bins=128,
        )

        # Initialize CTC decoder
        self._decoder = CTCDecoder(
            tokens_path=self.tokens_path,
            subsampling_factor=subsampling,
        )

        self._loaded = True

        # Log model info
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        logger.info(
            "MedASR loaded — inputs: %s, outputs: %s, vocab: %d, subsample: %dx",
            [(i.name, i.shape) for i in inputs],
            [(o.name, o.shape) for o in outputs],
            self._decoder.vocab_size,
            subsampling,
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> ASRResult:
        """
        Transcribe an audio chunk.

        Args:
            audio: Float32 numpy array, mono, 16kHz. Range [-1.0, 1.0].
            sample_rate: Audio sample rate (must be 16000).

        Returns:
            ASRResult with transcript, confidence, and timing.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        total_start = time.perf_counter()
        audio_duration_s = len(audio) / sample_rate

        # --- Step 1: Feature extraction ---
        feat_start = time.perf_counter()
        features, mask = self._feature_extractor.extract_with_mask(audio)
        feat_ms = (time.perf_counter() - feat_start) * 1000

        # --- Step 2: ONNX inference ---
        infer_start = time.perf_counter()
        logits, logits_len = self._session.run(
            None,
            {"x": features, "mask": mask},
        )
        infer_ms = (time.perf_counter() - infer_start) * 1000

        # --- Step 3: CTC decode ---
        decode_start = time.perf_counter()
        decoded = self._decoder.decode(logits, logits_len)
        text, confidence, token_ids = decoded[0]  # single-batch
        decode_ms = (time.perf_counter() - decode_start) * 1000

        # --- Step 4: Compute timestamps ---
        timestamps = self._decoder.compute_timestamps(
            logits[0], int(logits_len[0])
        )

        total_ms = (time.perf_counter() - total_start) * 1000

        # Build segments with timestamps
        segments = []
        if text:
            # Use first and last token timestamps for segment bounds
            start_s = timestamps[0][1] if timestamps else 0.0
            end_s = timestamps[-1][2] if timestamps else audio_duration_s
            segments.append(
                TranscriptSegment(
                    text=text,
                    is_final=True,
                    start_time_s=start_s,
                    end_time_s=end_s,
                    confidence=confidence,
                    language="en",
                    tokens=[self._decoder._id_to_token.get(t, "") for t in token_ids],
                )
            )

        # Update latency tracking
        self._inference_count += 1
        self._total_inference_ms += total_ms
        self._total_audio_s += audio_duration_s

        # Warn if latency target exceeded
        if total_ms > self.latency_target_ms:
            logger.warning(
                "ASR latency %.0fms exceeds target %dms (audio=%.1fs)",
                total_ms,
                self.latency_target_ms,
                audio_duration_s,
            )

        result = ASRResult(
            segments=segments,
            processing_time_ms=total_ms,
            model_name="medasr-int8-onnx",
            is_partial=False,
        )

        logger.debug(
            "ASR: '%s' (conf=%.2f, feat=%.0fms, infer=%.0fms, decode=%.0fms, total=%.0fms)",
            text[:80] if text else "(empty)",
            confidence,
            feat_ms,
            infer_ms,
            decode_ms,
            total_ms,
        )

        return result

    def transcribe_streaming(self, audio: np.ndarray, sample_rate: int = 16000) -> ASRResult:
        """
        Streaming transcription — returns partial results.

        For CTC models, streaming means we mark the result as partial
        (the full utterance may not be complete yet).
        """
        result = self.transcribe(audio, sample_rate)
        result.is_partial = True
        for seg in result.segments:
            seg.is_final = False
        return result

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def unload_model(self) -> None:
        """Release model resources."""
        self._session = None
        self._feature_extractor = None
        self._decoder = None
        self._loaded = False
        logger.info("ONNX ASR model unloaded")

    def get_model_info(self) -> Dict[str, str]:
        """Get model metadata."""
        info = {
            "model_path": str(self.model_path),
            "tokens_path": str(self.tokens_path),
            "providers": str(self.providers),
            "loaded": str(self._loaded),
        }
        if self._decoder is not None:
            info["vocab_size"] = str(self._decoder.vocab_size)
        if self._session is not None:
            inputs = self._session.get_inputs()
            outputs = self._session.get_outputs()
            info["inputs"] = str([(i.name, i.shape, i.type) for i in inputs])
            info["outputs"] = str([(o.name, o.shape, o.type) for o in outputs])
            meta = self._session.get_modelmeta()
            info["model_type"] = meta.custom_metadata_map.get("model_type", "unknown")
            info["subsampling_factor"] = meta.custom_metadata_map.get("subsampling_factor", "?")
        return info

    @property
    def latency_stats(self) -> Dict[str, float]:
        """Get latency statistics."""
        avg_ms = (
            self._total_inference_ms / self._inference_count
            if self._inference_count > 0
            else 0.0
        )
        avg_rtf = (
            (self._total_inference_ms / 1000.0) / self._total_audio_s
            if self._total_audio_s > 0
            else 0.0
        )
        return {
            "inference_count": self._inference_count,
            "total_ms": round(self._total_inference_ms, 1),
            "avg_ms": round(avg_ms, 1),
            "total_audio_s": round(self._total_audio_s, 1),
            "avg_rtf": round(avg_rtf, 3),
        }


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("OR-Symphony: MedASR ONNX Runner — Sanity Check")
    print("=" * 60)

    runner = OnnxASRRunner()

    if not runner.model_path.exists():
        print(f"Model not found at {runner.model_path}")
        sys.exit(1)

    runner.load_model()
    info = runner.get_model_info()
    for k, v in info.items():
        print(f"  {k}: {v}")

    # Test with silence (should produce empty/minimal output)
    print("\n--- Test 1: 1s silence ---")
    silence = np.zeros(16000, dtype=np.float32)
    result = runner.transcribe(silence)
    print(f"  Text: '{result.full_text}'")
    print(f"  Time: {result.processing_time_ms:.1f}ms")
    print(f"  Segments: {len(result.segments)}")

    # Test with synthetic speech-like audio
    print("\n--- Test 2: 2s speech-like signal ---")
    t = np.linspace(0, 2.0, 32000, dtype=np.float32)
    speech_like = np.zeros_like(t)
    for freq, amp in [(150, 0.3), (300, 0.2), (450, 0.1)]:
        speech_like += amp * np.sin(2 * np.pi * freq * t)
    speech_like += 0.05 * np.random.randn(len(t)).astype(np.float32)
    speech_like = speech_like / np.max(np.abs(speech_like)) * 0.7

    result = runner.transcribe(speech_like)
    print(f"  Text: '{result.full_text}'")
    print(f"  Time: {result.processing_time_ms:.1f}ms")
    if result.segments:
        print(f"  Confidence: {result.segments[0].confidence:.3f}")

    # Latency stats
    print(f"\n--- Latency Stats ---")
    stats = runner.latency_stats
    for k, v in stats.items():
        print(f"  {k}: {v}")

    runner.unload_model()
    print("\nDone.")
