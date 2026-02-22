"""
OR-Symphony: Sherpa-ONNX ASR Runner

ASR implementation using sherpa-onnx for MedASR CTC model.
sherpa-onnx bundles its own ONNX runtime (with GPU support) so
onnxruntime-gpu does NOT need to be installed separately.

The MedASR CTC model from
  csukuangfj/sherpa-onnx-medasr-ctc-en-int8-2025-12-25
is natively supported by sherpa-onnx's OfflineRecognizer.

Usage:
    runner = SherpaASRRunner()
    runner.load_model()
    result = runner.transcribe(audio_float32)
    print(result.full_text)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from src.asr.runner import ASRResult, BaseASRRunner, TranscriptSegment
from src.utils.constants import (
    ASR_CHUNK_LATENCY_TARGET_MS,
    AUDIO_SAMPLE_RATE,
    MEDASR_MODEL_PATH,
    MEDASR_TOKENS_PATH,
)

logger = logging.getLogger(__name__)


class SherpaASRRunner(BaseASRRunner):
    """
    Sherpa-ONNX based ASR runner for MedASR CTC model.

    Uses sherpa_onnx.OfflineRecognizer which handles feature
    extraction, ONNX inference, and CTC decoding internally.
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        tokens_path: Optional[str | Path] = None,
        latency_target_ms: int = ASR_CHUNK_LATENCY_TARGET_MS,
    ) -> None:
        self.model_path = Path(model_path or MEDASR_MODEL_PATH)
        self.tokens_path = Path(tokens_path or MEDASR_TOKENS_PATH)
        self.latency_target_ms = latency_target_ms

        self._recognizer = None
        self._loaded = False

        # Latency tracking
        self._inference_count = 0
        self._total_inference_ms = 0.0
        self._total_audio_s = 0.0

    def load_model(self) -> None:
        """Load the model via sherpa_onnx.OfflineRecognizer."""
        import sherpa_onnx

        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        if not self.tokens_path.exists():
            raise FileNotFoundError(f"Tokens file not found: {self.tokens_path}")

        logger.info("Loading Sherpa-ONNX ASR model: %s", self.model_path)

        self._recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model=str(self.model_path),
            tokens=str(self.tokens_path),
            num_threads=4,
            sample_rate=AUDIO_SAMPLE_RATE,
            feature_dim=128,
            debug=False,
        )

        self._loaded = True
        logger.info("Sherpa-ONNX ASR model loaded successfully")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> ASRResult:
        """
        Transcribe an audio chunk.

        Args:
            audio: Float32 numpy array, mono, 16kHz.
            sample_rate: Audio sample rate (must be 16000).

        Returns:
            ASRResult with transcript, confidence, and timing.
        """
        if not self._loaded or self._recognizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        total_start = time.perf_counter()
        audio_duration_s = len(audio) / sample_rate

        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Create stream, feed audio, decode
        stream = self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio)
        self._recognizer.decode_stream(stream)

        text = stream.result.text.strip()
        # sherpa-onnx doesn't provide per-token confidence for CTC greedy;
        # use a heuristic: non-empty = 0.85, empty = 0.0
        confidence = 0.85 if text else 0.0

        total_ms = (time.perf_counter() - total_start) * 1000

        segments = []
        if text:
            segments.append(
                TranscriptSegment(
                    text=text,
                    is_final=True,
                    start_time_s=0.0,
                    end_time_s=audio_duration_s,
                    confidence=confidence,
                    language="en",
                )
            )

        # Update latency tracking
        self._inference_count += 1
        self._total_inference_ms += total_ms
        self._total_audio_s += audio_duration_s

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
            model_name="medasr-sherpa-onnx",
            is_partial=False,
        )

        logger.debug(
            "Sherpa ASR: '%s' (conf=%.2f, total=%.0fms)",
            text[:80] if text else "(empty)",
            confidence,
            total_ms,
        )

        return result

    def transcribe_streaming(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> ASRResult:
        """Streaming transcription — returns partial results."""
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
        self._recognizer = None
        self._loaded = False
        logger.info("Sherpa-ONNX ASR model unloaded")

    def get_model_info(self) -> Dict[str, str]:
        """Get model metadata."""
        return {
            "model_path": str(self.model_path),
            "tokens_path": str(self.tokens_path),
            "loaded": str(self._loaded),
            "backend": "sherpa-onnx",
        }

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
