"""
OR-Symphony: Sherpa-ONNX ASR Runner

ASR implementation using sherpa-onnx for MedASR CTC model.
sherpa-onnx bundles its own ONNX runtime (with optional GPU support)
so onnxruntime-gpu does NOT need to be installed separately.

The MedASR CTC int8 model from
  csukuangfj/sherpa-onnx-medasr-ctc-en-int8-2025-12-25
requires:
  - Exactly 128 feature-frames per inference call (fixed ONNX axis).
  - Batch-mode decoding (``decode_streams``) so the mask tensor gets
    the correct 2-D shape ``[batch, time]``.

At 16 kHz with a 10 ms hop (160 samples/frame) each segment is
128 × 160 = 20 480 samples ≈ 1.28 s.  Longer audio is split into
non-overlapping windows; shorter audio is zero-padded.

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
from typing import Dict, List, Optional

import numpy as np

from src.asr.runner import ASRResult, BaseASRRunner, TranscriptSegment
from src.utils.constants import (
    ASR_CHUNK_LATENCY_TARGET_MS,
    AUDIO_SAMPLE_RATE,
    MEDASR_MODEL_PATH,
    MEDASR_TOKENS_PATH,
)

logger = logging.getLogger(__name__)

# ── Model geometry ────────────────────────────────────────────────────────
_FRAME_HOP = 160           # 10 ms @ 16 kHz
_MODEL_FRAMES = 128        # fixed ONNX time-axis
_SEGMENT_SAMPLES = _MODEL_FRAMES * _FRAME_HOP  # 20 480 samples ≈ 1.28 s
_MIN_VOICE_SAMPLES = 1600  # 0.1 s — ignore clicks / glitches


class SherpaASRRunner(BaseASRRunner):
    """
    Sherpa-ONNX based ASR runner for MedASR CTC model.

    Key implementation details:
    * Audio is split into fixed 20 480-sample windows before decoding.
    * ``decode_streams`` (batch API) is used instead of ``decode_stream``
      so the ONNX ``mask`` input gets the required rank-2 shape.
    * Every decode call is wrapped in a try/except so a single bad
      segment never crashes the pipeline.
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

    # ── Model lifecycle ───────────────────────────────────────────────────

    def load_model(self) -> None:
        """Load the MedASR CTC model via sherpa_onnx.OfflineRecognizer."""
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
            feature_dim=_MODEL_FRAMES,
            debug=False,
        )

        self._loaded = True
        logger.info("Sherpa-ONNX ASR model loaded successfully")

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def unload_model(self) -> None:
        """Release model resources."""
        self._recognizer = None
        self._loaded = False
        logger.info("Sherpa-ONNX ASR model unloaded")

    # ── Transcription ─────────────────────────────────────────────────────

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> ASRResult:
        """
        Transcribe an audio buffer.

        The audio is split into fixed 20 480-sample windows (128 frames).
        Each window is decoded via the batch API and the resulting texts
        are concatenated.

        Args:
            audio: Float32 numpy array, mono, 16 kHz.
            sample_rate: Must be 16 000.

        Returns:
            ASRResult with transcript, confidence, and timing.
        """
        if not self._loaded or self._recognizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        total_start = time.perf_counter()
        audio_duration_s = len(audio) / sample_rate

        # Ensure float32, contiguous
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if not audio.flags["C_CONTIGUOUS"]:
            audio = np.ascontiguousarray(audio)

        # Skip trivially short audio (clicks, glitches)
        if len(audio) < _MIN_VOICE_SAMPLES:
            return self._empty_result(audio_duration_s)

        # Split → decode → join
        windows = self._split_to_windows(audio)
        texts = [self._decode_window(w, sample_rate) for w in windows]
        text = " ".join(t for t in texts if t).strip()

        confidence = 0.85 if text else 0.0
        total_ms = (time.perf_counter() - total_start) * 1000

        segments: List[TranscriptSegment] = []
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

        # Stats
        self._inference_count += 1
        self._total_inference_ms += total_ms
        self._total_audio_s += audio_duration_s

        if total_ms > self.latency_target_ms:
            logger.warning(
                "ASR latency %.0fms exceeds target %dms (audio=%.1fs)",
                total_ms, self.latency_target_ms, audio_duration_s,
            )

        result = ASRResult(
            segments=segments,
            processing_time_ms=total_ms,
            model_name="medasr-sherpa-onnx",
            is_partial=False,
        )
        logger.debug(
            "Sherpa ASR: '%s' (conf=%.2f, %.0fms, %d win)",
            text[:80] if text else "(empty)",
            confidence, total_ms, len(windows),
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

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _split_to_windows(audio: np.ndarray) -> List[np.ndarray]:
        """
        Split *audio* into non-overlapping windows of exactly
        ``_SEGMENT_SAMPLES`` samples, zero-padding the last one.
        """
        if len(audio) == 0:
            return []

        windows: List[np.ndarray] = []
        for offset in range(0, len(audio), _SEGMENT_SAMPLES):
            chunk = audio[offset : offset + _SEGMENT_SAMPLES]
            if len(chunk) < _SEGMENT_SAMPLES:
                padded = np.zeros(_SEGMENT_SAMPLES, dtype=np.float32)
                padded[: len(chunk)] = chunk
                chunk = padded
            windows.append(chunk)
        return windows

    def _decode_window(self, audio: np.ndarray, sample_rate: int) -> str:
        """
        Decode exactly ``_SEGMENT_SAMPLES`` samples and return text.

        Uses ``decode_streams`` (batch API) so the ONNX mask input
        gets rank 2  ``[1, time]``  instead of rank 1  ``[time]``.
        """
        try:
            stream = self._recognizer.create_stream()
            stream.accept_waveform(sample_rate, audio)
            # Batch-mode decode — fixes "Invalid rank for input: mask"
            self._recognizer.decode_streams([stream])
            return stream.result.text.strip()
        except Exception as exc:
            logger.warning("Window decode failed (%s), skipping", exc)
            return ""

    @staticmethod
    def _empty_result(audio_duration_s: float) -> ASRResult:
        """Return an empty ASRResult for trivially short audio."""
        return ASRResult(
            segments=[],
            processing_time_ms=0.0,
            model_name="medasr-sherpa-onnx",
            is_partial=False,
        )

    # ── Metadata ──────────────────────────────────────────────────────────

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
