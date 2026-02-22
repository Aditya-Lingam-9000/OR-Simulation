"""
OR-Symphony: Faster-Whisper ASR Runner

ASR implementation using faster-whisper (CTranslate2 backend).
Works reliably on Kaggle T4 GPUs — no ONNX shape issues.

faster-whisper handles:
  - Any audio length (no fixed-frame constraints)
  - Automatic feature extraction
  - GPU acceleration via CTranslate2

Usage:
    runner = WhisperASRRunner(model_size="base.en")
    runner.load_model()
    result = runner.transcribe(audio_float32)
    print(result.full_text)
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import numpy as np

from src.asr.runner import ASRResult, BaseASRRunner, TranscriptSegment
from src.utils.constants import ASR_CHUNK_LATENCY_TARGET_MS, AUDIO_SAMPLE_RATE

logger = logging.getLogger(__name__)

# Minimum audio length to bother transcribing (0.3s)
_MIN_AUDIO_SAMPLES = int(AUDIO_SAMPLE_RATE * 0.3)


class WhisperASRRunner(BaseASRRunner):
    """
    Faster-whisper based ASR runner.

    Uses CTranslate2 for fast GPU inference. Handles any audio
    length without fixed-frame or mask-shape constraints.
    """

    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "auto",
        compute_type: str = "float16",
        latency_target_ms: int = ASR_CHUNK_LATENCY_TARGET_MS,
    ) -> None:
        """
        Args:
            model_size: Whisper model size (tiny.en, base.en, small.en, etc.)
            device: "cuda", "cpu", or "auto"
            compute_type: "float16" (GPU), "int8" (CPU), or "auto"
            latency_target_ms: Latency warning threshold.
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.latency_target_ms = latency_target_ms

        self._model = None
        self._loaded = False

        # Latency tracking
        self._inference_count = 0
        self._total_inference_ms = 0.0
        self._total_audio_s = 0.0

    def load_model(self) -> None:
        """Load the faster-whisper model."""
        from faster_whisper import WhisperModel

        # Auto-detect device
        device = self.device
        compute_type = self.compute_type
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        if device == "cpu" and compute_type == "float16":
            compute_type = "int8"

        logger.info(
            "Loading faster-whisper model: %s (device=%s, compute=%s)",
            self.model_size, device, compute_type,
        )

        self._model = WhisperModel(
            self.model_size,
            device=device,
            compute_type=compute_type,
        )

        self._loaded = True
        logger.info("Faster-whisper model loaded successfully")

    def is_loaded(self) -> bool:
        return self._loaded

    def unload_model(self) -> None:
        self._model = None
        self._loaded = False
        logger.info("Faster-whisper model unloaded")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> ASRResult:
        """
        Transcribe an audio buffer using faster-whisper.

        Args:
            audio: Float32 numpy array, mono, 16 kHz.
            sample_rate: Must be 16000.

        Returns:
            ASRResult with transcript and timing info.
        """
        if not self._loaded or self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        total_start = time.perf_counter()
        audio_duration_s = len(audio) / sample_rate

        # Ensure float32, contiguous
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if not audio.flags["C_CONTIGUOUS"]:
            audio = np.ascontiguousarray(audio)

        # Skip trivially short audio
        if len(audio) < _MIN_AUDIO_SAMPLES:
            return self._empty_result()

        # Run transcription
        try:
            segments_gen, info = self._model.transcribe(
                audio,
                language="en",
                beam_size=1,          # Greedy for speed
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=True,      # Skip silence segments
                vad_parameters=dict(
                    min_silence_duration_ms=300,
                    speech_pad_ms=200,
                ),
            )

            # Collect all segments
            texts: List[str] = []
            transcript_segments: List[TranscriptSegment] = []

            for seg in segments_gen:
                seg_text = seg.text.strip()
                if seg_text:
                    texts.append(seg_text)
                    transcript_segments.append(
                        TranscriptSegment(
                            text=seg_text,
                            is_final=True,
                            start_time_s=seg.start,
                            end_time_s=seg.end,
                            confidence=max(0.0, min(1.0, 1.0 - seg.no_speech_prob)),
                            language=info.language or "en",
                        )
                    )

        except Exception as exc:
            logger.warning("Whisper transcription failed: %s", exc)
            return self._empty_result()

        text = " ".join(texts).strip()
        total_ms = (time.perf_counter() - total_start) * 1000

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
            segments=transcript_segments,
            processing_time_ms=total_ms,
            model_name=f"whisper-{self.model_size}",
            is_partial=False,
        )

        logger.debug(
            "Whisper ASR: '%s' (%.0fms, %d segs)",
            text[:80] if text else "(empty)",
            total_ms, len(transcript_segments),
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

    @staticmethod
    def _empty_result() -> ASRResult:
        return ASRResult(
            segments=[],
            processing_time_ms=0.0,
            model_name="whisper",
            is_partial=False,
        )

    def get_model_info(self) -> Dict[str, str]:
        return {
            "model_size": self.model_size,
            "device": self.device,
            "compute_type": self.compute_type,
            "loaded": str(self._loaded),
            "backend": "faster-whisper",
        }

    @property
    def latency_stats(self) -> Dict[str, float]:
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
