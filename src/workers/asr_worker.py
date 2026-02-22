"""
OR-Symphony: ASR Worker

Reads audio chunks from an async queue, runs ASR inference via OnnxASRRunner,
and enqueues final transcripts for the rule engine and LLM pipeline.

Designed for single-writer, single-reader pattern with asyncio queues.

Usage:
    worker = ASRWorker(audio_queue, transcript_queue)
    await worker.start()  # runs until stop() is called
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional

import numpy as np

from src.asr.runner import ASRResult
from src.utils.constants import ASR_MAX_QUEUE_SIZE, AUDIO_SAMPLE_RATE

logger = logging.getLogger(__name__)


def _create_asr_runner():
    """Create the best available ASR runner (sherpa-onnx preferred)."""
    # Try sherpa-onnx first (bundles its own ONNX runtime, no CUDA lib issues)
    try:
        import sherpa_onnx  # noqa: F401
        from src.asr.sherpa_runner import SherpaASRRunner
        logger.info("ASR backend: sherpa-onnx (preferred)")
        return SherpaASRRunner()
    except ImportError:
        logger.info("sherpa-onnx not available, trying onnxruntime backend")

    # Fall back to raw onnxruntime
    from src.asr.onnx_runner import OnnxASRRunner
    logger.info("ASR backend: onnxruntime")
    return OnnxASRRunner()


class ASRWorker:
    """
    ASR processing worker.

    Consumes audio chunks from an ingest queue,
    runs ONNX ASR inference, and produces transcripts
    on an output queue.
    """

    def __init__(
        self,
        audio_queue: Optional[asyncio.Queue] = None,
        transcript_queue: Optional[asyncio.Queue] = None,
        runner: Optional[OnnxASRRunner] = None,
    ) -> None:
        """
        Initialize the ASR worker.

        Args:
            audio_queue: Input queue of audio chunks (np.ndarray float32).
            transcript_queue: Output queue for ASRResult objects.
            runner: Pre-configured ASR runner. Created if None.
        """
        self.audio_queue = audio_queue or asyncio.Queue(maxsize=ASR_MAX_QUEUE_SIZE)
        self.transcript_queue = transcript_queue or asyncio.Queue(maxsize=ASR_MAX_QUEUE_SIZE)
        self._runner = runner
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._processed_count = 0
        self._error_count = 0
        self._total_audio_s = 0.0
        self._total_inference_ms = 0.0
        logger.info("ASRWorker initialized")

    async def start(self) -> None:
        """Start the ASR worker processing loop."""
        if self._running:
            logger.warning("ASRWorker already running")
            return

        # Load model if not already loaded
        if self._runner is None:
            self._runner = _create_asr_runner()
        if not self._runner.is_loaded():
            logger.info("ASRWorker loading ASR model...")
            await asyncio.get_event_loop().run_in_executor(
                None, self._runner.load_model
            )
            logger.info("ASRWorker model loaded")

        self._running = True
        self._task = asyncio.create_task(self._process_loop())
        logger.info("ASRWorker started")

    async def _process_loop(self) -> None:
        """Main processing loop — consume audio, produce transcripts."""
        while self._running:
            try:
                # Wait for an audio chunk with timeout for clean shutdown
                try:
                    chunk = await asyncio.wait_for(
                        self.audio_queue.get(), timeout=0.5
                    )
                except asyncio.TimeoutError:
                    continue

                if chunk is None:
                    # Poison pill — stop processing
                    logger.info("ASRWorker received stop signal")
                    break

                # Run inference in executor to avoid blocking event loop
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self._transcribe_chunk, chunk
                )

                if result is not None:
                    await self.transcript_queue.put(result)
                    self._processed_count += 1

            except Exception:
                self._error_count += 1
                logger.exception("ASRWorker error processing chunk")

    def _transcribe_chunk(self, chunk: Any) -> Optional[ASRResult]:
        """
        Transcribe a single audio chunk (runs in thread pool).

        Args:
            chunk: Either a numpy array or an object with .audio attribute.

        Returns:
            ASRResult or None if chunk is invalid.
        """
        try:
            # Extract audio array from chunk
            if isinstance(chunk, np.ndarray):
                audio = chunk
            elif hasattr(chunk, "audio"):
                audio = chunk.audio
            else:
                logger.warning("Unknown chunk type: %s", type(chunk))
                return None

            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            audio_duration_s = len(audio) / AUDIO_SAMPLE_RATE
            self._total_audio_s += audio_duration_s

            result = self._runner.transcribe(audio)
            self._total_inference_ms += result.processing_time_ms

            if result.full_text:
                logger.debug(
                    "ASR: '%s' (%.0fms, conf=%.2f)",
                    result.full_text[:60],
                    result.processing_time_ms,
                    result.segments[0].confidence if result.segments else 0.0,
                )

            return result

        except Exception:
            logger.exception("Transcription failed")
            self._error_count += 1
            return None

    async def stop(self) -> None:
        """Stop the ASR worker gracefully."""
        self._running = False
        if self._task is not None:
            # Send poison pill
            await self.audio_queue.put(None)
            try:
                await asyncio.wait_for(self._task, timeout=5.0)
            except asyncio.TimeoutError:
                self._task.cancel()
            self._task = None
        logger.info(
            "ASRWorker stopped — processed=%d, errors=%d, audio=%.1fs",
            self._processed_count,
            self._error_count,
            self._total_audio_s,
        )

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def stats(self) -> Dict[str, Any]:
        avg_ms = (
            self._total_inference_ms / self._processed_count
            if self._processed_count > 0
            else 0.0
        )
        return {
            "running": self._running,
            "processed": self._processed_count,
            "errors": self._error_count,
            "total_audio_s": round(self._total_audio_s, 1),
            "total_inference_ms": round(self._total_inference_ms, 1),
            "avg_inference_ms": round(avg_ms, 1),
            "audio_queue_size": self.audio_queue.qsize(),
            "transcript_queue_size": self.transcript_queue.qsize(),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    worker = ASRWorker()
    print(f"ASRWorker stats: {worker.stats}")
