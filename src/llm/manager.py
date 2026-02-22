"""
OR-Symphony: LLM Manager — MedGemma Request Queue & Dispatcher

Manages the full LLM inference pipeline:
  1. Builds prompt via PromptBuilder (surgery-aware)
  2. Submits to LLMBatcher → GGUFRunner for inference
  3. Normalizes output via StateSerializer
  4. Falls back to rule-only mode on failure

Usage:
    from src.llm.manager import LLMManager
    manager = LLMManager(surgery="PCNL")
    await manager.start()
    response = await manager.submit(request)
    await manager.stop()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.llm.batcher import LLMBatcher
from src.llm.gguf_runner import GGUFRunner
from src.llm.prompts import PromptBuilder
from src.state.serializer import StateSerializer
from src.utils.constants import (
    LLM_MAX_BATCH_SIZE,
    LLM_MAX_WAIT_MS,
    MEDGEMMA_MODEL_PATH,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class LLMRequest:
    """A single request to the LLM."""

    request_id: str
    transcript_context: str
    surgery_type: str
    machines_dict: Dict[str, Any]
    current_phase: str = "Phase1"
    session_time_s: float = 0.0
    current_machines: Optional[Dict[str, List[str]]] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class LLMResponse:
    """Response from the LLM."""

    request_id: str
    output: Dict[str, Any]
    processing_time_ms: float = 0.0
    model_name: str = "medgemma-4b-Q3"
    success: bool = True
    error: Optional[str] = None
    degraded: bool = False


# ---------------------------------------------------------------------------
# Fallback response template
# ---------------------------------------------------------------------------

_FALLBACK_OUTPUT: Dict[str, Any] = {
    "metadata": {
        "reasoning": "LLM degraded — falling back to rule-only mode",
        "phase": "",
        "next_phase": "",
    },
    "machines": {"0": [], "1": []},
    "details": {},
    "suggestions": ["LLM inference failed — using rule engine only"],
    "confidence": 0.0,
    "source": "rule",
}


class LLMManager:
    """
    Manages the full LLM inference lifecycle.

    Pipeline:
      LLMRequest → PromptBuilder → LLMBatcher → GGUFRunner → StateSerializer → LLMResponse

    Features:
      - Surgery-aware prompt building with machine dictionary context
      - Micro-batched inference through GGUFRunner
      - Schema validation of LLM output
      - Automatic fallback to rule-only mode on any failure
      - Stub mode for testing (no model file required)
      - Stats tracking (requests, failures, latencies)

    Args:
        surgery: Surgery type (PCNL, Partial Hepatectomy, Lobectomy).
        model_path: Path to GGUF model file.
        max_batch_size: Maximum batch size for micro-batcher.
        max_wait_ms: Maximum wait time for micro-batcher.
        stub_mode: If True, GGUFRunner returns stubs (no model loading).
        validate_output: If True, validate LLM output against JSON schema.
        runner: Optional pre-configured GGUFRunner (for testing).
        batcher: Optional pre-configured LLMBatcher (for testing).
    """

    def __init__(
        self,
        surgery: str = "PCNL",
        model_path: str = str(MEDGEMMA_MODEL_PATH),
        max_batch_size: int = LLM_MAX_BATCH_SIZE,
        max_wait_ms: int = LLM_MAX_WAIT_MS,
        stub_mode: bool = False,
        validate_output: bool = True,
        runner: Optional[GGUFRunner] = None,
        batcher: Optional[LLMBatcher] = None,
    ) -> None:
        self.surgery = surgery
        self.model_path = model_path
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.stub_mode = stub_mode
        self.validate_output = validate_output

        # Core components
        self._runner = runner or GGUFRunner(
            model_path=model_path, stub_mode=stub_mode
        )
        self._batcher = batcher or LLMBatcher(
            runner=self._runner,
            max_batch=max_batch_size,
            max_wait_ms=max_wait_ms,
        )
        self._prompt_builder = PromptBuilder(surgery=surgery)
        self._serializer = StateSerializer()

        # State
        self._running = False
        self._model_loaded = False

        # Stats
        self._total_requests = 0
        self._total_failures = 0
        self._total_degraded = 0
        self._total_time_ms = 0.0

        logger.info(
            "LLMManager initialized — surgery=%s, model=%s, stub=%s, batch=%d",
            surgery,
            model_path,
            stub_mode,
            max_batch_size,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the LLM manager: load model and start batcher."""
        if self._running:
            logger.warning("LLMManager already running")
            return

        # Load model (sync — runs in the caller's context)
        try:
            loaded = self._runner.load_model()
            self._model_loaded = bool(loaded) and self._runner.is_loaded()
            if self._model_loaded:
                logger.info("LLMManager: model loaded successfully")
            else:
                logger.warning("LLMManager: model load returned False — degraded mode")
        except Exception as e:
            logger.error("LLMManager: model load failed: %s", e)
            self._model_loaded = False
            # Continue in degraded mode — all submits will fallback

        # Start batcher
        await self._batcher.start()
        self._running = True
        logger.info("LLMManager started (model_loaded=%s)", self._model_loaded)

    async def stop(self) -> None:
        """Stop the LLM manager: stop batcher and unload model."""
        self._running = False

        # Stop batcher
        await self._batcher.stop()

        # Unload model
        try:
            self._runner.unload()
        except Exception as e:
            logger.error("LLMManager: model unload error: %s", e)

        self._model_loaded = False
        logger.info("LLMManager stopped")

    # ------------------------------------------------------------------
    # Surgery switching
    # ------------------------------------------------------------------

    def set_surgery(self, surgery: str) -> None:
        """
        Switch the surgery type (rebuilds prompt builder).

        Args:
            surgery: New surgery type.
        """
        self.surgery = surgery
        self._prompt_builder = PromptBuilder(surgery=surgery)
        logger.info("LLMManager: surgery switched to %s", surgery)

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    async def submit(self, request: LLMRequest) -> LLMResponse:
        """
        Submit a request for LLM inference.

        Pipeline:
          1. Build prompt from transcript context + surgery config
          2. Submit to batcher for queued inference
          3. Normalize and validate the output
          4. On any failure → return degraded fallback

        Args:
            request: LLM request with transcript context.

        Returns:
            LLMResponse with structured JSON output.
        """
        self._total_requests += 1
        t0 = time.perf_counter()

        try:
            # ---- 1. Build prompt ----
            # Convert seconds to human-readable string for the prompt
            mins, secs = divmod(int(request.session_time_s), 60)
            session_time_str = f"{mins}m {secs}s"

            prompt = self._prompt_builder.build_completion_prompt(
                transcript_context=request.transcript_context,
                current_phase=request.current_phase,
                session_time=session_time_str,
                current_machines=request.current_machines,
            )

            # ---- 2. Submit to batcher ----
            if not self._model_loaded and not self.stub_mode:
                # Model not loaded — immediate fallback
                raise RuntimeError("Model not loaded — degraded mode")

            batch_result = await self._batcher.submit(
                prompt=prompt,
                request_id=request.request_id,
                use_chat=False,
            )

            if not batch_result.success:
                raise RuntimeError(batch_result.error or "Batcher processing failed")

            raw_output = batch_result.output

            # ---- 3. Normalize output ----
            normalized = self._serializer.normalize_llm_output(
                raw=raw_output,
                surgery=request.surgery_type,
                phase=request.current_phase,
            )

            # ---- 4. Validate schema (optional) ----
            if self.validate_output:
                try:
                    self._serializer.validate_schema(normalized)
                except Exception as ve:
                    logger.warning(
                        "LLM output failed schema validation: %s — using raw",
                        ve,
                    )
                    # Still use the normalized output, just log the warning

            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._total_time_ms += elapsed_ms

            response = LLMResponse(
                request_id=request.request_id,
                output=normalized,
                processing_time_ms=elapsed_ms,
                model_name="medgemma-4b-Q3" if not self.stub_mode else "medgemma-stub",
                success=True,
            )

            logger.debug(
                "LLM submit [%s]: %.1fms, confidence=%.2f",
                request.request_id,
                elapsed_ms,
                normalized.get("confidence", 0),
            )
            return response

        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._total_failures += 1
            self._total_degraded += 1
            self._total_time_ms += elapsed_ms

            logger.warning(
                "LLM submit [%s] failed (%.1fms): %s — falling back",
                request.request_id,
                elapsed_ms,
                e,
            )

            fallback = dict(_FALLBACK_OUTPUT)
            fallback["metadata"] = dict(fallback["metadata"])
            fallback["metadata"]["surgery"] = request.surgery_type
            fallback["metadata"]["phase"] = request.current_phase
            fallback["metadata"]["reasoning"] = f"LLM degraded: {e}"

            return LLMResponse(
                request_id=request.request_id,
                output=fallback,
                processing_time_ms=elapsed_ms,
                model_name="fallback",
                success=False,
                error=str(e),
                degraded=True,
            )

    # ------------------------------------------------------------------
    # Properties & Stats
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """Check if manager is running."""
        return self._running

    @property
    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded

    @property
    def queue_size(self) -> int:
        """Current batcher queue size."""
        return self._batcher.queue_size

    @property
    def total_requests(self) -> int:
        """Total requests submitted."""
        return self._total_requests

    @property
    def stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        avg_ms = (
            self._total_time_ms / self._total_requests
            if self._total_requests > 0
            else 0.0
        )
        return {
            "total_requests": self._total_requests,
            "total_failures": self._total_failures,
            "total_degraded": self._total_degraded,
            "avg_latency_ms": round(avg_ms, 1),
            "is_running": self._running,
            "model_loaded": self._model_loaded,
            "stub_mode": self.stub_mode,
            "surgery": self.surgery,
            "batcher": self._batcher.stats,
            "runner": self._runner.stats,
        }
