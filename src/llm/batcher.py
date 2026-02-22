"""
OR-Symphony: LLM Micro-Batcher

Collects inference requests into batches and processes them sequentially.
GGUF models are single-threaded, so batching here means queuing requests
and processing them one-at-a-time with efficient scheduling.

Features:
  - Configurable batch size and max wait time
  - Async queue with timeout-based flushing
  - Stats tracking (processed, failed, avg latency)
  - Graceful shutdown

Usage:
    from src.llm.batcher import LLMBatcher
    batcher = LLMBatcher(runner=runner, max_batch=4, max_wait_ms=500)
    await batcher.start()
    result = await batcher.submit(prompt)
    await batcher.stop()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from src.utils.constants import LLM_MAX_BATCH_SIZE, LLM_MAX_WAIT_MS

logger = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    """A single request in the batch queue."""

    request_id: str
    prompt: str
    future: asyncio.Future
    submitted_at: float = field(default_factory=time.time)
    use_chat: bool = False
    messages: Optional[List[Dict[str, str]]] = None


@dataclass
class BatchResult:
    """Result from processing a batch request."""

    request_id: str
    output: Dict[str, Any]
    processing_time_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None


class LLMBatcher:
    """
    Micro-batcher for LLM inference requests.

    Collects requests up to max_batch size or max_wait_ms timeout,
    then processes them sequentially through the GGUFRunner.

    Args:
        runner: GGUFRunner instance (or any object with generate/chat methods).
        max_batch: Maximum requests to collect before processing.
        max_wait_ms: Maximum wait time before processing a partial batch.
        process_fn: Optional custom processing function.
    """

    def __init__(
        self,
        runner: Any = None,
        max_batch: int = LLM_MAX_BATCH_SIZE,
        max_wait_ms: int = LLM_MAX_WAIT_MS,
        process_fn: Optional[Callable] = None,
    ) -> None:
        self.runner = runner
        self.max_batch = max_batch
        self.max_wait_ms = max_wait_ms
        self._process_fn = process_fn

        self._queue: asyncio.Queue[BatchRequest] = asyncio.Queue()
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None

        # Stats
        self._total_processed = 0
        self._total_failed = 0
        self._total_batches = 0
        self._total_wait_ms = 0.0
        self._total_process_ms = 0.0

        logger.info(
            "LLMBatcher initialized — max_batch=%d, max_wait=%dms",
            max_batch,
            max_wait_ms,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the batch processing worker."""
        if self._running:
            logger.warning("Batcher already running")
            return

        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("LLMBatcher started")

    async def stop(self) -> None:
        """Stop the batch processing worker gracefully."""
        self._running = False
        if self._worker_task is not None:
            # Put a sentinel to wake up the worker
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        logger.info("LLMBatcher stopped")

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    async def submit(
        self,
        prompt: str,
        request_id: str = "",
        use_chat: bool = False,
        messages: Optional[List[Dict[str, str]]] = None,
        timeout_s: float = 30.0,
    ) -> BatchResult:
        """
        Submit a request and wait for the result.

        Args:
            prompt: Formatted prompt string.
            request_id: Unique request identifier.
            use_chat: Use chat-style inference instead of completion.
            messages: Chat messages (required if use_chat=True).
            timeout_s: Maximum time to wait for result.

        Returns:
            BatchResult with output or error.
        """
        if not request_id:
            request_id = f"req_{self._total_processed + self._queue.qsize()}"

        loop = asyncio.get_event_loop()
        future = loop.create_future()

        request = BatchRequest(
            request_id=request_id,
            prompt=prompt,
            future=future,
            use_chat=use_chat,
            messages=messages,
        )

        await self._queue.put(request)
        logger.debug("Batcher submit [%s]: queued (queue=%d)", request_id, self._queue.qsize())

        try:
            result = await asyncio.wait_for(future, timeout=timeout_s)
            return result
        except asyncio.TimeoutError:
            self._total_failed += 1
            logger.error("Batcher timeout for request %s", request_id)
            return BatchResult(
                request_id=request_id,
                output={},
                success=False,
                error=f"Timeout after {timeout_s}s",
            )

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    async def _worker_loop(self) -> None:
        """Main worker loop: collect batches and process them."""
        logger.debug("Batcher worker loop started")

        while self._running:
            try:
                batch = await self._collect_batch()
                if batch:
                    await self._process_batch(batch)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Batcher worker error: %s", e)
                await asyncio.sleep(0.1)

        # Process remaining items
        remaining = []
        while not self._queue.empty():
            try:
                remaining.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        if remaining:
            await self._process_batch(remaining)

        logger.debug("Batcher worker loop ended")

    async def _collect_batch(self) -> List[BatchRequest]:
        """
        Collect up to max_batch requests, waiting up to max_wait_ms.

        Returns:
            List of collected BatchRequest objects.
        """
        batch: List[BatchRequest] = []

        # Wait for first request (blocks until available)
        try:
            first = await asyncio.wait_for(
                self._queue.get(), timeout=1.0  # Check running flag periodically
            )
            batch.append(first)
        except asyncio.TimeoutError:
            return batch  # Empty batch, loop will continue

        # Collect more requests up to max_batch, with max_wait_ms timeout
        deadline = time.time() + self.max_wait_ms / 1000.0
        while len(batch) < self.max_batch:
            remaining_s = deadline - time.time()
            if remaining_s <= 0:
                break
            try:
                req = await asyncio.wait_for(self._queue.get(), timeout=remaining_s)
                batch.append(req)
            except asyncio.TimeoutError:
                break

        if batch:
            self._total_batches += 1
            logger.debug("Collected batch of %d requests", len(batch))

        return batch

    async def _process_batch(self, batch: List[BatchRequest]) -> None:
        """
        Process a batch of requests sequentially.

        Args:
            batch: List of BatchRequest objects to process.
        """
        for request in batch:
            t0 = time.perf_counter()
            try:
                if self._process_fn is not None:
                    output = await self._run_process_fn(request)
                elif self.runner is not None:
                    output = await self._run_inference(request)
                else:
                    output = {"error": "No runner or process_fn configured"}

                elapsed_ms = (time.perf_counter() - t0) * 1000
                self._total_process_ms += elapsed_ms
                self._total_processed += 1

                result = BatchResult(
                    request_id=request.request_id,
                    output=output,
                    processing_time_ms=elapsed_ms,
                    success=True,
                )

            except Exception as e:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                self._total_failed += 1
                logger.error(
                    "Batcher process error [%s]: %s", request.request_id, e
                )
                result = BatchResult(
                    request_id=request.request_id,
                    output={},
                    processing_time_ms=elapsed_ms,
                    success=False,
                    error=str(e),
                )

            # Resolve the future
            if not request.future.done():
                request.future.set_result(result)

    async def _run_inference(self, request: BatchRequest) -> Dict[str, Any]:
        """
        Run inference through the GGUFRunner.

        Args:
            request: Batch request with prompt or messages.

        Returns:
            Parsed JSON output dict.
        """
        loop = asyncio.get_event_loop()

        if request.use_chat and request.messages:
            # Chat-style inference (run in executor to avoid blocking)
            output = await loop.run_in_executor(
                None, self.runner.chat, request.messages
            )
        else:
            # Completion-style inference
            output = await loop.run_in_executor(
                None, self.runner.generate, request.prompt
            )

        return output

    async def _run_process_fn(self, request: BatchRequest) -> Dict[str, Any]:
        """
        Run custom process function.

        Args:
            request: Batch request.

        Returns:
            Output dict from process function.
        """
        if asyncio.iscoroutinefunction(self._process_fn):
            return await self._process_fn(request.prompt)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._process_fn, request.prompt)

    # ------------------------------------------------------------------
    # Stats & properties
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """Check if batcher is running."""
        return self._running

    @property
    def queue_size(self) -> int:
        """Current queue size."""
        return self._queue.qsize()

    @property
    def stats(self) -> Dict[str, Any]:
        """Get batcher statistics."""
        avg_ms = (
            self._total_process_ms / self._total_processed
            if self._total_processed > 0
            else 0.0
        )
        return {
            "total_processed": self._total_processed,
            "total_failed": self._total_failed,
            "total_batches": self._total_batches,
            "avg_process_ms": round(avg_ms, 1),
            "total_process_ms": round(self._total_process_ms, 1),
            "queue_size": self.queue_size,
            "is_running": self._running,
            "max_batch": self.max_batch,
            "max_wait_ms": self.max_wait_ms,
        }
