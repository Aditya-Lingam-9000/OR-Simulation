"""
OR-Symphony: MedGemma GGUF Runner

Runs MedGemma inference via llama-cpp-python using the quantized GGUF model.
Supports CPU and GPU (via n_gpu_layers).

Features:
  - Real model loading with llama-cpp-python
  - JSON output extraction with retry and fallback
  - Stub mode for unit tests (no model file required)
  - Configurable context window, temperature, max tokens
  - Thread-safe for single-writer inference

Usage:
    from src.llm.gguf_runner import GGUFRunner
    runner = GGUFRunner()
    runner.load_model()
    output = runner.generate(prompt)
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.constants import (
    LLM_CONTEXT_WINDOW,
    LLM_MAX_OUTPUT_TOKENS,
    LLM_TEMPERATURE,
    MEDGEMMA_MODEL_PATH,
)
from src.utils.device import get_gguf_gpu_layers

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _suppress_stderr():
    """Temporarily redirect stderr to devnull.

    llama.cpp's tokenizer emits thousands of 'control token not marked
    as EOG' warnings via C-level fprintf(stderr, …) for large-vocab
    models like MedGemma (262 K tokens).  These bypass Python logging
    and cannot be silenced with verbose=False.  We redirect the *OS*
    file-descriptor so even the C layer is muted.
    """
    try:
        stderr_fd = sys.stderr.fileno()
        saved_fd = os.dup(stderr_fd)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stderr_fd)
        os.close(devnull)
        yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)


# ---------------------------------------------------------------------------
# Default stub response (used when model is not loaded or in stub mode)
# ---------------------------------------------------------------------------

_STUB_RESPONSE: Dict[str, Any] = {
    "metadata": {
        "reasoning": "stub_mode — no model loaded",
        "phase": "",
        "next_phase": "",
    },
    "machines": {"0": [], "1": []},
    "details": {},
    "suggestions": [],
    "confidence": 0.0,
    "source": "medgemma",
}


class GGUFRunner:
    """
    GGUF model runner using llama-cpp-python.

    Handles model loading, prompt formatting, and JSON output extraction.
    Supports CPU and partial/full GPU offloading.

    Args:
        model_path: Path to the GGUF model file.
        n_ctx: Context window size (tokens).
        n_gpu_layers: GPU layers to offload (-1=all, 0=CPU, None=auto-detect).
        temperature: Sampling temperature (low = more deterministic).
        max_tokens: Maximum output tokens.
        stub_mode: If True, skip model loading and return stub responses.
    """

    def __init__(
        self,
        model_path: str | Path = MEDGEMMA_MODEL_PATH,
        n_ctx: int = LLM_CONTEXT_WINDOW,
        n_gpu_layers: Optional[int] = None,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_OUTPUT_TOKENS,
        stub_mode: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_gpu_layers = (
            n_gpu_layers if n_gpu_layers is not None else get_gguf_gpu_layers()
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stub_mode = stub_mode

        self._model: Any = None
        self._loaded = False

        # Stats
        self._total_inferences = 0
        self._total_failures = 0
        self._total_time_ms = 0.0

        logger.info(
            "GGUFRunner initialized — model=%s, ctx=%d, gpu_layers=%d, stub=%s",
            self.model_path.name,
            n_ctx,
            self.n_gpu_layers,
            stub_mode,
        )

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> bool:
        """
        Load the GGUF model into memory.

        Returns:
            True if loaded successfully, False otherwise.
        """
        if self.stub_mode:
            logger.info("GGUFRunner in stub mode — skipping model load")
            self._loaded = False
            return True

        if self._loaded:
            logger.warning("Model already loaded — skipping reload")
            return True

        if not self.model_path.exists():
            logger.error("GGUF model not found: %s", self.model_path)
            return False

        file_size = self.model_path.stat().st_size
        logger.info(
            "Loading GGUF model: %s (%.2f GB, gpu_layers=%s)",
            self.model_path,
            file_size / (1024**3),
            self.n_gpu_layers,
        )

        if file_size < 1000:
            logger.error("GGUF file too small (%d bytes) — likely corrupt", file_size)
            return False

        try:
            from llama_cpp import Llama

            t0 = time.perf_counter()
            logger.info(
                "Loading GGUF model (suppressing tokenizer warnings)..."
            )
            with _suppress_stderr():
                self._model = Llama(
                    model_path=str(self.model_path),
                    n_ctx=self.n_ctx,
                    n_gpu_layers=self.n_gpu_layers,
                    verbose=False,
                )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._loaded = True
            logger.info(
                "GGUF model loaded in %.0fms — %s (%.2f GB, gpu_layers=%s)",
                elapsed_ms,
                self.model_path.name,
                file_size / (1024**3),
                self.n_gpu_layers,
            )
            return True

        except ImportError:
            logger.error("llama-cpp-python not installed — cannot load GGUF model")
            return False
        except Exception as e:
            logger.error("GGUF load failed (gpu_layers=%s): %s", self.n_gpu_layers, e)
            # Retry with CPU-only if GPU offload was requested
            if self.n_gpu_layers != 0:
                logger.warning("Retrying GGUF load with gpu_layers=0 (CPU only)...")
                try:
                    from llama_cpp import Llama

                    t0 = time.perf_counter()
                    with _suppress_stderr():
                        self._model = Llama(
                            model_path=str(self.model_path),
                            n_ctx=self.n_ctx,
                            n_gpu_layers=0,
                            verbose=False,
                        )
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    self._loaded = True
                    self.n_gpu_layers = 0
                    logger.info(
                        "GGUF model loaded (CPU fallback) in %.0fms — %s",
                        elapsed_ms,
                        self.model_path.name,
                    )
                    return True
                except Exception as e2:
                    logger.error("GGUF load failed even on CPU: %s", e2)
            return False

    def unload(self) -> None:
        """Release model resources."""
        if self._model is not None:
            del self._model
            self._model = None
        self._loaded = False
        logger.info("GGUFRunner model unloaded")

    def is_loaded(self) -> bool:
        """Check if model is loaded and ready for inference."""
        return self._loaded

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate a structured JSON response from the GGUF model.

        Args:
            prompt: Formatted prompt string (system + user content).
            temperature: Override default temperature.
            max_tokens: Override default max output tokens.

        Returns:
            Parsed JSON dict. Returns stub response on failure.
        """
        # Stub mode — return immediately
        if self.stub_mode:
            self._total_inferences += 1
            return dict(_STUB_RESPONSE)

        # Model not loaded — return stub
        if not self._loaded or self._model is None:
            logger.warning("Model not loaded — returning stub response")
            self._total_failures += 1
            return dict(_STUB_RESPONSE)

        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        t0 = time.perf_counter()
        try:
            response = self._model(
                prompt,
                max_tokens=max_tok,
                temperature=temp,
                stop=["```", "\n\n\n"],  # Stop at triple backtick or triple newline
                echo=False,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._total_time_ms += elapsed_ms
            self._total_inferences += 1

            # Extract text from response
            raw_text = self._extract_text(response)
            logger.debug(
                "GGUF inference: %.0fms, output=%d chars",
                elapsed_ms,
                len(raw_text),
            )

            # Parse JSON from response
            parsed = self._parse_json_output(raw_text)

            if elapsed_ms > 5000:
                logger.warning("GGUF inference slow: %.0fms", elapsed_ms)

            return parsed

        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._total_failures += 1
            logger.error("GGUF inference failed after %.0fms: %s", elapsed_ms, e)
            return dict(_STUB_RESPONSE)

    def generate_raw(self, prompt: str) -> str:
        """
        Generate raw text response (no JSON parsing).

        Args:
            prompt: Input prompt.

        Returns:
            Raw model output text.
        """
        if self.stub_mode or not self._loaded or self._model is None:
            return '{"metadata": {}, "machines": {"0": [], "1": []}, "details": {}, "suggestions": [], "confidence": 0.0, "source": "medgemma"}'

        try:
            response = self._model(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                echo=False,
            )
            return self._extract_text(response)
        except Exception as e:
            logger.error("Raw generation failed: %s", e)
            return ""

    # ------------------------------------------------------------------
    # Chat-style inference
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run chat-style inference with message list.

        Args:
            messages: List of {"role": "system"|"user"|"assistant", "content": ...}
            temperature: Override temperature.
            max_tokens: Override max tokens.

        Returns:
            Parsed JSON dict from model response.
        """
        if self.stub_mode:
            self._total_inferences += 1
            return dict(_STUB_RESPONSE)

        if not self._loaded or self._model is None:
            logger.warning("Model not loaded — returning stub response")
            self._total_failures += 1
            return dict(_STUB_RESPONSE)

        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        t0 = time.perf_counter()
        try:
            response = self._model.create_chat_completion(
                messages=messages,
                max_tokens=max_tok,
                temperature=temp,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._total_time_ms += elapsed_ms
            self._total_inferences += 1

            raw_text = response["choices"][0]["message"]["content"] or ""
            logger.info(
                "GGUF chat: %.0fms, output=%d chars, preview=%.200s",
                elapsed_ms, len(raw_text), raw_text[:200],
            )
            return self._parse_json_output(raw_text)

        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._total_failures += 1
            logger.error("GGUF chat failed after %.0fms: %s", elapsed_ms, e)
            return dict(_STUB_RESPONSE)

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Extract text content from llama-cpp response."""
        try:
            return response["choices"][0]["text"]
        except (KeyError, IndexError, TypeError):
            return str(response) if response else ""

    @staticmethod
    def _parse_json_output(raw_text: str) -> Dict[str, Any]:
        """
        Extract and parse JSON from model output text.

        Tries multiple strategies:
        1. Direct JSON parse of the full text
        2. Find outermost { ... } block and parse
        3. Find JSON after common prefixes like "```json"
        4. Return stub on failure

        Args:
            raw_text: Raw model output string.

        Returns:
            Parsed JSON dict.

        Raises:
            ValueError: If no valid JSON could be extracted.
        """
        text = raw_text.strip()
        if not text:
            raise ValueError("Empty model output")

        # Strategy 1: direct parse
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        # Strategy 2: find outermost { ... }
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            candidate = text[brace_start : brace_end + 1]
            try:
                result = json.loads(candidate)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

        # Strategy 3: look for ```json blocks
        json_block_match = re.search(
            r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL
        )
        if json_block_match:
            try:
                result = json.loads(json_block_match.group(1).strip())
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

        raise ValueError(f"No valid JSON found in model output: {text[:200]}")

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        avg_ms = (
            self._total_time_ms / self._total_inferences
            if self._total_inferences > 0
            else 0.0
        )
        return {
            "total_inferences": self._total_inferences,
            "total_failures": self._total_failures,
            "total_time_ms": round(self._total_time_ms, 1),
            "avg_inference_ms": round(avg_ms, 1),
            "model_loaded": self._loaded,
            "stub_mode": self.stub_mode,
            "model_path": str(self.model_path),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # --- Stub mode demo ---
    print("=== Stub Mode Demo ===")
    runner = GGUFRunner(stub_mode=True)
    runner.load_model()
    result = runner.generate("test prompt")
    print(f"Stub result: {json.dumps(result, indent=2)}")
    print(f"Stats: {runner.stats}")

    # --- JSON parsing demo ---
    print("\n=== JSON Parsing Demo ===")
    test_cases = [
        '{"machines": {"0": [], "1": ["M01"]}, "confidence": 0.9}',
        'Here is the JSON output:\n```json\n{"machines": {"0": [], "1": ["M03"]}}\n```',
        'The analysis shows:\n{"metadata": {"phase": "Phase3"}, "machines": {"0": [], "1": []}}',
    ]
    for i, test in enumerate(test_cases):
        try:
            parsed = GGUFRunner._parse_json_output(test)
            print(f"  Case {i + 1}: OK — {list(parsed.keys())}")
        except ValueError as e:
            print(f"  Case {i + 1}: FAIL — {e}")

    # --- Real model demo (optional) ---
    print("\n=== Real Model Demo ===")
    real_runner = GGUFRunner(n_gpu_layers=0)
    loaded = real_runner.load_model()
    if loaded and real_runner.is_loaded():
        prompt = "You are a surgical AI assistant. Given the transcript 'turn on the fluoroscopy', output a JSON with machine state changes."
        result = real_runner.generate(prompt)
        print(f"Model result: {json.dumps(result, indent=2)}")
        print(f"Stats: {real_runner.stats}")
        real_runner.unload()
    else:
        print("Model not loaded (expected without GGUF file in CI)")

