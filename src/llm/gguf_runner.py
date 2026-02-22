"""
OR-Symphony: MedGemma GGUF Runner

Runs MedGemma inference via llama-cpp-python using the quantized GGUF model.
Supports CPU and GPU (via n_gpu_layers).

Full implementation in Phase 6.

Usage:
    from src.llm.gguf_runner import GGUFRunner
    runner = GGUFRunner(model_path="onnx_models/medgemma/medgemma-4b-it-Q3_K_M.gguf")
    runner.load_model()
    output = runner.generate(prompt)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.constants import (
    LLM_CONTEXT_WINDOW,
    LLM_MAX_OUTPUT_TOKENS,
    LLM_TEMPERATURE,
    MEDGEMMA_MODEL_PATH,
)
from src.utils.device import get_gguf_gpu_layers

logger = logging.getLogger(__name__)


class GGUFRunner:
    """
    GGUF model runner using llama-cpp-python.

    Handles model loading, prompt formatting, and JSON output extraction.
    Supports CPU and partial/full GPU offloading.

    Full implementation in Phase 6.
    """

    def __init__(
        self,
        model_path: str | Path = MEDGEMMA_MODEL_PATH,
        n_ctx: int = LLM_CONTEXT_WINDOW,
        n_gpu_layers: Optional[int] = None,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_OUTPUT_TOKENS,
    ) -> None:
        """
        Initialize the GGUF runner.

        Args:
            model_path: Path to the GGUF model file.
            n_ctx: Context window size.
            n_gpu_layers: GPU layers to offload (-1=all, 0=CPU, None=auto).
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
        """
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers if n_gpu_layers is not None else get_gguf_gpu_layers()
        self.temperature = temperature
        self.max_tokens = max_tokens

        self._model = None
        self._loaded = False

        logger.info(
            "GGUFRunner initialized — model=%s, ctx=%d, gpu_layers=%d",
            self.model_path.name,
            n_ctx,
            self.n_gpu_layers,
        )

    def load_model(self) -> None:
        """Load the GGUF model. Full implementation in Phase 6."""
        # TODO: Phase 6 — load via llama_cpp.Llama()
        # from llama_cpp import Llama
        # self._model = Llama(
        #     model_path=str(self.model_path),
        #     n_ctx=self.n_ctx,
        #     n_gpu_layers=self.n_gpu_layers,
        #     verbose=False,
        # )
        logger.info("GGUFRunner model load — placeholder (Phase 6)")
        self._loaded = False  # Will be True after Phase 6 implementation

    def generate(self, prompt: str) -> Dict[str, Any]:
        """
        Generate a response from the GGUF model.

        Args:
            prompt: Formatted prompt string.

        Returns:
            Parsed JSON output dict.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if not self._loaded:
            logger.warning("GGUF model not loaded — returning stub response")
            return {
                "metadata": {"reasoning": "model_not_loaded"},
                "machines": {"0": [], "1": []},
                "details": {},
                "suggestions": ["GGUF model not yet loaded (Phase 6)"],
                "confidence": 0.0,
                "source": "medgemma",
            }

        # TODO: Phase 6 — actual inference
        # response = self._model(prompt, max_tokens=self.max_tokens, temperature=self.temperature)
        # return self._parse_json_output(response)
        return {}

    def _parse_json_output(self, raw_response: Any) -> Dict[str, Any]:
        """
        Extract and parse JSON from model output.

        Args:
            raw_response: Raw model response.

        Returns:
            Parsed JSON dict.

        Raises:
            ValueError: If output is not valid JSON.
        """
        # TODO: Phase 6 — extract text from response and parse JSON
        try:
            text = raw_response["choices"][0]["text"]
            # Try to find JSON in the output
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
            raise ValueError(f"No JSON found in model output: {text[:100]}")
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error("Failed to parse JSON from model output: %s", e)
            raise ValueError(f"Invalid JSON output: {e}") from e

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def unload(self) -> None:
        """Release model resources."""
        self._model = None
        self._loaded = False
        logger.info("GGUFRunner model unloaded")
