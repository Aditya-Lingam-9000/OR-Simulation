"""
OR-Symphony: ONNX ASR Runner

Concrete ASR implementation using ONNX Runtime for CPU/GPU inference.
Loads the MedASR INT8 quantized model.

Usage:
    runner = OnnxASRRunner("onnx_models/medasr/model.int8.onnx", "onnx_models/medasr/tokens.txt")
    runner.load_model()
    result = runner.transcribe(audio_array)

Sanity check:
    python -m src.asr.onnx_runner
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.asr.runner import ASRResult, BaseASRRunner, TranscriptSegment
from src.utils.device import get_onnx_providers

logger = logging.getLogger(__name__)


class OnnxASRRunner(BaseASRRunner):
    """
    ONNX Runtime-based ASR runner.

    Supports CPU and GPU execution providers.
    Uses quantized INT8 model for low-latency inference.
    """

    def __init__(
        self,
        model_path: str | Path,
        tokens_path: str | Path,
        providers: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the ONNX ASR runner.

        Args:
            model_path: Path to the ONNX model file.
            tokens_path: Path to the tokens/vocabulary file.
            providers: ONNX execution providers. Auto-detected if None.
        """
        self.model_path = Path(model_path)
        self.tokens_path = Path(tokens_path)
        self.providers = providers or get_onnx_providers()
        self._session = None
        self._tokens: Dict[int, str] = {}
        self._loaded = False

    def load_model(self) -> None:
        """Load the ONNX model and token vocabulary."""
        import onnxruntime as ort

        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        if not self.tokens_path.exists():
            raise FileNotFoundError(f"Tokens file not found: {self.tokens_path}")

        logger.info("Loading ONNX ASR model: %s", self.model_path)
        logger.info("Using providers: %s", self.providers)

        # Create session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4

        # Create inference session
        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=self.providers,
        )

        # Load token vocabulary
        self._tokens = self._load_tokens()
        logger.info("Loaded %d tokens from vocabulary", len(self._tokens))

        self._loaded = True
        logger.info("ONNX ASR model loaded successfully")

    def _load_tokens(self) -> Dict[int, str]:
        """Load token vocabulary from file."""
        tokens: Dict[int, str] = {}
        with open(self.tokens_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    token_str = parts[0]
                    token_id = int(parts[1])
                    tokens[token_id] = token_str
        return tokens

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> ASRResult:
        """
        Transcribe an audio chunk using the ONNX model.

        Args:
            audio: Audio data as numpy array (float32, mono, 16kHz).
            sample_rate: Audio sample rate in Hz.

        Returns:
            ASRResult with final transcript.
        """
        if not self._loaded or self._session is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start_time = time.perf_counter()

        # TODO: Implement actual inference when model I/O is mapped
        # For now, return a placeholder to allow pipeline integration
        # Actual implementation in Phase 3
        result = ASRResult(
            segments=[
                TranscriptSegment(
                    text="[ASR placeholder — inference not yet implemented]",
                    is_final=True,
                    confidence=0.0,
                )
            ],
            processing_time_ms=(time.perf_counter() - start_time) * 1000,
            model_name="medasr-int8-onnx",
            is_partial=False,
        )

        logger.debug(
            "ASR transcribe: %.1fms, text='%s'",
            result.processing_time_ms,
            result.full_text[:80],
        )
        return result

    def transcribe_streaming(self, audio: np.ndarray, sample_rate: int = 16000) -> ASRResult:
        """
        Streaming transcription — returns partial results.

        Args:
            audio: Audio chunk as numpy array.
            sample_rate: Sample rate in Hz.

        Returns:
            ASRResult with partial flag set.
        """
        # For streaming, we call transcribe and mark as partial
        result = self.transcribe(audio, sample_rate)
        result.is_partial = True
        return result

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def unload_model(self) -> None:
        """Release model resources."""
        self._session = None
        self._tokens = {}
        self._loaded = False
        logger.info("ONNX ASR model unloaded")

    def get_model_info(self) -> Dict[str, str]:
        """Get model metadata."""
        info = {
            "model_path": str(self.model_path),
            "tokens_path": str(self.tokens_path),
            "providers": str(self.providers),
            "loaded": str(self._loaded),
            "vocab_size": str(len(self._tokens)),
        }
        if self._session is not None:
            inputs = self._session.get_inputs()
            outputs = self._session.get_outputs()
            info["inputs"] = str([(i.name, i.shape, i.type) for i in inputs])
            info["outputs"] = str([(o.name, o.shape, o.type) for o in outputs])
        return info


if __name__ == "__main__":
    from src.utils.constants import MEDASR_MODEL_PATH, MEDASR_TOKENS_PATH

    logging.basicConfig(level=logging.INFO)

    print("Initializing ONNX ASR Runner...")
    runner = OnnxASRRunner(MEDASR_MODEL_PATH, MEDASR_TOKENS_PATH)

    if MEDASR_MODEL_PATH.exists():
        runner.load_model()
        info = runner.get_model_info()
        for k, v in info.items():
            print(f"  {k}: {v}")

        # Test with dummy audio
        dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        result = runner.transcribe(dummy_audio)
        print(f"  Test result: '{result.full_text}' ({result.processing_time_ms:.1f}ms)")

        runner.unload_model()
    else:
        print(f"  Model not found at {MEDASR_MODEL_PATH} — skipping load test")
