"""
OR-Symphony: Device Helper Utilities

Provides unified device detection for PyTorch, ONNX Runtime, and llama-cpp-python.
Handles CPU/GPU selection dynamically based on hardware availability.

Sanity check:
    python -m src.utils.device
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List

logger = logging.getLogger(__name__)


class DeviceType(str, Enum):
    """Supported device types."""

    CPU = "cpu"
    CUDA = "cuda"


@dataclass(frozen=True)
class DeviceInfo:
    """Immutable snapshot of current device capabilities."""

    device_type: DeviceType
    torch_device: str
    onnx_providers: List[str]
    gguf_gpu_layers: int
    cuda_available: bool
    cuda_device_name: str | None = None
    cuda_device_count: int = 0


def get_torch_device() -> str:
    """
    Get the appropriate PyTorch device string.

    Returns:
        'cuda' if GPU available, else 'cpu'.
    """
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
            logger.info("PyTorch device: CUDA (%s)", torch.cuda.get_device_name(0))
        else:
            device = "cpu"
            logger.info("PyTorch device: CPU (no CUDA available)")
        return device
    except ImportError:
        logger.warning("PyTorch not installed, defaulting to 'cpu'")
        return "cpu"


def get_onnx_providers() -> List[str]:
    """
    Get available ONNX Runtime execution providers.

    Prefers CUDA > CPU. Returns ordered list of providers.

    Returns:
        List of provider strings for onnxruntime.InferenceSession.
    """
    try:
        import onnxruntime as ort

        available = ort.get_available_providers()
        providers: List[str] = []

        # Prefer GPU providers
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
            logger.info("ONNX Runtime: CUDAExecutionProvider available")

        # Always include CPU as fallback
        if "CPUExecutionProvider" in available:
            providers.append("CPUExecutionProvider")

        if not providers:
            providers = ["CPUExecutionProvider"]
            logger.warning("No ONNX providers detected, defaulting to CPU")

        logger.info("ONNX Runtime providers: %s", providers)
        return providers
    except ImportError:
        logger.warning("onnxruntime not installed, returning CPU provider")
        return ["CPUExecutionProvider"]


def get_gguf_gpu_layers() -> int:
    """
    Determine how many layers to offload to GPU for GGUF models.

    Returns:
        Number of GPU layers (0 = CPU only, -1 = all layers on GPU).
    """
    try:
        import torch

        if torch.cuda.is_available():
            # Get available VRAM in GB
            vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            if vram_gb >= 8:
                logger.info("GGUF: offloading all layers to GPU (%.1f GB VRAM)", vram_gb)
                return -1  # All layers on GPU
            elif vram_gb >= 4:
                logger.info("GGUF: partial GPU offload (%.1f GB VRAM)", vram_gb)
                return 20  # Partial offload
            else:
                logger.info("GGUF: limited VRAM (%.1f GB), CPU only", vram_gb)
                return 0
        else:
            logger.info("GGUF: no CUDA, CPU inference only")
            return 0
    except ImportError:
        logger.info("GGUF: torch not available, CPU inference only")
        return 0


def get_device_info() -> DeviceInfo:
    """
    Get comprehensive device information for all inference backends.

    Returns:
        DeviceInfo dataclass with all device details.
    """
    torch_device = get_torch_device()
    onnx_providers = get_onnx_providers()
    gguf_layers = get_gguf_gpu_layers()

    cuda_available = torch_device == "cuda"
    cuda_name = None
    cuda_count = 0

    if cuda_available:
        try:
            import torch

            cuda_name = torch.cuda.get_device_name(0)
            cuda_count = torch.cuda.device_count()
        except Exception:
            pass

    return DeviceInfo(
        device_type=DeviceType.CUDA if cuda_available else DeviceType.CPU,
        torch_device=torch_device,
        onnx_providers=onnx_providers,
        gguf_gpu_layers=gguf_layers,
        cuda_available=cuda_available,
        cuda_device_name=cuda_name,
        cuda_device_count=cuda_count,
    )


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def print_device_summary() -> None:
    """Print a human-readable device summary to stdout."""
    info = get_device_info()
    print("=" * 50)
    print("OR-Symphony — Device Summary")
    print("=" * 50)
    print(f"  Device Type      : {info.device_type.value}")
    print(f"  Torch Device     : {info.torch_device}")
    print(f"  ONNX Providers   : {info.onnx_providers}")
    print(f"  GGUF GPU Layers  : {info.gguf_gpu_layers}")
    print(f"  CUDA Available   : {info.cuda_available}")
    if info.cuda_available:
        print(f"  CUDA Device      : {info.cuda_device_name}")
        print(f"  CUDA Count       : {info.cuda_device_count}")
    print("=" * 50)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print_device_summary()
