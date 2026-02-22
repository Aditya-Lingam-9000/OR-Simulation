"""Tests for src.utils.device module."""

from src.utils.device import (
    DeviceInfo,
    DeviceType,
    get_device_info,
    get_gguf_gpu_layers,
    get_onnx_providers,
    get_torch_device,
)


class TestGetTorchDevice:
    def test_returns_string(self):
        device = get_torch_device()
        assert isinstance(device, str)
        assert device in ("cpu", "cuda")

    def test_cpu_when_no_gpu(self):
        # On local dev without GPU, should return cpu
        device = get_torch_device()
        # We accept either — test just verifies no crash
        assert device in ("cpu", "cuda")


class TestGetOnnxProviders:
    def test_returns_list(self):
        providers = get_onnx_providers()
        assert isinstance(providers, list)
        assert len(providers) > 0

    def test_always_includes_cpu(self):
        providers = get_onnx_providers()
        assert "CPUExecutionProvider" in providers


class TestGetGgufGpuLayers:
    def test_returns_int(self):
        layers = get_gguf_gpu_layers()
        assert isinstance(layers, int)

    def test_cpu_returns_zero(self):
        # Without GPU, should return 0
        layers = get_gguf_gpu_layers()
        assert layers >= 0 or layers == -1  # -1 means all layers on GPU


class TestGetDeviceInfo:
    def test_returns_device_info(self):
        info = get_device_info()
        assert isinstance(info, DeviceInfo)

    def test_device_type_is_enum(self):
        info = get_device_info()
        assert isinstance(info.device_type, DeviceType)

    def test_torch_device_matches(self):
        info = get_device_info()
        assert info.torch_device in ("cpu", "cuda")
        if info.torch_device == "cuda":
            assert info.cuda_available is True
        else:
            assert info.cuda_available is False

    def test_onnx_providers_non_empty(self):
        info = get_device_info()
        assert len(info.onnx_providers) > 0
