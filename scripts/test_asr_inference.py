"""
OR-Symphony: Phase 3 — ASR Inference Test Script

Stand-alone script to test the full MedASR inference pipeline:
  1. Generate or load audio
  2. Run feature extraction → ONNX inference → CTC decode
  3. Report latency, decoded text, confidence, and RTF

Usage:
    python scripts/test_asr_inference.py
    python scripts/test_asr_inference.py --wav path/to/audio.wav
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.asr.onnx_runner import OnnxASRRunner
from src.utils.constants import AUDIO_SAMPLE_RATE, MEDASR_MODEL_PATH


def generate_test_signals() -> list[tuple[str, np.ndarray, float]]:
    """Generate a set of test audio signals.

    Returns list of (name, audio, duration_s) tuples.
    """
    sr = AUDIO_SAMPLE_RATE
    signals = []

    # 1. Silence (1s)
    signals.append(("silence_1s", np.zeros(sr, dtype=np.float32), 1.0))

    # 2. White noise (1s)
    np.random.seed(42)
    noise = (np.random.randn(sr) * 0.1).astype(np.float32)
    signals.append(("white_noise_1s", noise, 1.0))

    # 3. Speech-like harmonics (2s)
    t = np.linspace(0, 2.0, 2 * sr, dtype=np.float32)
    speech = np.zeros_like(t)
    for freq, amp in [(150, 0.3), (300, 0.2), (450, 0.1), (600, 0.08)]:
        speech += amp * np.sin(2 * np.pi * freq * t)
    speech += 0.02 * np.random.randn(len(t)).astype(np.float32)
    speech = (speech / np.max(np.abs(speech)) * 0.7).astype(np.float32)
    signals.append(("speech_harmonics_2s", speech, 2.0))

    # 4. Tone burst (0.5s per burst, 3s total)
    dur = 3.0
    n = int(sr * dur)
    t = np.linspace(0, dur, n, dtype=np.float32)
    tone = np.zeros(n, dtype=np.float32)
    for start_s in [0.5, 1.5, 2.5]:
        mask = (t >= start_s) & (t < start_s + 0.3)
        tone[mask] = 0.5 * np.sin(2 * np.pi * 440 * t[mask])
    signals.append(("tone_bursts_3s", tone, dur))

    return signals


def load_wav(path: str) -> tuple[np.ndarray, int]:
    """Load a WAV file as float32 mono."""
    import wave

    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if sample_width == 2:
        dtype = np.int16
    elif sample_width == 4:
        dtype = np.int32
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    audio = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    if dtype == np.int16:
        audio /= 32768.0
    elif dtype == np.int32:
        audio /= 2147483648.0

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels)[:, 0]

    return audio, sr


def main():
    parser = argparse.ArgumentParser(description="Test MedASR inference pipeline")
    parser.add_argument("--wav", type=str, help="Path to WAV file for testing")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    print("=" * 70)
    print("OR-Symphony: MedASR Inference Test")
    print("=" * 70)

    if not MEDASR_MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MEDASR_MODEL_PATH}")
        sys.exit(1)

    # Load model
    print("\nLoading MedASR model...")
    t0 = time.perf_counter()
    runner = OnnxASRRunner()
    runner.load_model()
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"  Model loaded in {load_ms:.0f}ms")

    info = runner.get_model_info()
    print(f"  Model type: {info.get('model_type', '?')}")
    print(f"  Vocab size: {info.get('vocab_size', '?')}")
    print(f"  Providers: {info.get('providers', '?')}")

    # Prepare test signals
    if args.wav:
        wav_path = Path(args.wav)
        if not wav_path.exists():
            print(f"ERROR: WAV file not found: {wav_path}")
            sys.exit(1)
        audio, sr = load_wav(str(wav_path))
        if sr != AUDIO_SAMPLE_RATE:
            print(f"WARNING: Resampling from {sr}Hz to {AUDIO_SAMPLE_RATE}Hz not implemented.")
            print("         Please provide 16kHz audio.")
            sys.exit(1)
        signals = [(wav_path.name, audio, len(audio) / sr)]
    else:
        signals = generate_test_signals()

    # Run inference
    print(f"\n{'='*70}")
    print(f"{'Signal':<25} {'Duration':>8} {'Latency':>10} {'RTF':>8} {'Conf':>8}  Text")
    print(f"{'-'*25} {'-'*8} {'-'*10} {'-'*8} {'-'*8}  {'-'*20}")

    for name, audio, duration_s in signals:
        result = runner.transcribe(audio)
        rtf = (result.processing_time_ms / 1000.0) / duration_s if duration_s > 0 else 0.0
        conf = result.segments[0].confidence if result.segments else 0.0
        text = result.full_text or "(empty)"

        print(
            f"{name:<25} {duration_s:>6.1f}s {result.processing_time_ms:>8.1f}ms "
            f"{rtf:>7.3f}x {conf:>7.3f}  {text[:40]}"
        )

    # Print latency summary
    stats = runner.latency_stats
    print(f"\n{'='*70}")
    print("Latency Summary:")
    print(f"  Inference count: {stats['inference_count']}")
    print(f"  Average latency: {stats['avg_ms']:.1f}ms")
    print(f"  Average RTF:     {stats['avg_rtf']:.3f}x")
    print(f"  Total audio:     {stats['total_audio_s']:.1f}s")

    # Pass/fail
    target_ms = 400
    passed = stats["avg_ms"] < target_ms
    status = "PASS" if passed else "FAIL"
    print(f"\n  Latency target: <{target_ms}ms → {status}")

    runner.unload_model()
    print("\nDone.")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
