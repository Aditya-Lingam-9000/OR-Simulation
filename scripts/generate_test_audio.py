"""
OR-Symphony: Generate Synthetic Test Audio

Creates WAV files with known patterns for testing the audio pipeline
without requiring a real microphone.

Generated files saved to tmp/test_audio/:
  - sine_440hz_2s.wav       (pure tone)
  - silence_5s.wav          (all zeros)
  - white_noise_2s.wav      (random noise)
  - speech_like_3s.wav      (multi-harmonic + noise)
  - speech_silence_mix.wav  (alternating speech/silence)

Usage:
    python -m scripts.generate_test_audio
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.ingest.audio_utils import (
    generate_silence,
    generate_sine,
    generate_speech_like,
    generate_speech_silence_pattern,
    generate_white_noise,
    save_wav,
    validate_wav,
)
from src.utils.constants import AUDIO_SAMPLE_RATE, TMP_DIR


def main() -> None:
    out_dir = TMP_DIR / "test_audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    sr = AUDIO_SAMPLE_RATE

    files = []

    # 1. Pure sine wave
    sine = generate_sine(frequency=440, duration_s=2.0, sample_rate=sr)
    p = save_wav(out_dir / "sine_440hz_2s.wav", sine, sr)
    files.append(p)
    print(f"  [1/5] {p.name} — {len(sine)/sr:.1f}s")

    # 2. Silence
    sil = generate_silence(duration_s=5.0, sample_rate=sr)
    p = save_wav(out_dir / "silence_5s.wav", sil, sr)
    files.append(p)
    print(f"  [2/5] {p.name} — {len(sil)/sr:.1f}s")

    # 3. White noise
    noise = generate_white_noise(duration_s=2.0, sample_rate=sr)
    p = save_wav(out_dir / "white_noise_2s.wav", noise, sr)
    files.append(p)
    print(f"  [3/5] {p.name} — {len(noise)/sr:.1f}s")

    # 4. Speech-like
    speech = generate_speech_like(duration_s=3.0, sample_rate=sr)
    p = save_wav(out_dir / "speech_like_3s.wav", speech, sr)
    files.append(p)
    print(f"  [4/5] {p.name} — {len(speech)/sr:.1f}s")

    # 5. Alternating speech + silence
    mix = generate_speech_silence_pattern(
        speech_duration_s=1.0,
        silence_duration_s=1.0,
        repetitions=3,
        sample_rate=sr,
    )
    p = save_wav(out_dir / "speech_silence_mix.wav", mix, sr)
    files.append(p)
    print(f"  [5/5] {p.name} — {len(mix)/sr:.1f}s")

    # Validate all
    print("\nValidation:")
    for f in files:
        result = validate_wav(f)
        status = "✅" if result.valid else "❌"
        print(f"  {status} {f.name}: {result}")

    print(f"\nAll files saved to: {out_dir}")


if __name__ == "__main__":
    print("Generating synthetic test audio...")
    main()
    print("Done.")
