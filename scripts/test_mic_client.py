"""
OR-Symphony: Microphone Test Client

Records live audio from the default microphone for a configurable duration,
runs it through VAD + chunking, and saves the resulting chunks as WAV files.

Usage:
    python -m scripts.test_mic_client                     # 10s recording
    python -m scripts.test_mic_client --duration 30       # 30s recording
    python -m scripts.test_mic_client --no-save           # don't save WAVs
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.ingest.audio_utils import save_wav, validate_wav
from src.ingest.mic_server import MicrophoneCapture
from src.utils.constants import AUDIO_SAMPLE_RATE, TMP_DIR


async def run_mic_test(duration_s: float, save: bool) -> None:
    """Record from mic, run VAD, produce and inspect chunks."""

    save_dir = TMP_DIR / "audio_chunks"

    print(f"🎤 Starting microphone capture for {duration_s:.0f}s...")
    print(f"   Sample rate: {AUDIO_SAMPLE_RATE} Hz")
    print(f"   Save chunks: {save} → {save_dir if save else 'disabled'}")
    print()

    mic = MicrophoneCapture(
        save_chunks=save,
        save_dir=save_dir,
    )

    chunks = []
    start = time.monotonic()

    await mic.start()

    try:
        while (time.monotonic() - start) < duration_s:
            chunk = await mic.get_chunk(timeout=0.5)
            if chunk is not None:
                chunks.append(chunk)
                elapsed = time.monotonic() - start
                print(
                    f"  [{elapsed:5.1f}s] Chunk #{chunk.chunk_id}: "
                    f"{chunk.duration_s:.2f}s "
                    f"[{chunk.timestamp_start:.1f}-{chunk.timestamp_end:.1f}]"
                )
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    finally:
        await mic.stop()

    elapsed_total = time.monotonic() - start
    stats = mic.stats if hasattr(mic, "stats") else {}

    print()
    print("=" * 50)
    print(f"Recording complete: {elapsed_total:.1f}s")
    print(f"Chunks produced: {len(chunks)}")
    if stats:
        print(f"Frames processed: {stats.get('frames_processed', '?')}")
    print()

    # Validate saved WAV files
    if save and chunks:
        print("Validating saved chunks:")
        for wav_file in sorted(save_dir.glob("chunk_*.wav")):
            result = validate_wav(wav_file)
            status = "✅" if result.valid else "❌"
            print(f"  {status} {wav_file.name}: {result}")

    # Summary
    if chunks:
        durations = [c.duration_s for c in chunks]
        print(f"\nChunk durations: min={min(durations):.2f}s, max={max(durations):.2f}s, avg={sum(durations)/len(durations):.2f}s")
    else:
        print("\n⚠️  No speech chunks detected. Try speaking louder or check your microphone.")


def main() -> None:
    parser = argparse.ArgumentParser(description="OR-Symphony Microphone Test")
    parser.add_argument("--duration", type=float, default=10.0, help="Recording duration in seconds")
    parser.add_argument("--no-save", action="store_true", help="Don't save WAV files")
    args = parser.parse_args()

    asyncio.run(run_mic_test(args.duration, save=not args.no_save))


if __name__ == "__main__":
    main()
