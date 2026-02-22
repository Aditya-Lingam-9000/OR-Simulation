/**
 * pcm-processor.js — AudioWorklet processor for real-time microphone capture.
 *
 * Runs in the audio rendering thread. Collects incoming samples,
 * resamples from the native rate (e.g., 48kHz) to 16kHz, and posts
 * float32 PCM buffers to the main thread every ~100ms.
 *
 * Registered as: 'pcm-processor'
 */

const TARGET_RATE = 16000;
const SEND_INTERVAL_FRAMES = 4800; // ~300ms at 16kHz → send buffer every ~0.3s

class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = [];
    this._sampleRate = sampleRate; // global `sampleRate` from AudioWorkletGlobalScope
    this._ratio = TARGET_RATE / this._sampleRate;
    this._resampleBuffer = new Float32Array(0);
    this._active = true;

    this.port.onmessage = (e) => {
      if (e.data === 'stop') {
        this._active = false;
      }
    };
  }

  /**
   * Simple linear interpolation resampler.
   * @param {Float32Array} input - Input samples at native rate
   * @returns {Float32Array} Resampled samples at 16kHz
   */
  _resample(input) {
    if (this._sampleRate === TARGET_RATE) return input;

    const outLen = Math.round(input.length * this._ratio);
    const out = new Float32Array(outLen);
    const step = (input.length - 1) / (outLen - 1 || 1);

    for (let i = 0; i < outLen; i++) {
      const srcIdx = i * step;
      const lo = Math.floor(srcIdx);
      const hi = Math.min(lo + 1, input.length - 1);
      const frac = srcIdx - lo;
      out[i] = input[lo] * (1 - frac) + input[hi] * frac;
    }
    return out;
  }

  process(inputs) {
    if (!this._active) return false;

    const input = inputs[0];
    if (!input || !input[0]) return true;

    const mono = input[0]; // first channel
    const resampled = this._resample(mono);

    // Accumulate
    const combined = new Float32Array(this._resampleBuffer.length + resampled.length);
    combined.set(this._resampleBuffer);
    combined.set(resampled, this._resampleBuffer.length);
    this._resampleBuffer = combined;

    // Send when we have enough samples
    if (this._resampleBuffer.length >= SEND_INTERVAL_FRAMES) {
      this.port.postMessage(this._resampleBuffer);
      this._resampleBuffer = new Float32Array(0);
    }

    return true;
  }
}

registerProcessor('pcm-processor', PCMProcessor);
