/**
 * MicRecorder — Browser microphone capture with WebSocket streaming.
 *
 * Uses AudioWorklet to capture, resample to 16kHz, and stream
 * float32 PCM audio to the backend via /ws/audio WebSocket.
 *
 * Props:
 *   backendUrl — optional base URL override (for remote Kaggle backend)
 */
import React, { useState, useRef, useCallback, useEffect } from "react";

const TARGET_SAMPLE_RATE = 16000;

/**
 * Build the WebSocket URL for audio streaming.
 * If backendUrl is set (e.g., ngrok URL), use it; otherwise use current host.
 */
function getAudioWsUrl(backendUrl) {
  if (backendUrl) {
    // backendUrl like "https://xxxx.ngrok-free.app" → "wss://xxxx.ngrok-free.app/ws/audio"
    const base = backendUrl.replace(/^http/, "ws").replace(/\/$/, "");
    return `${base}/ws/audio`;
  }
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws/audio`;
}

export default function MicRecorder({ backendUrl }) {
  const [recording, setRecording] = useState(false);
  const [status, setStatus] = useState("idle"); // idle | connecting | recording | error
  const [level, setLevel] = useState(0); // audio level 0-1 for visual feedback

  const wsRef = useRef(null);
  const audioCtxRef = useRef(null);
  const workletNodeRef = useRef(null);
  const streamRef = useRef(null);
  const analyserRef = useRef(null);
  const levelAnimRef = useRef(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => stopRecording();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const stopRecording = useCallback(() => {
    // Stop level animation
    if (levelAnimRef.current) {
      cancelAnimationFrame(levelAnimRef.current);
      levelAnimRef.current = null;
    }

    // Stop worklet
    if (workletNodeRef.current) {
      workletNodeRef.current.port.postMessage("stop");
      workletNodeRef.current.disconnect();
      workletNodeRef.current = null;
    }

    // Stop media stream
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }

    // Close audio context
    if (audioCtxRef.current && audioCtxRef.current.state !== "closed") {
      audioCtxRef.current.close().catch(() => {});
      audioCtxRef.current = null;
    }

    // Close WebSocket
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setRecording(false);
    setStatus("idle");
    setLevel(0);
  }, []);

  const startRecording = useCallback(async () => {
    try {
      setStatus("connecting");

      // 1. Get microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: { ideal: TARGET_SAMPLE_RATE },
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      streamRef.current = stream;

      // 2. Create AudioContext
      const audioCtx = new AudioContext({ sampleRate: undefined }); // browser default rate
      audioCtxRef.current = audioCtx;

      // 3. Load AudioWorklet
      await audioCtx.audioWorklet.addModule("/pcm-processor.js");

      // 4. Create worklet node
      const workletNode = new AudioWorkletNode(audioCtx, "pcm-processor");
      workletNodeRef.current = workletNode;

      // 5. Connect source → worklet
      const source = audioCtx.createMediaStreamSource(stream);
      source.connect(workletNode);
      // Don't connect worklet to destination (we don't want playback)

      // 6. Analyser for visual level
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      analyserRef.current = analyser;

      // 7. Open WebSocket to backend
      const ws = new WebSocket(getAudioWsUrl(backendUrl));
      wsRef.current = ws;

      ws.onopen = () => {
        // Send configuration
        ws.send(JSON.stringify({ sampleRate: TARGET_SAMPLE_RATE }));
        setStatus("recording");
        setRecording(true);
      };

      ws.onclose = () => {
        if (recording) {
          setStatus("error");
          stopRecording();
        }
      };

      ws.onerror = (err) => {
        console.error("[MicRecorder] WebSocket error:", err);
        setStatus("error");
      };

      // 8. Worklet → WebSocket: stream PCM
      workletNode.port.onmessage = (e) => {
        const pcm = e.data; // Float32Array
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(pcm.buffer);
        }
      };

      // 9. Audio level animation
      const updateLevel = () => {
        if (!analyserRef.current) return;
        const buf = new Uint8Array(analyserRef.current.frequencyBinCount);
        analyserRef.current.getByteTimeDomainData(buf);
        let sum = 0;
        for (let i = 0; i < buf.length; i++) {
          const x = (buf[i] - 128) / 128;
          sum += x * x;
        }
        setLevel(Math.sqrt(sum / buf.length));
        levelAnimRef.current = requestAnimationFrame(updateLevel);
      };
      updateLevel();
    } catch (err) {
      console.error("[MicRecorder] Start failed:", err);
      setStatus("error");
      stopRecording();
    }
  }, [backendUrl, recording, stopRecording]);

  const toggle = useCallback(() => {
    if (recording) {
      stopRecording();
    } else {
      startRecording();
    }
  }, [recording, startRecording, stopRecording]);

  // Status colours
  const statusColor = {
    idle: "text-gray-400",
    connecting: "text-yellow-400 animate-pulse",
    recording: "text-red-400",
    error: "text-red-500",
  }[status];

  const buttonLabel = recording ? "Stop Mic" : "Start Mic";
  const buttonClass = recording
    ? "bg-red-600 hover:bg-red-700"
    : "bg-green-600 hover:bg-green-700";

  return (
    <div className="flex items-center gap-3">
      <button
        onClick={toggle}
        className={`${buttonClass} text-white text-sm font-semibold px-3 py-1.5 rounded
                   transition-colors flex items-center gap-2`}
        aria-label={buttonLabel}
      >
        {/* Mic icon */}
        <svg
          width="14" height="14" viewBox="0 0 24 24" fill="none"
          stroke="currentColor" strokeWidth="2" strokeLinecap="round"
        >
          <rect x="9" y="1" width="6" height="12" rx="3" />
          <path d="M5 10a7 7 0 0 0 14 0" />
          <line x1="12" y1="17" x2="12" y2="21" />
          <line x1="8" y1="21" x2="16" y2="21" />
        </svg>
        {buttonLabel}
      </button>

      {/* Audio level bar */}
      {recording && (
        <div className="w-24 h-2 bg-gray-700 rounded overflow-hidden" aria-label="Audio level">
          <div
            className="h-full bg-green-400 transition-all duration-75"
            style={{ width: `${Math.min(level * 300, 100)}%` }}
          />
        </div>
      )}

      {/* Status */}
      <span className={`text-xs font-mono ${statusColor}`}>
        {status === "idle" && "Mic off"}
        {status === "connecting" && "Connecting…"}
        {status === "recording" && "● Live"}
        {status === "error" && "Error — retry"}
      </span>
    </div>
  );
}
