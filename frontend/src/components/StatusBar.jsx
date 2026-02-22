/**
 * StatusBar — bottom bar showing pipeline status.
 *
 * Displays: connection indicator, current phase, confidence %, source badge,
 * and any suggestions from the pipeline.
 *
 * Props:
 *   connected  — boolean (WebSocket connected)
 *   metadata   — { surgery, phase, timestamp, reasoning }
 *   confidence — 0.0–1.0
 *   source     — "rule" | "medgemma" | "rule+medgemma"
 *   suggestions— ["..."]
 */
import React from "react";

function sourceBadgeClass(source) {
  switch (source) {
    case "medgemma":       return "source-badge source-llm";
    case "rule+medgemma":  return "source-badge source-both";
    default:               return "source-badge source-rule";
  }
}

export default function StatusBar({ connected, metadata, confidence, source, suggestions }) {
  const phase = metadata?.phase || "—";
  const surgery = metadata?.surgery || "—";
  const reasoning = metadata?.reasoning || "";
  const pct = Math.round((confidence || 0) * 100);

  return (
    <div className="bg-or-accent border-t border-gray-700 px-4 py-2 flex flex-wrap items-center gap-4 text-xs font-mono">
      {/* Connection */}
      <span className="flex items-center gap-1">
        <span
          className={`inline-block w-2 h-2 rounded-full ${
            connected ? "bg-green-400" : "bg-red-500 animate-blink"
          }`}
        />
        {connected ? "Live" : "Offline"}
      </span>

      {/* Surgery + Phase */}
      <span className="text-gray-300">
        {surgery} · <span className="text-white">{phase}</span>
      </span>

      {/* Confidence */}
      <span className="text-gray-300">
        Confidence:{" "}
        <span
          className={`font-bold ${
            pct >= 80 ? "text-green-400" : pct >= 50 ? "text-yellow-400" : "text-red-400"
          }`}
        >
          {pct}%
        </span>
      </span>

      {/* Source badge */}
      <span className={sourceBadgeClass(source)}>{source || "rule"}</span>

      {/* Reasoning */}
      {reasoning && reasoning !== "normal" && (
        <span className="text-yellow-300 truncate max-w-[200px]" title={reasoning}>
          ⚡ {reasoning}
        </span>
      )}

      {/* Suggestions */}
      {suggestions && suggestions.length > 0 && (
        <span className="text-gray-400 truncate max-w-[300px]" title={suggestions.join("; ")}>
          💡 {suggestions[0]}
          {suggestions.length > 1 && ` (+${suggestions.length - 1})`}
        </span>
      )}
    </div>
  );
}
