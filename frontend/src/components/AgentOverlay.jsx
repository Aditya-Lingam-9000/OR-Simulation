/**
 * AgentOverlay — animating agent role indicators.
 *
 * Shows the current source-of-truth for each decision cycle:
 *   rule     → blue pulsing dot
 *   medgemma → purple confirmed check
 *   rule+medgemma → cyan dual badge
 *   degraded → red caution blink
 *
 * Props:
 *   source  — "rule" | "medgemma" | "rule+medgemma"
 *   agents  — [{ id, label, color }] from layout
 */
import React from "react";

const SOURCE_CONFIG = {
  rule:            { color: "#3b82f6", label: "Rule",       animation: "animate-pulse-slow" },
  medgemma:        { color: "#8b5cf6", label: "MedGemma",   animation: "" },
  "rule+medgemma": { color: "#06b6d4", label: "Rule+LLM",   animation: "" },
  degraded:        { color: "#ef4444", label: "Degraded",   animation: "animate-blink" },
};

export default function AgentOverlay({ source, agents }) {
  const cfg = SOURCE_CONFIG[source] || SOURCE_CONFIG.rule;

  return (
    <div className="flex items-center gap-3 py-1">
      {/* Source indicator dot */}
      <span
        className={`inline-block w-3 h-3 rounded-full ${cfg.animation}`}
        style={{ backgroundColor: cfg.color }}
        aria-hidden="true"
      />
      <span className="text-xs font-mono text-gray-300">
        Engine: <span style={{ color: cfg.color }}>{cfg.label}</span>
      </span>

      {/* Agent roster */}
      {agents && agents.length > 0 && (
        <span className="text-xs text-gray-500 ml-2">
          | {agents.map((a) => a.label).join(" · ")}
        </span>
      )}
    </div>
  );
}
