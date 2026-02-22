/**
 * MachineIcon — SVG icon for a single machine in the operating room.
 *
 * Renders an icon shape based on the machine type, coloured by state
 * (ON=green, OFF=grey, STANDBY=yellow, CAUTION=red).
 *
 * Props:
 *   machine   — { id, name, category, ui: {icon, zone}, ... }
 *   isOn      — boolean  (is the machine currently ON)
 *   position  — { x, y } from layout config
 *   onClick   — (machine) => void  — opens override dialog
 */
import React from "react";

const STATE_COLORS = {
  on:      "#22c55e",
  off:     "#6b7280",
  standby: "#eab308",
  caution: "#ef4444",
};

/**
 * Tiny SVG paths for different machine types.
 * Each returns an element placed at (0,0) — the <g> transform positions it.
 */
const ICON_SHAPES = {
  monitor:      () => <rect x="-14" y="-10" width="28" height="20" rx="3" />,
  anesthesia:   () => <><rect x="-10" y="-12" width="20" height="18" rx="2" /><line x1="0" y1="6" x2="0" y2="14" strokeWidth="2" /></>,
  ventilator:   () => <><rect x="-10" y="-8" width="20" height="16" rx="2" /><path d="M-6 0 L0-4 L6 0 L0 4Z" fill="rgba(255,255,255,0.4)" /></>,
  carm:         () => <><path d="M-8-12 A12 12 0 0 1 8-12 L8 6 L-8 6Z" /><rect x="-4" y="6" width="8" height="6" /></>,
  pump:         () => <circle cx="0" cy="0" r="12" />,
  suction:      () => <><rect x="-8" y="-12" width="16" height="20" rx="2" /><line x1="0" y1="8" x2="0" y2="14" strokeWidth="2" /></>,
  lithotripter: () => <><rect x="-12" y="-8" width="24" height="16" rx="3" /><circle cx="0" cy="0" r="4" fill="rgba(255,255,255,0.5)" /></>,
  camera:       () => <><circle cx="0" cy="0" r="10" /><circle cx="0" cy="0" r="5" fill="rgba(255,255,255,0.4)" /></>,
  light:        () => <><polygon points="0,-14 8,0 -8,0" /><rect x="-6" y="0" width="12" height="4" /></>,
  esu:          () => <><rect x="-12" y="-8" width="24" height="16" rx="2" /><line x1="-6" y1="0" x2="6" y2="0" strokeWidth="2" stroke="rgba(255,255,255,0.6)" /></>,
  argon:        () => <><ellipse cx="0" cy="0" rx="10" ry="12" /><text y="4" textAnchor="middle" fontSize="8" fill="rgba(255,255,255,0.6)">Ar</text></>,
  ultrasound:   () => <><rect x="-12" y="-8" width="24" height="16" rx="2" /><path d="M-4 4 Q0-6 4 4" fill="none" strokeWidth="1.5" stroke="rgba(255,255,255,0.5)" /></>,
  gas:          () => <><rect x="-8" y="-14" width="16" height="24" rx="4" /><ellipse cx="0" cy="-6" rx="5" ry="3" fill="rgba(255,255,255,0.3)" /></>,
  drain:        () => <><rect x="-6" y="-14" width="12" height="22" rx="2" /><line x1="0" y1="8" x2="0" y2="14" strokeWidth="2" /></>,
};

function getShape(iconName) {
  return ICON_SHAPES[iconName] || ICON_SHAPES.monitor;
}

export default function MachineIcon({ machine, isOn, position, onClick }) {
  if (!machine || !position) return null;

  const color = isOn ? STATE_COLORS.on : STATE_COLORS.off;
  const iconName = machine.ui?.icon || "monitor";
  const ShapeFn = getShape(iconName);
  const label = `${machine.id || ""} ${machine.name || ""}`.trim();

  return (
    <g
      className="machine-group"
      transform={`translate(${position.x}, ${position.y})`}
      onClick={() => onClick && onClick(machine)}
      role="button"
      tabIndex={0}
      aria-label={`${label} — ${isOn ? "ON" : "OFF"}`}
      onKeyDown={(e) => {
        if ((e.key === "Enter" || e.key === " ") && onClick) {
          e.preventDefault();
          onClick(machine);
        }
      }}
      style={{ cursor: "pointer" }}
    >
      {/* Background glow */}
      <circle cx="0" cy="0" r="18" fill={color} opacity="0.15" />

      {/* Icon shape */}
      <g fill={color} stroke={color} strokeWidth="1">
        <ShapeFn />
      </g>

      {/* Label */}
      <text
        y="24"
        textAnchor="middle"
        fill="#e2e8f0"
        fontSize="9"
        fontFamily="monospace"
        pointerEvents="none"
      >
        {machine.id}
      </text>
    </g>
  );
}
