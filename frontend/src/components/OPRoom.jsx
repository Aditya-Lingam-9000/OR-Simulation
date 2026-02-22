/**
 * OPRoom — SVG operating room visualisation.
 *
 * Loads the layout config for the selected surgery and renders:
 *   - Room background with zone labels
 *   - Operating bed
 *   - Agent positions (dots with labels)
 *   - Machine icons at configured positions
 *
 * Props:
 *   layout        — layout JSON (room, zones, bed, agents, machines positions)
 *   machinesData  — machine definitions { M01: { name, ui, ... }, ... }
 *   machineStates — { "0": ["M02",...], "1": ["M01",...] }
 *   onMachineClick — (machine) => void
 */
import React from "react";
import MachineIcon from "./MachineIcon";

export default function OPRoom({ layout, machinesData, machineStates, onMachineClick }) {
  if (!layout || !layout.room) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-400">
        No layout loaded
      </div>
    );
  }

  const { room, zones = {}, bed, agents = [], machines: machinePositions = {} } = layout;
  const onIds = new Set(machineStates?.["1"] || []);

  return (
    <svg
      className="or-room-svg w-full h-auto"
      viewBox={`0 0 ${room.width} ${room.height}`}
      xmlns="http://www.w3.org/2000/svg"
      role="img"
      aria-label="Operating room layout"
    >
      {/* Room background */}
      <rect
        x="0" y="0"
        width={room.width} height={room.height}
        rx="16"
        fill={room.background || "#1e293b"}
      />

      {/* Zone labels */}
      {Object.entries(zones).map(([id, zone]) => (
        <text
          key={id}
          x={zone.x}
          y={zone.y}
          textAnchor="middle"
          fill="#475569"
          fontSize="11"
          fontFamily="monospace"
          pointerEvents="none"
        >
          {zone.label}
        </text>
      ))}

      {/* Operating bed / table */}
      {bed && (
        <g>
          <rect
            x={bed.x}
            y={bed.y}
            width={bed.width}
            height={bed.height}
            rx="12"
            fill="#334155"
            stroke="#475569"
            strokeWidth="2"
          />
          <text
            x={bed.x + bed.width / 2}
            y={bed.y + bed.height / 2 + 4}
            textAnchor="middle"
            fill="#64748b"
            fontSize="12"
            fontFamily="monospace"
          >
            Operating Table
          </text>
        </g>
      )}

      {/* Agents */}
      {agents.map((agent) => (
        <g key={agent.id} transform={`translate(${agent.x}, ${agent.y})`}>
          <circle cx="0" cy="0" r="8" fill={agent.color || "#3b82f6"} opacity="0.7" />
          <text
            y="-12"
            textAnchor="middle"
            fill="#cbd5e1"
            fontSize="9"
            fontFamily="monospace"
          >
            {agent.label}
          </text>
        </g>
      ))}

      {/* Machines */}
      {Object.entries(machinePositions).map(([machineId, pos]) => {
        const machineData = machinesData?.[machineId];
        if (!machineData) return null;
        const enriched = { ...machineData, id: machineId };
        return (
          <MachineIcon
            key={machineId}
            machine={enriched}
            isOn={onIds.has(machineId)}
            position={pos}
            onClick={onMachineClick}
          />
        );
      })}
    </svg>
  );
}
