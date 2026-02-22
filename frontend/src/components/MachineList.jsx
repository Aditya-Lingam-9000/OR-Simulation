/**
 * MachineList — accessible text list fallback for all machines.
 *
 * Renders a simple table/list showing every machine, its state (ON/OFF),
 * category, and lets the user click to trigger the override dialog.
 * Acts as the screen-reader-friendly alternative to the SVG room.
 *
 * Props:
 *   machinesData  — { M01: { name, category, ui, ... }, ... }
 *   machineStates — { "0": [...], "1": [...] }
 *   onMachineClick— (machine) => void
 */
import React from "react";

export default function MachineList({ machinesData, machineStates, onMachineClick }) {
  if (!machinesData || Object.keys(machinesData).length === 0) {
    return <p className="text-gray-500 text-sm p-2">No machines loaded.</p>;
  }

  const onIds = new Set(machineStates?.["1"] || []);
  const sorted = Object.entries(machinesData).sort(([a], [b]) => a.localeCompare(b));

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs font-mono text-gray-300 border-collapse">
        <thead>
          <tr className="text-left border-b border-gray-700">
            <th className="px-2 py-1">ID</th>
            <th className="px-2 py-1">Machine</th>
            <th className="px-2 py-1">Category</th>
            <th className="px-2 py-1">State</th>
            <th className="px-2 py-1">Action</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map(([id, m]) => {
            const isOn = onIds.has(id);
            const enriched = { ...m, id };
            return (
              <tr
                key={id}
                className="border-b border-gray-800 hover:bg-or-accent/30 transition-colors"
              >
                <td className="px-2 py-1 text-gray-400">{id}</td>
                <td className="px-2 py-1">{m.name}</td>
                <td className="px-2 py-1 text-gray-400">{m.category}</td>
                <td className="px-2 py-1">
                  <span
                    className={`inline-block w-2 h-2 rounded-full mr-1 ${
                      isOn ? "bg-green-400" : "bg-gray-500"
                    }`}
                  />
                  {isOn ? "ON" : "OFF"}
                </td>
                <td className="px-2 py-1">
                  <button
                    onClick={() => onMachineClick && onMachineClick(enriched)}
                    className="text-or-highlight hover:underline"
                    aria-label={`Override ${m.name}`}
                  >
                    Override
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
