/**
 * SurgerySelector — dropdown to choose the active surgery type.
 *
 * Fetches available surgeries from GET /surgeries on mount,
 * then POSTs to /select_surgery when the user picks a new one.
 */
import React, { useEffect, useState } from "react";
import { useApiBase, ngrokFetchOpts } from "../providers/StateProvider";

export default function SurgerySelector({ currentSurgery, onSwitched }) {
  const [surgeries, setSurgeries] = useState([]);
  const [loading, setLoading] = useState(false);
  const apiBase = useApiBase();

  useEffect(() => {
    fetch(`${apiBase}/surgeries`, ngrokFetchOpts())
      .then((r) => r.json())
      .then((data) => setSurgeries(data))
      .catch((err) => console.error("[SurgerySelector] fetch error:", err));
  }, [apiBase]);

  const handleChange = async (e) => {
    const selected = e.target.value;
    if (selected === currentSurgery) return;
    setLoading(true);
    try {
      const res = await fetch(`${apiBase}/select_surgery`, ngrokFetchOpts({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ surgery: selected }),
      }));
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      onSwitched && onSwitched(selected);
    } catch (err) {
      console.error("[SurgerySelector] switch failed:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex items-center gap-2">
      <label htmlFor="surgery-select" className="text-sm text-gray-300 font-mono">
        Surgery:
      </label>
      <select
        id="surgery-select"
        value={currentSurgery}
        onChange={handleChange}
        disabled={loading}
        className="bg-or-accent text-gray-200 text-sm rounded px-2 py-1 border border-gray-600
                   focus:outline-none focus:ring-2 focus:ring-or-highlight disabled:opacity-50"
      >
        {surgeries.length === 0 && (
          <option value={currentSurgery}>{currentSurgery}</option>
        )}
        {surgeries.map((s) => (
          <option key={s} value={s}>
            {s}
          </option>
        ))}
      </select>
      {loading && <span className="text-xs text-yellow-400 animate-pulse">switching…</span>}
    </div>
  );
}
