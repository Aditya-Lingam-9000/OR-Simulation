/**
 * OverrideDialog — modal for manual machine state override.
 *
 * Opens when a machine icon is clicked.  The user picks an action
 * (ON / OFF / STANDBY), types a reason, and submits to POST /override.
 *
 * Props:
 *   machine  — { id, name, category, ... } or null (closed)
 *   isOn     — current state (for display)
 *   onClose  — () => void
 */
import React, { useState } from "react";
import { useApiBase, ngrokFetchOpts } from "../providers/StateProvider";

const ACTIONS = ["ON", "OFF", "STANDBY"];

export default function OverrideDialog({ machine, isOn, onClose }) {
  const [action, setAction] = useState("ON");
  const [reason, setReason] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState(null);
  const apiBase = useApiBase();

  if (!machine) return null;

  const handleSubmit = async (e) => {
    e.preventDefault();
    setSubmitting(true);
    setResult(null);
    try {
      const res = await fetch(`${apiBase}/override`, ngrokFetchOpts({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          machine_id: machine.id,
          action,
          reason: reason.trim() || "Manual override",
        }),
      }));
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `HTTP ${res.status}`);
      }
      const data = await res.json();
      setResult({ ok: true, msg: `${data.machine_id} → ${data.action}` });
      setTimeout(() => onClose(), 1200);
    } catch (err) {
      setResult({ ok: false, msg: err.message });
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div
      className="dialog-overlay"
      role="dialog"
      aria-modal="true"
      aria-label={`Override ${machine.name}`}
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div className="dialog-content">
        <h2 className="text-lg font-bold text-white mb-1">
          Override: {machine.name}
        </h2>
        <p className="text-xs text-gray-400 mb-3 font-mono">
          {machine.id} · {machine.category} · Currently{" "}
          <span className={isOn ? "text-green-400" : "text-gray-400"}>
            {isOn ? "ON" : "OFF"}
          </span>
        </p>

        <form onSubmit={handleSubmit} className="space-y-3">
          {/* Action selector */}
          <fieldset className="flex gap-2">
            <legend className="text-sm text-gray-300 mb-1">Action</legend>
            {ACTIONS.map((a) => (
              <label
                key={a}
                className={`px-3 py-1 rounded text-sm cursor-pointer border transition-colors
                  ${action === a
                    ? "bg-or-highlight text-white border-or-highlight"
                    : "bg-or-accent text-gray-300 border-gray-600 hover:border-gray-400"
                  }`}
              >
                <input
                  type="radio"
                  name="action"
                  value={a}
                  checked={action === a}
                  onChange={() => setAction(a)}
                  className="sr-only"
                />
                {a}
              </label>
            ))}
          </fieldset>

          {/* Reason */}
          <div>
            <label htmlFor="override-reason" className="text-sm text-gray-300">
              Reason
            </label>
            <input
              id="override-reason"
              type="text"
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              placeholder="Manual override"
              className="w-full mt-1 px-2 py-1 rounded bg-or-bg border border-gray-600
                         text-gray-200 text-sm focus:outline-none focus:ring-2 focus:ring-or-highlight"
            />
          </div>

          <div className="flex justify-between items-center pt-1">
            <button
              type="button"
              onClick={onClose}
              className="text-sm text-gray-400 hover:text-white transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={submitting}
              className="px-4 py-1 rounded bg-or-highlight text-white text-sm font-semibold
                         hover:bg-red-600 transition-colors disabled:opacity-50"
            >
              {submitting ? "Sending…" : "Apply Override"}
            </button>
          </div>

          {/* Result */}
          {result && (
            <p
              className={`text-xs mt-1 ${
                result.ok ? "text-green-400" : "text-red-400"
              }`}
            >
              {result.msg}
            </p>
          )}
        </form>
      </div>
    </div>
  );
}
