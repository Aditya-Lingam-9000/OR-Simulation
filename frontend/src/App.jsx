/**
 * App.jsx — Main OR-Symphony application shell.
 *
 * Assembles:
 *   Header       — title + SurgerySelector
 *   OPRoom       — SVG operating room (centre)
 *   AgentOverlay — source/engine indicator
 *   MachineList  — accessible text table
 *   StatusBar    — bottom bar
 *   OverrideDialog — modal (on machine click)
 */
import React, { useEffect, useState, useCallback } from "react";
import { StateProvider, useORState, useApiBase, ngrokFetchOpts } from "./providers/StateProvider";
import OPRoom from "./components/OPRoom";
import MachineList from "./components/MachineList";
import SurgerySelector from "./components/SurgerySelector";
import OverrideDialog from "./components/OverrideDialog";
import AgentOverlay from "./components/AgentOverlay";
import StatusBar from "./components/StatusBar";
import MicRecorder from "./components/MicRecorder";

// Layout JSON imports (static, bundled at build)
import pcnlLayout from "@configs/layouts/pcnl_layout.json";
import hepatectomyLayout from "@configs/layouts/partial_hepatectomy_layout.json";
import lobectomyLayout from "@configs/layouts/lobectomy_layout.json";

const LAYOUTS = {
  PCNL: pcnlLayout,
  "Partial Hepatectomy": hepatectomyLayout,
  Lobectomy: lobectomyLayout,
};

function AppContent() {
  const { state, connected, backendUrl, setBackendUrl } = useORState();
  const apiBase = useApiBase();
  const [machinesData, setMachinesData] = useState({});
  const [overrideMachine, setOverrideMachine] = useState(null);
  const [urlInput, setUrlInput] = useState("");

  const surgery = state.metadata?.surgery || "PCNL";
  const layout = LAYOUTS[surgery] || LAYOUTS.PCNL;

  // Fetch machines config when surgery changes
  useEffect(() => {
    fetch(`${apiBase}/machines`, ngrokFetchOpts())
      .then((r) => r.json())
      .then((data) => setMachinesData(data))
      .catch((err) => console.error("[App] machines fetch:", err));
  }, [surgery, apiBase]);

  const handleMachineClick = useCallback((machine) => {
    setOverrideMachine(machine);
  }, []);

  const onIds = new Set(state.machines?.["1"] || []);

  return (
    <div className="min-h-screen bg-or-bg flex flex-col">
      {/* ── Header ── */}
      <header className="bg-or-panel border-b border-gray-700 px-4 py-3 flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-3">
          <h1 className="text-lg font-bold text-white tracking-wide">
            OR-Symphony
          </h1>
          <span className="text-xs text-gray-500 font-mono hidden sm:inline">
            Predictive Surgical State Engine
          </span>
        </div>
        <div className="flex items-center gap-4 flex-wrap">
          <MicRecorder backendUrl={backendUrl} />
          <SurgerySelector
            currentSurgery={surgery}
            onSwitched={() => {
              /* state updates arrive via WebSocket */
            }}
          />
        </div>
      </header>

      {/* ── Connection Bar — shown when not connected and no backend URL ── */}
      {!connected && (
        <div className="bg-yellow-900/40 border-b border-yellow-700 px-4 py-2 flex flex-wrap items-center gap-2 text-xs font-mono">
          <span className="text-yellow-300">
            {backendUrl
              ? `⏳ Connecting to ${backendUrl} …`
              : "⚠ No backend URL — paste your ngrok URL below or add ?backend=URL to the address bar"}
          </span>
          {!backendUrl && (
            <form
              className="flex items-center gap-2"
              onSubmit={(e) => {
                e.preventDefault();
                if (urlInput.trim()) setBackendUrl(urlInput.trim());
              }}
            >
              <input
                type="text"
                value={urlInput}
                onChange={(e) => setUrlInput(e.target.value)}
                placeholder="https://xxxx.ngrok-free.app"
                className="bg-gray-800 text-white border border-gray-600 rounded px-2 py-1 text-xs w-72 font-mono"
              />
              <button
                type="submit"
                className="bg-green-600 hover:bg-green-500 text-white px-3 py-1 rounded text-xs font-bold"
              >
                Connect
              </button>
            </form>
          )}
        </div>
      )}

      {/* ── Main content ── */}
      <main className="flex-1 flex flex-col lg:flex-row gap-4 p-4 overflow-auto">
        {/* Left: SVG Room */}
        <section className="flex-1 min-w-0">
          <div className="bg-or-panel rounded-lg border border-gray-700 p-3">
            <AgentOverlay source={state.source} agents={layout?.agents} />
            <OPRoom
              layout={layout}
              machinesData={machinesData}
              machineStates={state.machines}
              onMachineClick={handleMachineClick}
            />
          </div>
        </section>

        {/* Right: Machine list + Details */}
        <aside className="w-full lg:w-80 flex-shrink-0 space-y-4">
          {/* Machine list */}
          <div className="bg-or-panel rounded-lg border border-gray-700 p-3">
            <h2 className="text-sm font-semibold text-gray-300 mb-2 font-mono">
              Machines — {surgery}
            </h2>
            <MachineList
              machinesData={machinesData}
              machineStates={state.machines}
              onMachineClick={handleMachineClick}
            />
          </div>

          {/* Suggestions */}
          {state.suggestions && state.suggestions.length > 0 && (
            <div className="bg-or-panel rounded-lg border border-gray-700 p-3">
              <h2 className="text-sm font-semibold text-gray-300 mb-2 font-mono">
                Suggestions
              </h2>
              <ul className="space-y-1">
                {state.suggestions.map((s, i) => (
                  <li key={i} className="text-xs text-gray-400">
                    💡 {s}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Details / Debug */}
          {state.details && Object.keys(state.details).length > 0 && (
            <details className="bg-or-panel rounded-lg border border-gray-700 p-3">
              <summary className="text-sm font-semibold text-gray-400 cursor-pointer font-mono">
                Raw Details
              </summary>
              <pre className="mt-2 text-xs text-gray-500 overflow-x-auto max-h-48 whitespace-pre-wrap">
                {JSON.stringify(state.details, null, 2)}
              </pre>
            </details>
          )}
        </aside>
      </main>

      {/* ── Status Bar ── */}
      <StatusBar
        connected={connected}
        metadata={state.metadata}
        confidence={state.confidence}
        source={state.source}
        suggestions={state.suggestions}
      />

      {/* ── Override Dialog ── */}
      <OverrideDialog
        machine={overrideMachine}
        isOn={overrideMachine ? onIds.has(overrideMachine.id) : false}
        onClose={() => setOverrideMachine(null)}
      />
    </div>
  );
}

/**
 * Resolve the backend URL from (in priority order):
 *   1. ?backend=... URL search parameter
 *   2. VITE_BACKEND_URL environment variable
 *   3. null (use the current host / Vite proxy)
 */
function resolveBackendUrl() {
  // URL param (e.g., http://localhost:3000/?backend=https://xxxx.ngrok-free.app)
  const params = new URLSearchParams(window.location.search);
  const fromParam = params.get("backend");
  if (fromParam) return fromParam.replace(/\/$/, "");

  // Vite env variable (set in .env or at build time)
  const fromEnv = import.meta.env.VITE_BACKEND_URL;
  if (fromEnv) return fromEnv.replace(/\/$/, "");

  return null; // local proxy
}

export default function App() {
  const backendUrl = resolveBackendUrl();
  return (
    <StateProvider backendUrl={backendUrl}>
      <AppContent />
    </StateProvider>
  );
}
