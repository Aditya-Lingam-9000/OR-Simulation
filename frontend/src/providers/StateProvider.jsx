/**
 * StateProvider — WebSocket connection to OR-Symphony backend.
 *
 * Provides surgery state + connection status via React Context.
 * Auto-reconnects on disconnect with exponential backoff.
 *
 * Supports remote backend URL via:
 *   - `backendUrl` prop (resolved from ?backend= query param or env var)
 *   - `setBackendUrl()` exposed in context for runtime URL changes
 */
import React, { createContext, useContext, useEffect, useRef, useState, useCallback } from "react";

const StateContext = createContext(null);

const INITIAL_STATE = {
  metadata: { surgery: "", phase: "Phase1", timestamp: "", reasoning: "normal" },
  machines: { "0": [], "1": [] },
  details: {},
  suggestions: [],
  confidence: 0.0,
  source: "rule",
};

/**
 * Build WebSocket URL for state.
 * @param {string|null} backendUrl
 */
function buildWsUrl(backendUrl) {
  if (backendUrl) {
    const base = backendUrl.replace(/^http/, "ws").replace(/\/$/, "");
    return `${base}/ws/state`;
  }
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws/state`;
}

const RECONNECT_BASE_MS = 1000;
const RECONNECT_MAX_MS = 16000;

export function StateProvider({ children, backendUrl: initialBackendUrl = null }) {
  const [state, setState] = useState(INITIAL_STATE);
  const [connected, setConnected] = useState(false);
  const [reconnectCount, setReconnectCount] = useState(0);
  const [backendUrl, setBackendUrlState] = useState(initialBackendUrl);

  const wsRef = useRef(null);
  const reconnectTimer = useRef(null);
  const mountedRef = useRef(true);
  // Always-fresh ref so callbacks never have stale closures
  const backendUrlRef = useRef(backendUrl);
  backendUrlRef.current = backendUrl;

  /** Allow runtime URL changes (e.g., from a connection input bar) */
  const setBackendUrl = useCallback((url) => {
    const cleaned = url ? url.replace(/\/$/, "") : null;
    setBackendUrlState(cleaned);
  }, []);

  const connect = useCallback(() => {
    if (!mountedRef.current) return;
    const url = backendUrlRef.current; // always fresh

    const doConnect = () => {
      try {
        const wsUrl = buildWsUrl(url);
        console.log("[OR-Symphony] Connecting WebSocket:", wsUrl);
        const ws = new WebSocket(wsUrl);
        wsRef.current = ws;

        ws.onopen = () => {
          if (!mountedRef.current) return;
          setConnected(true);
          setReconnectCount(0);
          console.log("[OR-Symphony] WebSocket connected:", wsUrl);
        };

        ws.onmessage = (event) => {
          if (!mountedRef.current) return;
          try {
            const data = JSON.parse(event.data);
            setState(data);
          } catch (err) {
            console.error("[OR-Symphony] Invalid JSON:", err);
          }
        };

        ws.onclose = () => {
          if (!mountedRef.current) return;
          setConnected(false);
          wsRef.current = null;
          scheduleReconnect();
        };

        ws.onerror = (err) => {
          console.error("[OR-Symphony] WebSocket error:", err);
          ws.close();
        };
      } catch (err) {
        console.error("[OR-Symphony] WebSocket creation failed:", err);
        scheduleReconnect();
      }
    };

    // ngrok free tier: hit HTTP endpoint with bypass header first
    if (url && url.includes("ngrok")) {
      const healthUrl = `${url.replace(/\/$/, "")}/health`;
      console.log("[OR-Symphony] Warming ngrok via", healthUrl);
      fetch(healthUrl, { headers: { "ngrok-skip-browser-warning": "true" } })
        .then(() => doConnect())
        .catch(() => doConnect());
    } else {
      doConnect();
    }
  }, []);     // deps intentionally empty — uses ref for backendUrl

  const scheduleReconnect = useCallback(() => {
    if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
    setReconnectCount((prev) => {
      const next = prev + 1;
      const delay = Math.min(RECONNECT_BASE_MS * Math.pow(2, next - 1), RECONNECT_MAX_MS);
      console.log(`[OR-Symphony] Reconnect #${next} in ${delay}ms`);
      reconnectTimer.current = setTimeout(() => {
        if (mountedRef.current) connect();
      }, delay);
      return next;
    });
  }, [connect]);

  // When backendUrl changes (initial mount OR runtime change), reconnect
  useEffect(() => {
    if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
    if (wsRef.current) {
      wsRef.current.onclose = null;
      wsRef.current.close();
      wsRef.current = null;
    }
    setConnected(false);
    setReconnectCount(0);
    connect();
  }, [backendUrl]); // eslint-disable-line react-hooks/exhaustive-deps

  // Cleanup on unmount
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.close();
      }
    };
  }, []);

  const value = { state, connected, reconnectCount, backendUrl, setBackendUrl };

  return <StateContext.Provider value={value}>{children}</StateContext.Provider>;
}

/**
 * Hook to consume the surgery state context.
 * @returns {{ state, connected, reconnectCount, backendUrl, setBackendUrl }}
 */
export function useORState() {
  const ctx = useContext(StateContext);
  if (!ctx) throw new Error("useORState must be used within <StateProvider>");
  return ctx;
}

/**
 * Build the REST API base URL for fetch calls.
 * Returns "" (empty, relative) for local proxied dev, or the full ngrok URL for remote.
 */
export function useApiBase() {
  const { backendUrl } = useORState();
  return backendUrl ? backendUrl.replace(/\/$/, "") : "";
}

/**
 * Build fetch options that include ngrok-skip-browser-warning header.
 * Use: fetch(url, ngrokFetchOpts())
 */
export function ngrokFetchOpts(extra = {}) {
  return {
    ...extra,
    headers: {
      ...(extra.headers || {}),
      "ngrok-skip-browser-warning": "true",
    },
  };
}
