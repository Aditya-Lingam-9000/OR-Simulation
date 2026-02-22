/**
 * StateProvider — WebSocket connection to OR-Symphony backend.
 *
 * Provides surgery state + connection status via React Context.
 * Auto-reconnects on disconnect with exponential backoff.
 *
 * Supports remote backend URL via `backendUrl` prop (e.g., ngrok URL for Kaggle).
 * If not set, uses the current browser host (with Vite proxy in dev mode).
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
 * @param {string|null} backendUrl — e.g. "https://xxxx.ngrok-free.app"
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

export function StateProvider({ children, backendUrl }) {
  const [state, setState] = useState(INITIAL_STATE);
  const [connected, setConnected] = useState(false);
  const [reconnectCount, setReconnectCount] = useState(0);
  const wsRef = useRef(null);
  const reconnectTimer = useRef(null);
  const mountedRef = useRef(true);

  const connect = useCallback(() => {
    if (!mountedRef.current) return;

    try {
      const ws = new WebSocket(buildWsUrl(backendUrl));
      wsRef.current = ws;

      ws.onopen = () => {
        if (!mountedRef.current) return;
        setConnected(true);
        setReconnectCount(0);
        console.log("[OR-Symphony] WebSocket connected");
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
        console.log("[OR-Symphony] WebSocket disconnected — scheduling reconnect");
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
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

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

  useEffect(() => {
    mountedRef.current = true;
    connect();
    return () => {
      mountedRef.current = false;
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      if (wsRef.current) {
        wsRef.current.onclose = null; // prevent reconnect loop
        wsRef.current.close();
      }
    };
  }, [connect]);

  const value = { state, connected, reconnectCount, backendUrl };

  return <StateContext.Provider value={value}>{children}</StateContext.Provider>;
}

/**
 * Hook to consume the surgery state context.
 * @returns {{ state: object, connected: boolean, reconnectCount: number, backendUrl: string|null }}
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
