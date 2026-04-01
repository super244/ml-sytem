import { useState, useEffect, useRef, useCallback, createContext, useContext } from 'react';

export interface WsTelemetryMessage {
  type: 'gpu_telemetry' | 'job_progress' | 'agent_decision' | 'cluster_status' | 'log_line';
  data: unknown;
}

interface WebSocketState {
  gpuTelemetry: unknown | null;
  jobProgress: unknown | null;
  agentDecision: unknown | null;
  clusterStatus: unknown | null;
  logLine: unknown | null;
  isConnected: boolean;
}

const WebSocketContext = createContext<WebSocketState>({
  gpuTelemetry: null,
  jobProgress: null,
  agentDecision: null,
  clusterStatus: null,
  logLine: null,
  isConnected: false,
});

function getWsUrl(): string {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}/ws/telemetry`;
}

export function WebSocketProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<WebSocketState>({
    gpuTelemetry: null,
    jobProgress: null,
    agentDecision: null,
    clusterStatus: null,
    logLine: null,
    isConnected: false,
  });
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeout = useRef<ReturnType<typeof setTimeout>>();

  const connect = useCallback(() => {
    try {
      const ws = new WebSocket(getWsUrl());
      wsRef.current = ws;

      ws.onopen = () => {
        setState(prev => ({ ...prev, isConnected: true }));
      };

      ws.onmessage = (event) => {
        try {
          const msg: WsTelemetryMessage = JSON.parse(event.data);
          setState(prev => {
            switch (msg.type) {
              case 'gpu_telemetry':
                return { ...prev, gpuTelemetry: msg.data };
              case 'job_progress':
                return { ...prev, jobProgress: msg.data };
              case 'agent_decision':
                return { ...prev, agentDecision: msg.data };
              case 'cluster_status':
                return { ...prev, clusterStatus: msg.data };
              case 'log_line':
                return { ...prev, logLine: msg.data };
              default:
                return prev;
            }
          });
        } catch {
          // ignore non-JSON messages
        }
      };

      ws.onclose = () => {
        setState(prev => ({ ...prev, isConnected: false }));
        reconnectTimeout.current = setTimeout(connect, 3000);
      };

      ws.onerror = () => {
        ws.close();
      };
    } catch {
      reconnectTimeout.current = setTimeout(connect, 3000);
    }
  }, []);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimeout.current) clearTimeout(reconnectTimeout.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return (
    <WebSocketContext.Provider value={state}>
      {children}
    </WebSocketContext.Provider>
  );
}

export function useWebSocket(): WebSocketState {
  return useContext(WebSocketContext);
}
