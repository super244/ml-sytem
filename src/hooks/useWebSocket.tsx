import { useState, useEffect, useRef, useCallback, createContext, useContext } from 'react';

interface WebSocketState {
  gpuTelemetry: any | null;
  jobProgress: any | null;
  agentDecision: any | null;
  clusterStatus: any | null;
  logLine: any | null;
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
          const msg = JSON.parse(event.data);
          setState(prev => {
            switch (msg.type) {
              case 'gpu_telemetry':
                return { ...prev, gpuTelemetry: msg };
              case 'job_update':
                return { ...prev, jobProgress: msg };
              case 'job_complete':
              case 'job_failed':
                return { ...prev, jobProgress: msg };
              case 'agent_decision':
                return { ...prev, agentDecision: msg };
              case 'cluster_update':
                return { ...prev, clusterStatus: msg };
              case 'log_line':
                return { ...prev, logLine: msg };
              case 'automl_update':
                return prev;
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
