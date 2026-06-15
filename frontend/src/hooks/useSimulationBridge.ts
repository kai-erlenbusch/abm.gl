import { useEffect, useRef, useState, useCallback } from 'react';
import { useSimulationStore } from '@/store/simulationStore';

export type Policy = {
  infection_radius: number;
  policy_speed_map?: number[][];
  message?: string;
};

export function useSimulationBridge(url: string = 'ws://localhost:8000/ws') {
  const ws = useRef<WebSocket | null>(null);
  const [currentPolicy, setCurrentPolicy] = useState<Policy | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const setIsMacroThinking = useSimulationStore(state => state.setIsMacroThinking);

  useEffect(() => {
    const token = process.env.NEXT_PUBLIC_SIMULATION_TOKEN || "dev_secret_token";
    let reconnectTimeoutId: any;
    let reconnectDelay = 1000;
    const maxDelay = 30000;
    let isMounted = true;

    function connect() {
      if (!isMounted) return;
      ws.current = new WebSocket(url);

      ws.current.onopen = () => {
        if (!isMounted) return;
        setIsConnected(true);
        reconnectDelay = 1000; // Reset delay on successful connection
        ws.current?.send(JSON.stringify({ auth_token: token }));
      };

      ws.current.onclose = () => {
        if (!isMounted) return;
        setIsConnected(false);
        setIsMacroThinking(false); // Make sure simulation isn't stuck paused
        
        // Exponential backoff
        clearTimeout(reconnectTimeoutId);
        reconnectTimeoutId = setTimeout(() => {
          reconnectDelay = Math.min(reconnectDelay * 2, maxDelay);
          connect();
        }, reconnectDelay);
      };

      ws.current.onmessage = (event) => {
        if (!isMounted) return;
        const data = JSON.parse(event.data);
        if (data.type === 'POLICY_UPDATE') {
          setCurrentPolicy(data.payload);
          setIsMacroThinking(false); // Shachi has finished
        }
      };
    }

    connect();

    return () => {
      isMounted = false;
      clearTimeout(reconnectTimeoutId);
      ws.current?.close();
    };
  }, [url]);

  const sendAggregateStats = useCallback((stats: any) => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      setIsMacroThinking(true); // Pause the micro simulation
      ws.current.send(JSON.stringify(stats));
    }
  }, [currentPolicy]);

  return { isConnected, currentPolicy, sendAggregateStats };
}
