import { useEffect, useRef, useState, useCallback } from 'react';

export type Policy = {
  infection_radius: number;
  movement_speed: number;
  message?: string;
};

export function useSimulationBridge(url: string = 'ws://localhost:8000/ws') {
  const ws = useRef<WebSocket | null>(null);
  const [currentPolicy, setCurrentPolicy] = useState<Policy | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isMacroThinking, setIsMacroThinking] = useState(false);

  useEffect(() => {
    ws.current = new WebSocket(url);

    ws.current.onopen = () => setIsConnected(true);
    ws.current.onclose = () => setIsConnected(false);

    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'POLICY_UPDATE') {
        setCurrentPolicy(data.payload);
        setIsMacroThinking(false); // Shachi has finished
      }
    };

    return () => {
      ws.current?.close();
    };
  }, [url]);

  const sendAggregateStats = useCallback((stats: any) => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      setIsMacroThinking(true); // Pause the micro simulation
      ws.current.send(JSON.stringify(stats));
    }
  }, [currentPolicy]);

  return { isConnected, isMacroThinking, currentPolicy, sendAggregateStats };
}
