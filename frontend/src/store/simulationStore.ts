import { create } from 'zustand';

interface SimulationState {
  isPaused: boolean;
  setIsPaused: (val: boolean) => void;
  lastLlmSend: number;
  setLastLlmSend: (time: number) => void;
}

export const useSimulationStore = create<SimulationState>((set) => ({
  isPaused: false,
  setIsPaused: (val: boolean) => set({ isPaused: val }),
  lastLlmSend: 0,
  setLastLlmSend: (time: number) => set({ lastLlmSend: time }),
}));
