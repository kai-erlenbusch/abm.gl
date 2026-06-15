import { create } from 'zustand';

interface SimulationState {
  isPaused: boolean;
  setIsPaused: (val: boolean) => void;
  lastLlmSend: number;
  setLastLlmSend: (time: number) => void;
  
  // Dynamic parameters mapped from UI
  dynamicParams: Record<string, number>;
  setDynamicParam: (key: string, value: number) => void;

  // Setup trigger
  setupTrigger: number;
  triggerSetup: () => void;
}

export const useSimulationStore = create<SimulationState>((set) => ({
  isPaused: false,
  setIsPaused: (val: boolean) => set({ isPaused: val }),
  lastLlmSend: 0,
  setLastLlmSend: (time: number) => set({ lastLlmSend: time }),
  
  dynamicParams: {
    infection_radius: 0.2,
    initial_infected: 100,
    transmission_probability: 0.0,
    recovery_time: 1.0,
  },
  setDynamicParam: (key: string, value: number) => 
    set((state) => ({ dynamicParams: { ...state.dynamicParams, [key]: value } })),
    
  setupTrigger: 0,
  triggerSetup: () => set((state) => ({ setupTrigger: state.setupTrigger + 1 })),
}));
