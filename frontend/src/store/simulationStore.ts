import { create } from 'zustand';

export interface ModelConfig {
  [key: string]: number;
}

interface SimulationState {
  isPaused: boolean;
  setIsPaused: (val: boolean) => void;
  
  // Generic parameters mapped from UI
  dynamicParams: ModelConfig;
  setDynamicParam: (key: string, value: number) => void;

  // Setup trigger
  setupTrigger: number;
  triggerSetup: () => void;
}

export const useSimulationStore = create<SimulationState>((set) => ({
  isPaused: false,
  setIsPaused: (val: boolean) => set({ isPaused: val }),
  
  dynamicParams: {
    agent_count: 100000,
    // Epidemic default params (to keep current canvas running during transition)
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
