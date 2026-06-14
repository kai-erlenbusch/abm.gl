import asyncio
import os
import json
from pydantic import BaseModel
from dotenv import load_dotenv
from litellm import acompletion

load_dotenv()

# Define the exact JSON schema bridging the Frontend and Backend
class GridCell(BaseModel):
    density: int
    average_speed: float

class AggregateStats(BaseModel):
    active_agents: int
    average_speed: float
    tick: float
    system_status: str
    grid: list[list[GridCell]] | None = None

class PolicyResponse(BaseModel):
    infection_radius: float
    policy_speed_map: list[list[float]]
    message: str

class ShachiEnvironment:
    def __init__(self):
        self.current_state = None
        
    async def step(self, stats: dict) -> PolicyResponse:
        """
        The core Shachi gym loop using actual LLM inference.
        """
        # Strictly validate incoming GPU data
        parsed_stats = AggregateStats(**stats)
        print(f"[Shachi Env] Processing tick {parsed_stats.tick} for {parsed_stats.active_agents} agents...")
        
        # Analyze spatial grid to find hotspots
        spatial_context = ""
        if parsed_stats.grid:
            max_density = 0
            hot_sector = (0,0)
            hot_speed = 0.0
            for r, row in enumerate(parsed_stats.grid):
                for c, cell in enumerate(row):
                    if cell.density > max_density:
                        max_density = cell.density
                        hot_sector = (r, c)
                        hot_speed = cell.average_speed
            spatial_context = f"\n        Spatial Context: The highest concentration of agents ({max_density} agents) is currently localized in Sector {hot_sector} with local speed {hot_speed:.4f}."

        # Format the system prompt and user data
        prompt = f"""
        You are the 'Mayor' Macro Agent for a {parsed_stats.active_agents} agent simulation.
        Current Simulation State:
        - Active Agents: {parsed_stats.active_agents}
        - Global Average Speed: {parsed_stats.average_speed:.4f}{spatial_context}
        
        You must output a 10x10 `policy_speed_map` (list of 10 lists, each containing 10 floats) representing the speed limit for each of the 100 spatial sectors.
        If a sector is the hotspot, you may want to lower the speed there (e.g. 0.05) to quarantine them, while letting other sectors move faster (e.g. 0.8).
        Provide a brief message explaining your decision.
        """

        print(f"[Shachi Env] Sending data to LLM...")
        
        try:
            # Check if API key is set
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key or api_key == "sk-proj-YOUR_API_KEY_HERE":
                print("[Shachi Env] OPENAI_API_KEY missing. Falling back to mock response.")
                await asyncio.sleep(2)
                
                speed_map = [[0.8 for _ in range(10)] for _ in range(10)]
                if parsed_stats.grid:
                    speed_map[hot_sector[0]][hot_sector[1]] = 0.05
                    
                return PolicyResponse(
                    infection_radius=10.0,
                    policy_speed_map=speed_map,
                    message="Fallback: OPENAI_API_KEY not set. Applied local quarantine to hotspot."
                )

            # LiteLLM structured outputs
            # Passing the exact Pydantic class forces the model to adhere
            response = await acompletion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format=PolicyResponse,
            )
            
            content = response.choices[0].message.content
            print(f"[Shachi Env] LLM Response JSON: {content}")
            
            # Parse the JSON string back into our Pydantic model
            return PolicyResponse.model_validate_json(content)
            
        except Exception as e:
            print(f"[Shachi Env] Error during LLM inference: {e}")
            speed_map = [[0.1 for _ in range(10)] for _ in range(10)]
            return PolicyResponse(
                infection_radius=10.0,
                policy_speed_map=speed_map,
                message=f"Error: {str(e)}"
            )


