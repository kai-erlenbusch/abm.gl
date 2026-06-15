import asyncio
import os
import json
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv
from litellm import acompletion
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

# Define the exact JSON schema bridging the Frontend and Backend
class GridCell(BaseModel):
    density: int
    average_speed: float
    infected_count: int

class AggregateStats(BaseModel):
    active_agents: int
    average_speed: float
    tick: float
    system_status: str
    grid: list[list[GridCell]] | None = None

class Intervention(BaseModel):
    row: int
    col: int
    target_speed: float

class AdvisorResponse(BaseModel):
    proposed_global_speed: float
    proposed_interventions: list[Intervention]
    rationale: str

class LLMPolicyResponse(BaseModel):
    global_baseline_speed: float
    interventions: list[Intervention]
    message: str

class FrontendPolicyPayload(BaseModel):
    infection_radius: float
    policy_speed_map: list[list[float]]
    message: str

class ShachiEnvironment:
    def __init__(self):
        self.current_state = None
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True)
    async def get_advisor_proposal(self, role: str, prompt: str) -> AdvisorResponse | None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "sk-proj-YOUR_API_KEY_HERE":
            return None
        
        response = await acompletion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format=AdvisorResponse,
        )
        content = response.choices[0].message.content
        print(f"[{role}] Proposal JSON: {content}")
        return AdvisorResponse.model_validate_json(content)

    async def safe_get_advisor_proposal(self, role: str, prompt: str) -> AdvisorResponse | None:
        try:
            return await self.get_advisor_proposal(role, prompt)
        except ValidationError as e:
            print(f"[{role}] Pydantic validation failed: {e}")
            return None
        except Exception as e:
            print(f"[{role}] Failed after retries: {e}")
            return None

    async def step(self, stats: dict) -> FrontendPolicyPayload:
        """
        The core Shachi gym loop using actual LLM inference.
        """
        # Strictly validate incoming GPU data
        parsed_stats = AggregateStats(**stats)
        print(f"[Shachi Env] Processing tick {parsed_stats.tick} for {parsed_stats.active_agents} agents...")
        
        # Analyze spatial grid to find hotspots and build ASCII heatmap
        spatial_context = ""
        hot_sector = (0,0)
        if parsed_stats.grid:
            max_density = 0
            max_infected = 0
            hot_speed = 0.0
            
            for r, row in enumerate(parsed_stats.grid):
                for c, cell in enumerate(row):
                    if cell.density > max_density:
                        max_density = cell.density
                    if cell.infected_count > max_infected:
                        max_infected = cell.infected_count
                        hot_sector = (r, c)
                        hot_speed = cell.average_speed
            
            heatmap_str = "No agents present."
            infection_str = "No infections present."
            if max_density > 0:
                heatmap_rows = []
                infection_rows = []
                for row in parsed_stats.grid:
                    heatmap_chars = []
                    infection_chars = []
                    for cell in row:
                        val = int((cell.density / max_density) * 9)
                        heatmap_chars.append(str(val))
                        
                        inf_val = int((cell.infected_count / max_infected) * 9) if max_infected > 0 else 0
                        infection_chars.append(str(inf_val))
                        
                    heatmap_rows.append("".join(heatmap_chars))
                    infection_rows.append("".join(infection_chars))
                    
                heatmap_str = "\n".join(heatmap_rows)
                infection_str = "\n".join(infection_rows)
            
            spatial_context = f"\n        Spatial Context: The highest concentration of INFECTED agents ({max_infected} infected) is currently localized in Sector {hot_sector} with local speed {hot_speed:.4f}.\n        Population Density Heatmap (0-9 scale):\n        " + heatmap_str.replace("\n", "\n        ") + f"\n        Infection Hotspot Heatmap (0-9 scale):\n        " + infection_str.replace("\n", "\n        ")

        health_prompt = f"""
        You are the 'Public Health Advisor' for a {parsed_stats.active_agents} agent simulation.
        Current Simulation State:
        - Active Agents: {parsed_stats.active_agents}
        - Global Average Speed: {parsed_stats.average_speed:.4f}{spatial_context}
        
        Your goal is to MINIMIZE pathogen spread. Identify high-infection hotspots and recommend strict quarantine interventions.
        To impose a lockdown in Sector (X, Y), set its target_speed to 0.0.
        Provide your proposed global speed, interventions (max 5), and a rationale.
        """

        econ_prompt = f"""
        You are the 'Chief Economist Advisor' for a {parsed_stats.active_agents} agent simulation.
        Current Simulation State:
        - Active Agents: {parsed_stats.active_agents}
        - Global Average Speed: {parsed_stats.average_speed:.4f}{spatial_context}
        
        Your goal is to MAXIMIZE economic throughput. Recommend high-speed highways. 
        To keep the economy flowing in safe sectors, set target_speed to 0.8.
        Provide your proposed global speed, interventions (max 5), and a rationale.
        """

        print(f"[Shachi Env] Dispatching concurrent requests to Advisors...")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "sk-proj-YOUR_API_KEY_HERE":
            print("[Shachi Env] OPENAI_API_KEY missing. Falling back to mock response.")
            await asyncio.sleep(2)
            
            speed_map = [[0.8 for _ in range(10)] for _ in range(10)]
            if parsed_stats.grid:
                speed_map[hot_sector[0]][hot_sector[1]] = 0.05
                
            return FrontendPolicyPayload(
                infection_radius=10.0,
                policy_speed_map=speed_map,
                message="Fallback: OPENAI_API_KEY not set. Applied local quarantine to hotspot."
            )

        # Run Advisors Concurrently
        health_proposal, econ_proposal = await asyncio.gather(
            self.safe_get_advisor_proposal("Health Advisor", health_prompt),
            self.safe_get_advisor_proposal("Economist Advisor", econ_prompt)
        )

        mayor_prompt = f"""
        You are the 'Mayor' (Governor Synthesizer) for a {parsed_stats.active_agents} agent simulation.
        Current Simulation State:
        - Active Agents: {parsed_stats.active_agents}
        - Global Average Speed: {parsed_stats.average_speed:.4f}{spatial_context}
        
        You have received the following proposals from your advisors:
        
        [Public Health Advisor]:
        {health_proposal.model_dump_json(indent=2) if health_proposal else "No response"}
        
        [Chief Economist Advisor]:
        {econ_proposal.model_dump_json(indent=2) if econ_proposal else "No response"}
        
        You must synthesize these competing priorities. You may choose to favor one, or strike a balance.
        To impose a lockdown in an infected Sector (X, Y), set its target_speed to 0.0. To keep the economy flowing in safe sectors, set target_speed to 0.8.
        Set a `global_baseline_speed` for the entire 10x10 grid, and a list of `interventions` (at most 5).
        Provide a brief message explaining your final decision.
        """

        print(f"[Shachi Env] Sending synthesized data to Mayor...")

        try:
            response = await acompletion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": mayor_prompt}],
                response_format=LLMPolicyResponse,
            )
            
            content = response.choices[0].message.content
            print(f"[Mayor] LLM Response JSON: {content}")
            
            llm_policy = LLMPolicyResponse.model_validate_json(content)
            
            # Reconstruct the 10x10 map
            speed_map = [[llm_policy.global_baseline_speed for _ in range(10)] for _ in range(10)]
            for inv in llm_policy.interventions:
                if 0 <= inv.row < 10 and 0 <= inv.col < 10:
                    speed_map[inv.row][inv.col] = inv.target_speed
                    
            return FrontendPolicyPayload(
                infection_radius=10.0,
                policy_speed_map=speed_map,
                message=llm_policy.message
            )
            
        except ValidationError as e:
            print(f"[Shachi Env] Pydantic validation failed during Mayor inference: {e}")
            speed_map = [[0.1 for _ in range(10)] for _ in range(10)]
            return FrontendPolicyPayload(
                infection_radius=10.0,
                policy_speed_map=speed_map,
                message=f"Validation Error: {str(e)}"
            )
        except Exception as e:
            print(f"[Shachi Env] Error during Mayor inference: {e}")
            speed_map = [[0.1 for _ in range(10)] for _ in range(10)]
            return FrontendPolicyPayload(
                infection_radius=10.0,
                policy_speed_map=speed_map,
                message=f"Error: {str(e)}"
            )
