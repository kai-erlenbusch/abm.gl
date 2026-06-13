import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from macro_agent import env

app = FastAPI(title="abm.gl Macro Engine")

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast_policy(self, policy: dict):
        for connection in self.active_connections:
            await connection.send_json({"type": "POLICY_UPDATE", "payload": policy})

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Receive GPU aggregate stats
            data = await websocket.receive_text()
            aggregate_stats = json.loads(data)
            
            # Shachi Lockstep Time execution
            # The frontend is currently paused waiting for this response
            policy = await env.step(aggregate_stats)
            
            # Broadcast the new policy down to the WebGPU compute shaders
            await manager.broadcast_policy(policy.model_dump())
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
