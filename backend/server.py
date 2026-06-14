import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from macro_agent import ShachiEnvironment

app = FastAPI(title="abm.gl Macro Engine")

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[WebSocket, ShachiEnvironment] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[websocket] = ShachiEnvironment()

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            del self.active_connections[websocket]

    async def broadcast_policy(self, websocket: WebSocket, policy: dict):
        await websocket.send_json({"type": "POLICY_UPDATE", "payload": policy})

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            aggregate_stats = json.loads(data)
            
            env = manager.active_connections[websocket]
            policy = await env.step(aggregate_stats)
            
            await manager.broadcast_policy(websocket, policy.model_dump())
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
