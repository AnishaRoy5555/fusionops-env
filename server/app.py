"""
FusionOps OpenEnv Server
FastAPI application exposing the scheduling environment via HTTP and WebSocket.
"""

from __future__ import annotations

import json
import uuid
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from src.models import Graph
from src.environment import FusionOpsEnv, parse_action
from src.tasks import load_task, get_task_config, list_tasks

app = FastAPI(title="FusionOps", description="ML Graph Scheduling Environment")

# Store active sessions
sessions: dict[str, FusionOpsEnv] = {}


class ResetRequest(BaseModel):
    task: str = "task1_linear"


class StepRequest(BaseModel):
    command: str


class ResetResponse(BaseModel):
    session_id: str
    observation: str
    done: bool = False
    reward: float = 0.0


class StepResponse(BaseModel):
    observation: str
    reward: float
    done: bool
    info: dict = {}
    score: Optional[float] = None


# ============================================================
# HTTP Endpoints
# ============================================================

@app.get("/")
async def root():
    return {"status": "ok", "environment": "fusionops", "tasks": list_tasks()}


@app.post("/reset")
async def reset(request: ResetRequest = ResetRequest()):
    task_name = request.task
    try:
        graph = load_task(task_name)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    cfg = get_task_config(task_name)
    env = FusionOpsEnv(graph, max_steps=cfg["max_steps"])
    result = env.reset()

    session_id = str(uuid.uuid4())
    sessions[session_id] = env

    return ResetResponse(
        session_id=session_id,
        observation=result.observation,
        done=result.done,
        reward=result.reward,
    )


@app.post("/step/{session_id}")
async def step(session_id: str, request: StepRequest):
    if session_id not in sessions:
        return JSONResponse(status_code=404, content={"error": "Session not found"})

    env = sessions[session_id]
    action = parse_action(request.command, env.graph)

    if action is None:
        return StepResponse(
            observation="Failed to parse action. Use format: SCHEDULE ops=[0,1] config=[128,128,1] retain=[]",
            reward=-0.1,
            done=False,
            info={"error": "Parse error"},
        )

    result = env.step(action)

    response = StepResponse(
        observation=result.observation,
        reward=result.reward,
        done=result.done,
        info=result.info,
    )

    if result.done:
        response.score = env.get_score()
        # Clean up session
        del sessions[session_id]

    return response


@app.get("/state/{session_id}")
async def get_state(session_id: str):
    if session_id not in sessions:
        return JSONResponse(status_code=404, content={"error": "Session not found"})
    return sessions[session_id].get_state()


@app.get("/tasks")
async def get_tasks():
    result = {}
    for name in list_tasks():
        cfg = get_task_config(name)
        result[name] = cfg
    return result


@app.get("/web")
async def web_ui():
    """Simple web UI for testing."""
    return HTMLResponse("""
    <html>
    <head><title>FusionOps Environment</title></head>
    <body>
        <h1>FusionOps - ML Graph Scheduling Environment</h1>
        <p>Use the API endpoints:</p>
        <ul>
            <li>POST /reset - Start a new episode</li>
            <li>POST /step/{session_id} - Take an action</li>
            <li>GET /state/{session_id} - Get current state</li>
            <li>GET /tasks - List available tasks</li>
        </ul>
    </body>
    </html>
    """)


# ============================================================
# WebSocket Endpoint (for OpenEnv compatibility)
# ============================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    env: Optional[FusionOpsEnv] = None

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "")

            if msg_type == "reset":
                task_name = data.get("task", "task1_linear")
                try:
                    graph = load_task(task_name)
                except ValueError as e:
                    await websocket.send_json({"error": str(e)})
                    continue

                cfg = get_task_config(task_name)
                env = FusionOpsEnv(graph, max_steps=cfg["max_steps"])
                result = env.reset()

                await websocket.send_json({
                    "type": "reset_result",
                    "observation": result.observation,
                    "done": result.done,
                    "reward": result.reward,
                })

            elif msg_type == "step":
                if env is None:
                    await websocket.send_json({"error": "Must reset first"})
                    continue

                command = data.get("command", "")
                action = parse_action(command, env.graph)

                if action is None:
                    await websocket.send_json({
                        "type": "step_result",
                        "observation": "Parse error. Use: SCHEDULE ops=[0,1] config=[128,128,1] retain=[]",
                        "reward": -0.1,
                        "done": False,
                        "info": {"error": "Parse error"},
                    })
                    continue

                result = env.step(action)
                response = {
                    "type": "step_result",
                    "observation": result.observation,
                    "reward": result.reward,
                    "done": result.done,
                    "info": result.info,
                }

                if result.done:
                    response["score"] = env.get_score()

                await websocket.send_json(response)

            elif msg_type == "state":
                if env is None:
                    await websocket.send_json({"error": "Must reset first"})
                    continue
                await websocket.send_json({
                    "type": "state_result",
                    **env.get_state(),
                })

            elif msg_type == "close":
                break

            else:
                await websocket.send_json({"error": f"Unknown message type: {msg_type}"})

    except WebSocketDisconnect:
        pass


def main():
    """Entry point for running the server via 'server' script or python -m."""
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
