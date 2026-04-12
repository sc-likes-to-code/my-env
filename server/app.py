from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager
from typing import Optional
import traceback

from server.your_environment import SupportEnv
from models import Action

# ── App lifespan (replaces deprecated @app.on_event) ────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: pre-initialize environment
    app.state.env = SupportEnv()
    app.state.env.reset("easy")
    yield
    # shutdown: cleanup
    try:
        app.state.env.close()
    except Exception:
        pass


app = FastAPI(
    title="OpenEnv Customer Support Environment",
    description="Multi-step customer support ticket resolution environment for AI agent evaluation.",
    version="1.0.0",
    lifespan=lifespan,
)

VALID_TASKS = {"easy", "medium", "hard"}


# ── Health check (judges ping this to verify Space is live) ──────────────────
@app.get("/")
@app.get("/health")
def health():
    return {"status": "ok", "message": "OpenEnv Support Environment Running"}


# ── reset ────────────────────────────────────────────────────────────────────
@app.post("/reset")
def reset(
    task: str = Query(default="easy", description="Task difficulty: easy | medium | hard")
):
    if task not in VALID_TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task '{task}'. Must be one of: {sorted(VALID_TASKS)}"
        )
    try:
        obs = app.state.env.reset(task)
        return obs.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"reset() failed: {str(e)}")


# ── step ─────────────────────────────────────────────────────────────────────
@app.post("/step")
def step(action: dict):
    # auto-recover if env was never reset
    if app.state.env.state_data is None:
        app.state.env.reset("easy")

    try:
        action_obj = Action(**action)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid action payload: {str(e)}"
        )

    try:
        obs, reward, done, info = app.state.env.step(action_obj)
        return {
            "observation": obs.model_dump(),
            "reward":      reward.score,
            "feedback":    reward.feedback,
            "done":        done,
            "info":        info,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"step() failed: {str(e)}\n{traceback.format_exc()}"
        )


# ── state ────────────────────────────────────────────────────────────────────
@app.get("/state")
def state():
    try:
        return app.state.env.state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"state() failed: {str(e)}")