from fastapi import FastAPI
from env.environment import SupportEnv
from env.models import Action

app = FastAPI()

env = SupportEnv()

@app.get("/")
def root():
    return {"message": "OpenEnv Support Environment Running"}

@app.post("/reset")
def reset(task: str = "easy"):
    obs = env.reset(task)
    return obs.model_dump()

@app.post("/step")
def step(action: dict):
    action_obj = Action(**action)
    obs, reward, done, info = env.step(action_obj)

    return {
        "observation": obs.model_dump(),
        "reward": reward.score,
        "done": done,
        "info": info
    }

@app.get("/state")
def state():
    return env.state()