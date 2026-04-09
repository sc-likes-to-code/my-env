import requests

class SupportEnvClient:
    def __init__(self, base_url="http://localhost:7860"):
        self.base_url = base_url

    def reset(self, task="easy"):
        response = requests.post(f"{self.base_url}/reset", params={"task": task})
        return response.json()

    def step(self, action: dict):
        response = requests.post(f"{self.base_url}/step", json=action)
        return response.json()

    def state(self):
        response = requests.get(f"{self.base_url}/state")
        return response.json()