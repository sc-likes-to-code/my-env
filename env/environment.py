from env.tasks import load_task
from env.grader import evaluate_action

class SupportEnv:
    def __init__(self):
        self.state_data = None
        self.current_task = None

    def reset(self, task="easy"):
        self.current_task = load_task(task)
        self.state_data = {"history": []}
        return self._get_observation()

    def step(self, action):
        reward = evaluate_action(self.current_task, action, self.state_data)
        self.state_data["history"].append(action.__dict__)

        done = reward.score >= 0.95 or len(self.state_data["history"]) > 5

        return self._get_observation(), reward, done, {}

    def state(self):
        return self.state_data

    def _get_observation(self):
        return {
            "tickets": self.current_task["tickets"],
            "current_ticket_id": self.current_task["tickets"][0]["id"]
        }