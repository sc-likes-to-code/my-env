from tasks import load_task
from grader import evaluate_action
from models import Observation, Ticket

class SupportEnv:
    def __init__(self):
        self.state_data = None
        self.current_task = None

    def reset(self, task="easy"):
        self.current_task = load_task(task)
        self.state_data = {
            "history": [],
            "step_count": 0,
            "asked_info": False
        }
        return self._get_observation()

    def step(self, action):
        self.state_data["step_count"] += 1

        reward = evaluate_action(self.current_task, action, self.state_data)

        # 🔥 store interaction memory
        self.state_data["history"].append({
            "user": self.current_task["tickets"][0]["text"],
            "agent": action.content or ""
        })

        if action.action_type == "ask":
            self.state_data["asked_info"] = True

        done = (
            reward.score >= 0.9
            or action.action_type == "respond"
            or self.state_data["step_count"] >= 6
        )

        return self._get_observation(), reward, done, {}

    def state(self):
        return self.state_data

    def _get_observation(self):
        tickets = [Ticket(**t) for t in self.current_task["tickets"]]

        return Observation(
            tickets=tickets,
            current_ticket_id=tickets[0].id if tickets else None,
            history=self.state_data["history"]  # 🔥 include memory
        )