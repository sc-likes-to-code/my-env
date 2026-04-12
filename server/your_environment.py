from server.tasks import load_task
from server.grader import evaluate_action
from models import Observation, Ticket, Reward


# Max steps per difficulty level
TASK_MAX_STEPS = {
    "easy":   6,
    "medium": 6,
    "hard":   8,
}


class SupportEnv:
    def __init__(self):
        self.state_data    = None
        self.current_task  = None
        self.task_level    = "easy"

    # ── reset ────────────────────────────────────────────────────────────────
    def reset(self, task: str = "easy") -> Observation:
        self.task_level   = task
        self.current_task = load_task(task)
        self.state_data   = {
            "history":    [],
            "step_count": 0,
            "asked_info": False,
            "max_steps":  TASK_MAX_STEPS.get(task, 8),
            "task_level": task,
        }
        return self._get_observation()

    # ── step ─────────────────────────────────────────────────────────────────
    def step(self, action):
        # safety: auto-init if called before reset
        if self.state_data is None or self.current_task is None:
            self.reset(self.task_level)

        self.state_data["step_count"] += 1
        step = self.state_data["step_count"]

        # grade the action
        reward: Reward = evaluate_action(self.current_task, action, self.state_data)

        # update memory flags
        if action.action_type == "ask":
            self.state_data["asked_info"] = True

        # store full history entry (action_type needed by grader repeat-check)
        self.state_data["history"].append({
            "user":        self.current_task["tickets"][0]["text"],
            "agent":       action.content or "",
            "action_type": action.action_type,
        })

        done = self._is_done(action, reward, step)

        return self._get_observation(), reward, done, {}

    # ── state ────────────────────────────────────────────────────────────────
    def state(self) -> dict:
        if self.state_data is None:
            return {"status": "not_initialized"}
        return {
            "task_level":  self.state_data.get("task_level"),
            "step_count":  self.state_data.get("step_count"),
            "max_steps":   self.state_data.get("max_steps"),
            "asked_info":  self.state_data.get("asked_info"),
            "history_len": len(self.state_data.get("history", [])),
            "history":     self.state_data.get("history", []),
        }

    # ── close ────────────────────────────────────────────────────────────────
    def close(self):
        """Cleanup hook — called by inference.py after episode ends."""
        self.state_data   = None
        self.current_task = None

    # ── internal helpers ─────────────────────────────────────────────────────
    def _is_done(self, action, reward: Reward, step: int) -> bool:
        max_steps = self.state_data.get("max_steps", 8)
        expected  = self.current_task.get("expected", {})

        # always end if max steps reached
        if step >= max_steps:
            return True

        # always end on very high reward (near-perfect episode)
        if reward.score >= 0.95:
            return True

        # EASY / MEDIUM: done after a valid respond action
        if self.task_level in ("easy", "medium"):
            if action.action_type == "respond" and reward.score > 0.0:
                return True

        # HARD: done only after respond/escalate AND agent has asked for info
        elif self.task_level == "hard":
            needs_info = expected.get("needs_info", False)
            if action.action_type in ("respond", "escalate"):
                if not needs_info or self.state_data.get("asked_info"):
                    return True

        return False

    def _get_observation(self) -> Observation:
        tickets = [Ticket(**t) for t in self.current_task["tickets"]]
        return Observation(
            tickets=tickets,
            current_ticket_id=tickets[0].id if tickets else None,
            history=self.state_data["history"],
        )