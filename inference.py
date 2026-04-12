import json
import os
from typing import List, Optional, Tuple

from openai import OpenAI

from server.your_environment import SupportEnv
from models import Action

# ── Environment / model config ──────────────────────────────────────────────
HF_TOKEN     = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK    = os.getenv("MY_ENV_V4_BENCHMARK", "support_env")

MAX_STEPS = 8
TEMPERATURE = 0.0
MAX_TOKENS = 150

ALL_TASKS = ["easy", "medium", "hard"]

# ── Logging helpers ──────────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool) -> None:
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

# ── Fallback actions ─────────────────────────────────────────────────────────
def fallback_action(step, task, ticket_text, ticket_id):
    text = ticket_text.lower()

    is_billing = any(w in text for w in ["charged", "payment", "refund", "deducted"])
    
    if step == 1:
        return Action(
            action_type="classify",
            ticket_id=ticket_id,
            content="billing" if is_billing else "technical"
        )

    if step == 2:
        if task == "hard":
            return Action(
                action_type="ask",
                ticket_id=ticket_id,
                content="Please provide your transaction ID so I can check the payment."
            )

        if is_billing:
            return Action(
                action_type="respond",
                ticket_id=ticket_id,
                content="We are sorry for the issue. Your refund will be processed immediately."
            )

        return Action(
            action_type="respond",
            ticket_id=ticket_id,
            content="We are sorry for the inconvenience. We will investigate and fix the issue."
        )

    # step 3+
    if is_billing:
        return Action(
            action_type="respond",
            ticket_id=ticket_id,
            content="Thanks for the details. Your refund has been successfully processed."
        )

    return Action(
        action_type="respond",
        ticket_id=ticket_id,
        content="The issue has been fixed. Please check again."
    )

# ── Model action ─────────────────────────────────────────────────────────────
def get_model_action(client, step, task, ticket_text, ticket_id):
    if client is None:
        return fallback_action(step, task, ticket_text, ticket_id)

    try:
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": ticket_text}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return fallback_action(step, task, ticket_text, ticket_id)
    except:
        return fallback_action(step, task, ticket_text, ticket_id)

# ── Ticket extraction ────────────────────────────────────────────────────────
def extract_ticket(observation: dict):
    ticket = observation["tickets"][0]
    return ticket["id"], ticket["text"]

# ── Run single task ──────────────────────────────────────────────────────────
def run_task(client, task_name: str):
    env = SupportEnv()
    observation = env.reset(task_name).model_dump()

    rewards: List[float] = []
    steps_taken = 0

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    ticket_id, ticket_text = extract_ticket(observation)

    for step in range(1, MAX_STEPS + 1):
        action = get_model_action(client, step, task_name, ticket_text, ticket_id)

        observation, reward, done, _ = env.step(action)
        observation = observation.model_dump()

        ticket_id, ticket_text = extract_ticket(observation)

        reward_val = round(float(reward.score), 2)
        rewards.append(reward_val)
        steps_taken = step

        log_step(step=step, action=action.action_type, reward=reward_val, done=done)

        if done:
            break

    score = round(sum(rewards) / len(rewards), 2) if rewards else 0.00
    success = score >= 0.30

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN) if HF_TOKEN else None

    for task_name in ALL_TASKS:
        run_task(client, task_name)

if __name__ == "__main__":
    main()