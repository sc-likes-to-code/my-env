import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from env.environment import SupportEnv
from env.models import Action

IMAGE_NAME = os.getenv("IMAGE_NAME") # If you are using docker image 
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("MY_ENV_V4_TASK", "hard")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "support_env")
MAX_STEPS = 8
TEMPERATURE = 0.7
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.1  # normalized score in [0, 1]

# Max possible reward per step is 1.0 in the local environment.
MAX_TOTAL_REWARD = MAX_STEPS

SYSTEM_PROMPT = """
You are an AI customer support agent solving a multi-step task.

You must act in SEQUENCE across steps:

Step 1 → classify
Step 2 → respond
Step 3 → escalate (only if necessary)

Return STRICT JSON:
{
  "action_type": "classify/respond/escalate/ask",
  "content": "string or null"
}

Rules:
- classification must be EXACT: "billing" or "technical"
- responses should include keywords like "refund" for billing issues
- escalate ONLY if issue is critical (e.g. money deducted, urgent)

Be consistent across steps.
Do NOT skip steps.
Do NOT repeat the same action multiple times.
"""


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(step: int, ticket_text: str, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        You are at step {step}.
        Follow sequence:
        1. First classify
        2. Then respond
        3. Then escalate if needed

        Current ticket: {ticket_text}
        Last reward: {last_reward:.2f}
        Previous steps:
        {history_block}
        Return the next action as JSON.
        """
    ).strip()


def _fallback_action(step: int, ticket_text: str, ticket_id: int) -> Action:
    lowered_ticket = ticket_text.lower()

    if step == 1:
        classification = "billing" if any(word in lowered_ticket for word in ["charged", "payment", "refund", "money", "subscription", "deducted"]) else "technical"
        return Action(action_type="classify", ticket_id=ticket_id, content=classification)

    if any(word in lowered_ticket for word in ["refund", "charged", "payment", "deducted"]):
        if step == 2:
            return Action(action_type="respond", ticket_id=ticket_id, content="Please issue a refund.")
        return Action(action_type="escalate", ticket_id=ticket_id, content=None)

    if any(word in lowered_ticket for word in ["crash", "settings", "error", "bug"]):
        if step == 2:
            return Action(action_type="respond", ticket_id=ticket_id, content="We are investigating the issue.")
        return Action(action_type="escalate", ticket_id=ticket_id, content=None)

    if step == 2:
        return Action(action_type="respond", ticket_id=ticket_id, content="Thanks for the report.")

    return Action(action_type="escalate", ticket_id=ticket_id, content=None)


def get_model_action(client: Optional[OpenAI], step: int, ticket_text: str, ticket_id: int, last_reward: float, history: List[str]) -> Action:
    if client is None:
        return _fallback_action(step, ticket_text, ticket_id)

    step_instruction = ""

    if step == 1:
        step_instruction = "Perform classification."
    elif step == 2:
        step_instruction = "If information is missing, ask a clarifying question. Otherwise respond."
    else:
        step_instruction = "Decide whether to escalate."

    user_prompt = build_user_prompt(step, ticket_text, last_reward, history)
    user_prompt = f"{step_instruction}\n\n{user_prompt}"
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        if not text:
            return _fallback_action(step, ticket_text, ticket_id)

        parsed = json.loads(text)
        action_type = str(parsed.get("action_type", "")).strip() or "classify"
        content = parsed.get("content")
        if content is not None:
            content = str(content)
        return Action(action_type=action_type, ticket_id=ticket_id, content=content)
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return _fallback_action(step, ticket_text, ticket_id)


def _extract_current_ticket(observation: dict) -> tuple[int, str]:
    tickets = observation.get("tickets") or []
    if not tickets:
        return 0, ""

    current_ticket = tickets[0]
    ticket_id = int(current_ticket.get("id") or observation.get("current_ticket_id") or 0)
    ticket_text = str(current_ticket.get("text") or "")

    return ticket_id, ticket_text


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_KEY else None

    env = SupportEnv()
    observation = env.reset(TASK_NAME).model_dump()

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        ticket_id, ticket_text = _extract_current_ticket(observation)
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            action = get_model_action(client, step, ticket_text, ticket_id, last_reward, history)

            observation, reward, done, _ = env.step(action)
            observation = observation.model_dump()
            ticket_id, ticket_text = _extract_current_ticket(observation)

            error = None

            reward_val = float(reward.score or 0.0)
            rewards.append(reward_val)
            steps_taken = step
            last_reward = reward_val

            action_str = json.dumps(action.model_dump(), separators=(",", ":"))
            log_step(step=step, action=action_str, reward=reward_val, done=done, error=error)

            history.append(f"Step {step}: {action.model_dump()} -> reward {reward_val:+.2f}")

            if done:
                break

        score = sum(rewards)
        score = min(score, 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            if hasattr(env, "close"):
                env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)

        log_end(success, steps_taken, score, rewards)

if __name__ == "__main__":
    main()