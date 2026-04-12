---
title: OpenEnv Customer Support Environment
emoji: 🎫
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - reinforcement-learning
  - customer-support
  - multi-step
  - nlp
  - real-world
  - agent-evaluation
---

# 🎫 Support Ticket Resolution Environment (OpenEnv)

A real-world, multi-step customer support simulation environment built on the **OpenEnv** framework. Designed to train and evaluate AI agents on tasks that mirror genuine human support workflows.

---

## 🌍 Overview

Unlike toy RL environments, this system simulates a realistic customer support pipeline where an agent must:

- Classify support tickets by issue type
- Detect sentiment and apply policy-aware responses
- Ask clarifying questions when information is missing
- Use conversation memory across turns
- Decide whether to resolve or escalate issues

This makes it a high-value benchmark for evaluating **multi-step reasoning**, **policy compliance**, and **stateful decision-making** in AI agents.

---

## 🏗️ Project Structure

```
my-env/
├── inference.py          # Baseline agent — runs all 3 tasks
├── models.py             # Pydantic models: Action, Observation, Reward
├── openenv.yaml          # OpenEnv spec metadata (name, tasks, spaces, rewards)
├── pyproject.toml        # Project metadata and dependencies
├── requirements.txt      # Python dependencies
├── uv.lock               # Locked dependency versions
├── Dockerfile            # Container definition
├── client.py             # HTTP client for the environment
├── __init__.py           # Root package
└── server/
    ├── __init__.py       # Server package
    ├── app.py            # FastAPI server (reset / step / state / health)
    ├── grader.py         # Task graders with reward shaping
    ├── tasks.py          # Task definitions (easy / medium / hard)
    └── your_environment.py  # Core SupportEnv class
```

---

## 📜 OpenEnv Spec (`openenv.yaml`)

This environment is fully compliant with the OpenEnv specification:

```yaml
name: openenv-customer-support-env
version: "1.0.0"
tags: [openenv]
entrypoint: server.app:app
tasks:
  - id: easy   | difficulty: easy   | max_steps: 6
  - id: medium | difficulty: medium | max_steps: 6
  - id: hard   | difficulty: hard   | max_steps: 8
```

Validated via:
```bash
openenv validate
```

---

## ⚙️ Core API

```python
reset(task: str) -> Observation        # Initialize episode for given task
step(action: Action) -> (Observation, Reward, done, info)  # Take one action
state() -> dict                        # Inspect full current episode state
```

### HTTP Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset?task=easy` | Start a new episode |
| `POST` | `/step` | Submit an action |
| `GET`  | `/state` | Get current episode state |
| `GET`  | `/health` | Health check (returns 200) |

---

## 📊 Observation Space

```json
{
  "tickets": [
    {"id": 1, "text": "Customer message here"}
  ],
  "current_ticket_id": 1,
  "history": [
    {"user": "...", "agent": "...", "action_type": "classify"}
  ]
}
```

---

## 🎮 Action Space

```python
Action(
    action_type: str,       # "classify" | "respond" | "escalate" | "ask"
    ticket_id: int,         # ID of the ticket being handled
    content: Optional[str]  # Classification label or response text
)
```

| Action | Purpose |
|--------|---------|
| `classify` | Categorize the issue: `billing` or `technical` |
| `respond` | Provide a resolution or acknowledgment |
| `ask` | Request missing information from the user |
| `escalate` | Hand off to a human agent |

---

## 📋 Tasks

### 🟢 Easy — Ticket Classification & Response
**Max steps:** 6

Agent must classify the ticket and provide an appropriate response.

| Step | Action | Reward |
|------|--------|--------|
| 1 | Correct classification (`billing`/`technical`) | +0.5 |
| 2 | Response with refund/sorry keywords | +0.4 |
| 2 | Urgency keyword bonus (urgent/immediately) | +0.1 |

**Max achievable:** 1.0

---

### 🟡 Medium — Sentiment-Aware Policy Resolution
**Max steps:** 6

Agent must detect issue type, show empathy, and generate a policy-compliant response.

| Step | Action | Reward |
|------|--------|--------|
| 1 | Correct classification | +0.3 |
| 2 | Empathetic language (sorry/understand/apologize) | +0.3 |
| 2 | Policy-compliant response (fix/refund keywords) | +0.4 |

**Max achievable:** 1.0

---

### 🔴 Hard — Multi-Turn Memory-Based Resolution
**Max steps:** 8

Agent must follow the full sequence: classify → ask → respond, using conversation memory.

| Step | Action | Reward |
|------|--------|--------|
| 1 | Correct classification | +0.2 |
| 2 | Ask for missing info (transaction ID etc.) | +0.2 |
| 3+ | Response with correct keywords | +0.3 |
| 3+ | Memory bonus (asked_info was used) | +0.1 |
| Any | Efficiency bonus (resolved in ≤3 steps) | +0.1 |
| Any | Correct escalation decision | +0.1 |

**Penalties:**

| Condition | Penalty |
|-----------|---------|
| Respond at step 2 without asking first | −0.3 |
| Unnecessary escalation | −0.2 |
| Repeated action type | −0.2 |

**Max achievable:** 1.0

---

## 🏆 Reward Design

This environment uses **dense reward shaping** — agents receive meaningful signal at every step, not just at episode end.

- Partial credit for each correct intermediate action
- Efficiency bonuses for faster resolution
- Memory bonuses for using context from prior turns
- Penalties for skipping required steps, repeating actions, or escalating unnecessarily
- Episode ends early on near-perfect score (≥0.95) or after max steps

---

## 📈 Baseline Scores

Achieved by the fallback rule-based agent (no LLM, no API key required):

| Task | Score | Success | Steps |
|------|-------|---------|-------|
| Easy | 0.500 | ✅ | 2 |
| Medium | 0.500 | ✅ | 2 |
| Hard | 0.300 | ✅ | 3 |
| **Aggregate** | **0.433** | | |

> A frontier LLM agent is expected to score significantly higher.

---

## 🚀 Setup Instructions

### Clone repository
```bash
git clone <repo-url>
cd my-env
```

### Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux/Mac
```

### Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | ✅ Yes | — | Your Hugging Face API token |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |

> If no API key is set, the agent runs in **fallback mode** using rule-based actions. All 3 tasks still complete successfully.

---

## ▶️ Run Inference

```bash
python inference.py
```

Runs all 3 tasks in sequence and outputs:

```
[START] task=easy env=support_env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={...} reward=0.50 done=false error=null
[STEP] step=2 action={...} reward=0.50 done=true error=null
[END] success=true steps=2 rewards=0.50,0.50
[SUMMARY] task=easy score=0.500 success=true steps=2
...
[AGGREGATE] tasks=3 avg_score=0.433
```

---

## 🐳 Docker Usage

```bash
docker build -t support-env .
docker run -p 7860:7860 support-env
```

Test endpoints:
```bash
curl -X POST http://localhost:7860/reset?task=easy
curl http://localhost:7860/health
```

---

## ✅ OpenEnv Validation

```bash
pip install openenv-core
openenv validate
```

---

## ☁️ Deployment

Deployed as a **Hugging Face Docker Space** — fully containerized, CPU-friendly, and responds within the 20-minute inference runtime limit.

---

## 🔮 Future Improvements

- Multi-ticket queue handling
- Memory persistence across episodes
- Advanced policy rule engine
- Human-in-the-loop simulation
- More task difficulty levels

---

## 📄 License

MIT
