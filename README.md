# Support Ticket Resolution Environment (OpenEnv)

## Overview
This project implements a real-world customer support simulation environment using the OpenEnv framework.

It is designed to evaluate AI agents on multi-step decision-making tasks that closely resemble real human workflows.

Unlike toy environments, this system simulates:
- Ticket classification
- Information extraction
- Policy-aware responses
- Multi-turn resolution workflows

## Motivation
Customer support is a high-impact real-world domain where agents must:
- Understand ambiguous user queries
- Follow company policies
- Make multi-step decisions
- Balance speed and correctness

This environment provides a realistic benchmark for testing such capabilities.

## Environment Design

### Core API
```python
reset(task: str) -> Observation
step(action: Action) -> (Observation, Reward, done, info)
state() -> dict
```

### Observation Space
```json
{
  "tickets": [
    {"id": int, "text": str}
  ],
  "current_ticket_id": int
}
```

Represents the current support ticket context.

### Action Space
```python
Action(
  action_type: str,  # classify / respond / escalate / ask
  ticket_id: int,
  content: Optional[str]
)
```

## Action Types
- classify → categorize issue
- respond → provide resolution
- ask → request missing info
- escalate → hand off to human

## Tasks

### Easy — Classification + Priority
Agent must:
- Classify issue (billing / technical)
- Assign priority (low / medium / high)

Reward:
- 0.5 → classification
- 0.5 → priority

### Medium — Resolution + Policy Compliance
Agent must:
- Identify issue
- Detect sentiment
- Generate policy-compliant response

Includes:
- refund rules
- troubleshooting steps

Reward:
- 0.3 → extraction
- 0.3 → policy compliance
- 0.4 → response quality

### Hard — Multi-Turn Resolution (Memory-Based)
Agent must:
- Classify issue
- Ask for missing info
- Use conversation memory
- Decide: resolve / escalate
- Generate final response

Key features:
- Stateful multi-step interaction
- Session-based memory (avoid repetition)

Reward:
- 0.2 → classification
- 0.2 → appropriate question
- 0.2 → correct decision
- 0.3 → response quality
- 0.1 → efficiency

Bonuses & Penalties:
- +0.1 → correct memory usage
- −0.1 → unnecessary escalation
- −0.1 → irrelevant response
- −0.2 → repeated actions

## Reward Design
This environment uses dense reward shaping:
- Partial rewards for intermediate steps
- Encourages structured reasoning
- Penalizes inefficient or incorrect behavior

Ensures meaningful learning across the entire trajectory.

## Baseline Agent (inference.py)
The baseline agent:
- Uses OpenAI-compatible API
- Performs multi-step reasoning
- Outputs structured logs

### Output Format
```
[START] ...
[STEP] ...
[END] ...
```

## Features
- Step-aware prompting
- JSON action generation
- Fallback safety (no crashes)

## Setup Instructions

### Clone repository
```bash
git clone <repo-url>
cd support-env
```

### Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### Install dependencies
```bash
pip install -r requirements.txt
```

## Environment Variables
Set these before running:
```bash
API_BASE_URL=<your-endpoint>
MODEL_NAME=<model-name>
HF_TOKEN=<your-api-key>
```

## Run Inference
```bash
python inference.py
```

## Docker Usage
```bash
docker build -t support-env .
docker run support-env
```

## Deployment
Designed for Hugging Face Spaces (Docker):
- Fully containerized
- Lightweight (CPU-friendly)
- Fast runtime (<20 min)

## Evaluation Criteria
This environment is evaluated on:
- Real-world utility
- Task difficulty progression
- Reward design quality
- Agent consistency

## Key Highlights
- Real-world support workflow simulation
- Multi-step reasoning environment
- Policy-aware decision making
- Dense reward shaping
- Robust evaluation pipeline

## Future Improvements
- Multi-ticket queue handling
- Memory across conversations
- Advanced policy systems
- Human-in-the-loop simulation

## Conclusion
This environment bridges the gap between:
- Toy RL tasks
- Real-world AI agent evaluation

Making it highly valuable for training and benchmarking next-generation AI systems.