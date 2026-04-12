from models import Reward


def evaluate_action(task: dict, action, state: dict) -> Reward:
    score   = 0.0
    penalty = 0.0
    expected = task["expected"]
    text     = (action.content or "").lower()
    step     = state["step_count"]

    # ──────────────────────────────────────────────
    # 🟢 EASY — Classification + Priority Response
    # Max score: 1.0
    #   +0.5 correct classify
    #   +0.5 appropriate response (refund / sorry)
    # ──────────────────────────────────────────────
    if "priority" in expected:

        if action.action_type == "classify":
            if expected["classification"] in text:
                score += 0.5

        elif action.action_type == "respond":
            # check response contains resolution keywords
            if any(kw in text for kw in ["refund", "sorry", "apologize", "charged", "reimburs"]):
                score += 0.4
            # bonus: mentions urgency / priority
            if any(kw in text for kw in ["urgent", "priority", "immediately", "right away"]):
                score += 0.1

    # ──────────────────────────────────────────────
    # 🟡 MEDIUM — Sentiment-Aware Policy Resolution
    # Max score: 1.0
    #   +0.3 correct classify
    #   +0.3 empathetic language
    #   +0.4 policy-compliant response
    # ──────────────────────────────────────────────
    elif "sentiment" in expected:

        if action.action_type == "classify":
            if expected["classification"] in text:
                score += 0.3

        elif action.action_type == "respond":
            # empathy check
            if any(kw in text for kw in ["sorry", "understand", "apologize", "frustrat"]):
                score += 0.3

            # policy compliance based on classification
            if expected["classification"] == "technical":
                if any(kw in text for kw in ["investigate", "fix", "restart", "working", "resolve", "issue"]):
                    score += 0.4
            elif expected["classification"] == "billing":
                if any(kw in text for kw in ["refund", "charged", "reimburs", "process"]):
                    score += 0.4

    # ──────────────────────────────────────────────
    # 🔴 HARD — Multi-Turn Memory-Based Resolution
    # Max score: 1.0
    #   +0.2  step 1 correct classify
    #   +0.2  step 2 ask for info
    #   +0.1  efficiency bonus (done in ≤3 steps)
    #   +0.3  final response with correct keywords
    #   +0.1  memory bonus (used asked_info context)
    #   +0.1  correct escalation decision
    #   −0.2  unnecessary escalation
    #   −0.3  respond at step 2 without having asked first
    #   −0.2  repeated action
    # ──────────────────────────────────────────────
    elif "needs_info" in expected:

        # Step 1 — classify
        if step == 1:
            if action.action_type == "classify":
                if expected["classification"] in text:
                    score += 0.2

        # Step 2 — ask for missing info
        elif step == 2:
            if action.action_type == "ask":
                score += 0.2
                state["asked_info"] = True
            elif action.action_type == "respond":
                # penalize responding without asking first
                if not state.get("asked_info"):
                    penalty += 0.3

        # Step 3+ — final resolution
        elif step >= 3:
            if action.action_type == "respond":
                # correct keywords in final response
                if any(kw in text for kw in expected.get("final_keywords", [])):
                    score += 0.3
                # memory bonus — agent used info gathered earlier
                if state.get("asked_info"):
                    score += 0.1

            elif action.action_type == "escalate":
                if expected.get("decision") == "escalate":
                    score += 0.1   # correct escalation
                else:
                    penalty += 0.2  # unnecessary escalation

        # efficiency bonus — resolved within 3 steps
        if step <= 3 and action.action_type in ("respond", "escalate"):
            score += 0.1

    # ──────────────────────────────────────────────
    # Global: repeated action_type penalty
    # ──────────────────────────────────────────────
    if state["history"]:
        last_actions = [h.get("action_type", "") for h in state["history"]]
        if last_actions and last_actions[-1] == action.action_type:
            penalty += 0.2

    final_score = round(max(min(score - penalty, 0.99), 0.01), 4)

    return Reward(
        score=final_score,
        feedback=(
            f"step={step} action={action.action_type} "
            f"score={score:.2f} penalty={penalty:.2f} final={final_score:.4f}"
        )
    )