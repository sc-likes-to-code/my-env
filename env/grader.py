def evaluate_action(task, action, state):
    score = 0.0
    penalty = 0.0
    expected = task["expected"]

    text = (action.content or "").lower()

    # 🟢 EASY
    if "priority" in expected:
        if action.action_type == "classify":
            if expected["classification"] in text:
                score += 0.5
            if expected["priority"] in text:
                score += 0.5

    # 🟡 MEDIUM
    elif "sentiment" in expected:
        if action.action_type == "classify":
            if expected["classification"] in text:
                score += 0.3

        if action.action_type == "respond":
            if "sorry" in text or "understand" in text:
                score += 0.3

            if "restart" in text or "steps" in text:
                score += 0.3

    # 🔴 HARD
    elif "needs_info" in expected:
        step = state["step_count"]

        if step == 1 and action.action_type == "classify":
            if expected["classification"] in text:
                score += 0.2

        elif step == 2 and action.action_type == "ask":
            score += 0.2

        elif step >= 3:
            if action.action_type == "respond":
                if any(k in text for k in expected["final_keywords"]):
                    score += 0.3

            if action.action_type == "escalate":
                if expected["decision"] == "escalate":
                    score += 0.2

        # memory usage bonus
        if len(state["history"]) > 1:
            score += 0.1

    # PENALTIES
    if action.action_type == "escalate" and expected.get("decision") != "escalate":
        penalty -= 0.1

    # repeated action penalty
    if state["history"] and text == (state["history"][-1]["agent"]).lower():
        penalty -= 0.2

    final_score = max(min(score + penalty, 1.0), 0.0)

    return type("RewardObj", (), {
        "score": final_score,
        "feedback": f"score={final_score}"
    })()