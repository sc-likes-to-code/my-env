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

        if action.action_type == "respond":
            if "refund" in text or "sorry" in text:
                score += 0.5

    # 🟡 MEDIUM
    elif "sentiment" in expected:
        if action.action_type == "classify":
            if expected["classification"] in text:
                score += 0.3

        if action.action_type == "respond":
            if expected["classification"] == "technical":
                if any(word in text for word in ["investigate", "fix", "issue", "working"]):
                    score += 0.4

            if "sorry" in text or "understand" in text:
                score += 0.2

    # 🔴 HARD
    elif "needs_info" in expected:
        step = state["step_count"]
        
        # efficiency bonus (faster resolution = better)
        if step <= 2:
            score += 0.1

        if step == 1 and action.action_type == "classify":
            if expected["classification"] in text:
                score += 0.2

        elif step == 2:
            if action.action_type == "ask":
                score += 0.2
            elif action.action_type == "respond":
                # allow shortcut resolution
                if any(k in text for k in expected["final_keywords"]):
                    score += 0.2

        if action.action_type == "respond":
            if any(k in text for k in expected["final_keywords"]):
                score += 0.3

        if action.action_type == "escalate":
            if expected["decision"] == "escalate":
                score += 0.2


    # PENALTIES
    if action.action_type == "escalate":
        if expected.get("decision") != "escalate":
            penalty -= 0.2

    # repeated action penalty
    if state["history"] and action.__dict__ == state["history"][-1]:
        penalty -= 0.2

    final_score = max(min(score + penalty, 1.0), 0.0)

    return type("RewardObj", (), {
        "score": final_score,
        "feedback": f"score={final_score}"
    })()