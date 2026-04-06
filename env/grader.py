def evaluate_action(task, action, state):
    score = 0.0
    expected = task["expected"]

    if action.action_type == "classify":
        if action.content == expected.get("classification"):
            score += 0.4

    if action.action_type == "respond":
        if action.content and "refund" in action.content.lower():
            score += 0.4

    if action.action_type == "escalate":
        if expected.get("escalate"):
            score += 0.2

    return type("RewardObj", (), {
        "score": min(score, 1.0),
        "feedback": f"Score: {score}"
    })()