def load_task(level):
    if level == "easy":
        return {
            "tickets": [
                {"id": 1, "text": "I was charged twice for my subscription."}
            ],
            "expected": {
                "classification": "billing",
                "priority": "high"
            }
        }

    elif level == "medium":
        return {
            "tickets": [
                {"id": 1, "text": "App crashes when I open settings. This is very frustrating."}
            ],
            "expected": {
                "classification": "technical",
                "sentiment": "angry",
                "policy": "troubleshoot"
            }
        }

    elif level == "hard":
        return {
            "tickets": [
                {"id": 1, "text": "Payment deducted but no confirmation received."}
            ],
            "expected": {
                "classification": "billing",
                "needs_info": True,
                "decision": "resolve",
                "final_keywords": ["refund", "transaction", "confirm"]
            }
        }