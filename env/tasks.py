def load_task(level):
    if level == "easy":
        return {
            "tickets": [
                {"id": 1, "text": "I was charged twice for my subscription"}
            ],
            "expected": {
                "classification": "billing"
            }
        }

    elif level == "medium":
        return {
            "tickets": [
                {"id": 1, "text": "App crashes when I open settings"}
            ],
            "expected": {
                "classification": "technical",
                "keywords": ["crash", "settings"]
            }
        }

    elif level == "hard":
        return {
            "tickets": [
                {"id": 1, "text": "Payment failed but money deducted. Need refund ASAP"}
            ],
            "expected": {
                "classification": "billing",
                "action": "refund",
                "escalate": True
            }
        }