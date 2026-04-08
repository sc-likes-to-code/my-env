from env.environment import SupportEnv

env = SupportEnv()

obs = env.reset("easy")
print("OBS:", obs)

action = type("A", (), {
    "action_type": "classify",
    "ticket_id": 1,
    "content": "billing"
})()

obs, reward, done, _ = env.step(action)

print("REWARD:", reward.score)
print("DONE:", done)