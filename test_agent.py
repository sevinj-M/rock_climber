# test_agent.py
import numpy as np
from env.climbing_env import ClimbingEnv
from agent.dqn_agent import DQNAgent

config = {
    "wall_width":        6,
    "wall_height":       8,
    "max_steps":         200,
    "lr":                1e-4,
    "gamma":             0.99,
    "epsilon_start":     1.0,
    "epsilon_end":       0.05,
    "epsilon_decay":     5000,
    "buffer_size":       10000,
    "batch_size":        64,
    "target_update_tau": 0.005,
    "hidden_dim":        256,
}

env   = ClimbingEnv(config)
state_dim = env.observation_space.shape[0]   # 12
n_actions = env.action_space.n               # 4 * n_holds

agent = DQNAgent(config, state_dim, n_actions)
print(f"Device: {agent.device}")
print(f"Online network:\n{agent.online}")

# ── Run a few episodes to fill the buffer ──────────────────
for ep in range(5):
    obs, _ = env.reset()
    done = truncated = False
    ep_reward = 0.0

    while not (done or truncated):
        mask   = env.get_valid_action_mask()
        action = agent.act(obs, mask)
        next_obs, reward, done, truncated, info = env.step(action)

        next_mask = env.get_valid_action_mask()
        agent.store(obs, action, reward, next_obs, done or truncated, next_mask)

        loss = agent.learn()
        obs  = next_obs
        ep_reward += reward

    print(f"  ep={ep+1}  reward={ep_reward:.3f}  "
          f"ε={agent.epsilon:.3f}  "
          f"buffer={len(agent.buffer)}  "
          f"loss={loss}")

# ── Save / load round-trip ─────────────────────────────────
import os
os.makedirs("checkpoints", exist_ok=True)
agent.save("checkpoints/test.pt")
agent.load("checkpoints/test.pt")

print("\nAgent test passed.")