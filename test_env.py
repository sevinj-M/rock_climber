# test_env.py
import numpy as np
from env.climbing_env import ClimbingEnv

config = {"wall_width": 6, "wall_height": 8, "max_steps": 200}
env = ClimbingEnv(config)

obs, _ = env.reset()
print("Initial state:", obs)
print("State shape:  ", obs.shape)          # expect (12,)
print("Target hold:  ", env.target_hold, env.holds[env.target_hold])

mask = env.get_valid_action_mask()
valid = np.where(mask)[0]
print(f"Valid actions: {len(valid)} / {len(mask)}")

# Take 5 random valid steps
for i in range(5):
    action = int(np.random.choice(valid))
    limb, hold = divmod(action, env.n_holds)
    obs, reward, done, truncated, info = env.step(action)
    print(f"Step {i+1}: limb={limb} hold={hold} "
          f"reward={reward:.3f} done={done} info={info}")
    if done:
        break
    mask = env.get_valid_action_mask()
    valid = np.where(mask)[0]

print("Env test passed.")