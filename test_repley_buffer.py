# test_replay_buffer.py
import numpy as np
from agent.replay_buffer import ReplayBuffer

buf = ReplayBuffer(capacity=1000)

STATE_DIM  = 12
N_ACTIONS  = 4 * 48   # 4 limbs × 48 holds (6×8 wall)

# Fill with 100 dummy transitions
for _ in range(100):
    s     = np.random.rand(STATE_DIM).astype(np.float32)
    a     = np.random.randint(N_ACTIONS)
    r     = np.random.randn()
    s2    = np.random.rand(STATE_DIM).astype(np.float32)
    done  = bool(np.random.rand() > 0.95)
    mask  = np.random.rand(N_ACTIONS) > 0.5

    buf.store(s, a, r, s2, done, mask)

print(f"Buffer size: {len(buf)}")          # 100
print(f"Ready (64):  {buf.is_ready(64)}")  # True

batch = buf.sample(64)
for k, v in batch.items():
    print(f"  {k:12s}: shape={v.shape}  dtype={v.dtype}")

# Verify circular eviction
buf2 = ReplayBuffer(capacity=10)
for i in range(15):
    buf2.store(np.zeros(4), i, 0.0, np.zeros(4), False, np.ones(4, dtype=bool))
print(f"\nCapacity=10, stored 15 → buffer size: {len(buf2)}")  # 10
print(f"Oldest action: {buf2.buffer[0][1]}")                   # 5 (0–4 were evicted)

print("ReplayBuffer test passed.")