import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """
    Circular buffer storing (s, a, r, s', done, mask') transitions.
    'mask' is the valid action mask for the *next* state — the agent
    needs it at training time to restrict the target Q argmax.
    """

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)   # auto-drops oldest when full

    # ── Public API ─────────────────────────────────────────────────────────────

    def store(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
        next_mask:  np.ndarray,
    ) -> None:
        """Push one transition. Oldest entry is silently dropped when at capacity."""
        self.buffer.append((state, action, reward, next_state, done, next_mask))

    def sample(self, batch_size: int) -> dict:
        """
        Returns a dict of numpy arrays, each of length batch_size.
        Raises ValueError if the buffer doesn't have enough transitions yet.
        """
        if len(self) < batch_size:
            raise ValueError(
                f"Not enough transitions to sample: "
                f"have {len(self)}, need {batch_size}"
            )

        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones, next_masks = zip(*batch)

        return {
            "states":      np.array(states,      dtype=np.float32),
            "actions":     np.array(actions,     dtype=np.int64),
            "rewards":     np.array(rewards,     dtype=np.float32),
            "next_states": np.array(next_states, dtype=np.float32),
            "dones":       np.array(dones,       dtype=np.float32),  # float for math
            "next_masks":  np.array(next_masks,  dtype=bool),
        }

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """Convenience check before calling sample()."""
        return len(self) >= batch_size