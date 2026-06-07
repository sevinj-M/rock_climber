import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy

from agent.network import QNetwork
from agent.replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Standard DQN with:
      - ε-greedy exploration with linear decay
      - Separate target network with soft updates
      - Action masking at both act-time and train-time
    """

    def __init__(self, config: dict, state_dim: int, n_actions: int):
        self.n_actions = n_actions
        self.batch_size = config["batch_size"]
        self.gamma      = config["gamma"]
        self.tau        = config["target_update_tau"]

        # Epsilon schedule
        self.epsilon       = config["epsilon_start"]
        self.epsilon_end   = config["epsilon_end"]
        self.epsilon_decay = config["epsilon_decay"]   # steps over which to decay
        self.steps_done    = 0

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        hidden_dim    = config.get("hidden_dim", 256)
        self.online   = QNetwork(state_dim, n_actions, hidden_dim).to(self.device)
        self.target   = copy.deepcopy(self.online).to(self.device)
        self.target.eval()   # target net is never trained directly

        # Optimizer & loss
        self.optimizer = optim.Adam(self.online.parameters(), lr=config["lr"])
        self.loss_fn   = nn.MSELoss()

        # Replay buffer
        self.buffer = ReplayBuffer(config["buffer_size"])

    # ── Interaction ────────────────────────────────────────────────────────────

    def act(self, state: np.ndarray, valid_mask: np.ndarray) -> int:
        """
        ε-greedy action selection.
        - With probability ε: pick a random valid action
        - Otherwise:          pick the greedy action from the online network
        Always respects the valid action mask.
        """
        self.steps_done += 1
        self._decay_epsilon()

        if np.random.rand() < self.epsilon:
            valid_indices = np.where(valid_mask)[0]
            return int(np.random.choice(valid_indices))

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mask_t  = torch.BoolTensor(valid_mask).unsqueeze(0).to(self.device)
        return self.online.act_greedy(state_t, mask_t)

    def store(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
        next_mask:  np.ndarray,
    ) -> None:
        self.buffer.store(state, action, reward, next_state, done, next_mask)

    # ── Learning ───────────────────────────────────────────────────────────────

    def learn(self) -> float | None:
        """
        Sample a minibatch and do one gradient step.
        Returns the loss value (float), or None if the buffer isn't ready yet.
        """
        if not self.buffer.is_ready(self.batch_size):
            return None

        batch = self.buffer.sample(self.batch_size)

        states      = torch.FloatTensor(batch["states"]).to(self.device)
        actions     = torch.LongTensor(batch["actions"]).to(self.device)
        rewards     = torch.FloatTensor(batch["rewards"]).to(self.device)
        next_states = torch.FloatTensor(batch["next_states"]).to(self.device)
        dones       = torch.FloatTensor(batch["dones"]).to(self.device)
        next_masks  = torch.BoolTensor(batch["next_masks"]).to(self.device)

        # ── Current Q-values ──────────────────────────────────────────────────
        # online(states) → (B, n_actions); gather the action that was taken
        all_q   = self.online(states, torch.ones_like(next_masks))  # unmask for training
        current_q = all_q.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

        # ── Target Q-values (no gradient) ─────────────────────────────────────
        with torch.no_grad():
            next_q      = self.target(next_states, next_masks)   # masked
            max_next_q  = next_q.max(dim=1).values               # (B,)
            target_q    = rewards + self.gamma * max_next_q * (1 - dones)

        # ── Bellman loss & update ─────────────────────────────────────────────
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()

        # ── Soft-update target network ─────────────────────────────────────────
        self._soft_update()

        return loss.item()

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        torch.save({
            "online":     self.online.state_dict(),
            "target":     self.target.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "steps_done": self.steps_done,
            "epsilon":    self.epsilon,
        }, path)
        print(f"Checkpoint saved → {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.steps_done = ckpt["steps_done"]
        self.epsilon    = ckpt["epsilon"]
        print(f"Checkpoint loaded ← {path}")

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _decay_epsilon(self) -> None:
        """Linear decay from epsilon_start → epsilon_end over epsilon_decay steps."""
        progress = min(self.steps_done / self.epsilon_decay, 1.0)
        self.epsilon = self.epsilon_end + (1.0 - progress) * (
            self.epsilon - self.epsilon_end
        )

    def _soft_update(self) -> None:
        """θ_target ← τ·θ_online + (1-τ)·θ_target"""
        for online_p, target_p in zip(
            self.online.parameters(), self.target.parameters()
        ):
            target_p.data.copy_(
                self.tau * online_p.data + (1 - self.tau) * target_p.data
            )