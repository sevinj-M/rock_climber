import torch
import torch.nn as nn
import numpy as np


class QNetwork(nn.Module):
    """
    Fully-connected Q-network.

    Input:  state vector (12-dim)
    Output: Q-value for every action (4 * n_holds), with invalid actions
            masked to -inf so they can never be argmax-selected.
    """

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

        self._init_weights()

    def forward(self, state: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state : (B, state_dim)  float32
            mask  : (B, n_actions)  bool — True = valid action
        Returns:
            q     : (B, n_actions)  float32 — invalid actions set to -inf
        """
        q = self.net(state)
        q = q.masked_fill(~mask, float("-inf"))
        return q

    def act_greedy(self, state: torch.Tensor, mask: torch.Tensor) -> int:
        """Single-sample greedy action. Used at eval time."""
        with torch.no_grad():
            q = self.forward(state, mask)
            return int(q.argmax(dim=-1).item())

    def _init_weights(self):
        """
        Small orthogonal init on hidden layers, near-zero on output.
        Keeps Q-values close to zero early in training, which helps
        the agent explore rather than commit to arbitrary preferences.
        """
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        # Output layer — smaller scale
        last = self.net[-1]
        nn.init.orthogonal_(last.weight, gain=0.01)
        nn.init.zeros_(last.bias)