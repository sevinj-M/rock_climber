import numpy as np
import gymnasium as gym
from gymnasium import spaces
from env.physics import in_reach, is_stable, compute_center_of_mass


class ClimbingEnv(gym.Env):
    """
    2D grid climbing environment.

    Wall:   W columns × H rows of holds, (0,0) = bottom-left, y increases upward.
    Agent:  4 limbs — left hand (LH), right hand (RH), left foot (LF), right foot (RF).
            Each limb is either on a hold (index into self.holds) or None.
    Action: Discrete — (limb_id × n_holds). Only valid actions are allowed.
    State:  12-dim float32 vector (see _get_state).
    """

    LH, RH, LF, RF = 0, 1, 2, 3  # limb indices

    def __init__(self, config: dict):
        super().__init__()
        self.W = config["wall_width"]
        self.H = config["wall_height"]
        self.max_steps = config.get("max_steps", 200)

        self.holds = self._generate_holds()   # list of (x, y)
        self.n_holds = len(self.holds)
        self.target_hold = self._pick_target_hold()

        # Spaces
        # State: [lh_x, lh_y, rh_x, rh_y, lf_x, lf_y, rf_x, rf_y,
        #         com_x, com_y, target_x, target_y]
        low  = np.full(12, -1.0, dtype=np.float32)
        high = np.array([self.W, self.H] * 4 + [self.W, self.H, self.W, self.H],
                        dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(4 * self.n_holds)

        # Runtime state (populated in reset)
        self.limb_holds = [None, None, None, None]
        self.step_count = 0

    # ── Gym API ────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        # Place feet on the two bottom-centre holds, hands just above
        cx = self.W // 2
        lf_hold = self._find_hold(cx - 1, 0)
        rf_hold = self._find_hold(cx,     0)
        lh_hold = self._find_hold(cx - 1, 1)
        rh_hold = self._find_hold(cx,     1)

        self.limb_holds = [lh_hold, rh_hold, lf_hold, rf_hold]
        return self._get_state(), {}

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action {action}"
        self.step_count += 1

        # 1. Decode
        limb_id, hold_id = divmod(action, self.n_holds)

        # 2. Record previous COM height for progress reward
        prev_com_y = self._com_y()

        # 3. Validate (mask should prevent this, but guard anyway)
        if not self._is_action_valid(limb_id, hold_id):
            # Penalise and end episode — agent tried an illegal move
            return self._get_state(), -2.0, True, False, {"reason": "illegal_action"}

        # 4. Move limb
        self.limb_holds[limb_id] = hold_id

        # 5. Stability check
        active_positions = [
            self.holds[h] for h in self.limb_holds if h is not None
        ]
        fell = not is_stable(active_positions)

        # 6. Target check
        reached_target = (hold_id == self.target_hold)

        # 7. Reward
        new_com_y = self._com_y()
        reward = self._compute_reward(prev_com_y, new_com_y, fell, reached_target)

        # 8. Done?
        truncated = self.step_count >= self.max_steps
        done = fell or reached_target

        info = {
            "fell": fell,
            "reached_target": reached_target,
            "step": self.step_count,
        }
        return self._get_state(), reward, done, truncated, info

    # ── State ──────────────────────────────────────────────────────────────────

    def _get_state(self) -> np.ndarray:
        """
        Returns a 12-dim float32 vector:
          [lh_x, lh_y, rh_x, rh_y, lf_x, lf_y, rf_x, rf_y,
           com_x, com_y, target_x, target_y]
        Limbs not on a hold are represented as (-1, -1).
        """
        coords = []
        for h in self.limb_holds:
            coords.extend(self.holds[h] if h is not None else (-1.0, -1.0))

        active = [self.holds[h] for h in self.limb_holds if h is not None]
        com = compute_center_of_mass(active) if active else np.array([-1.0, -1.0])

        target_pos = self.holds[self.target_hold]
        state = np.array(coords + list(com) + list(target_pos), dtype=np.float32)
        return state

    def get_valid_action_mask(self) -> np.ndarray:
        """
        Boolean array of shape (4 * n_holds,).
        Action (limb, hold) is valid if:
          - The hold is in reach of that limb's current position
          - The hold is not already occupied by another limb
        If a limb has no current hold, use the nearest foot hold as reference.
        """
        occupied = set(h for h in self.limb_holds if h is not None)
        mask = np.zeros(4 * self.n_holds, dtype=bool)

        for limb in range(4):
            current = self.limb_holds[limb]
            ref_pos = self.holds[current] if current is not None \
                      else self._fallback_position(limb)

            for hold_id, hold_pos in enumerate(self.holds):
                if hold_id in occupied:
                    continue
                if in_reach(ref_pos, hold_pos):
                    mask[limb * self.n_holds + hold_id] = True

        return mask

    # ── Reward ─────────────────────────────────────────────────────────────────

    def _compute_reward(
        self,
        prev_com_y: float,
        new_com_y: float,
        fell: bool,
        reached_target: bool,
    ) -> float:
        if reached_target:
            return +10.0
        if fell:
            return  -5.0
        upward_progress = new_com_y - prev_com_y   # positive = climbed up
        step_penalty    = -0.01
        return upward_progress * 0.1 + step_penalty

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _generate_holds(self) -> list[tuple]:
        """Full grid of holds. Rows ordered bottom-up (y=0 at bottom)."""
        holds = []
        for y in range(self.H):
            for x in range(self.W):
                holds.append((float(x), float(y)))
        return holds

    def _pick_target_hold(self) -> int:
        """Top-centre hold is the goal."""
        cx = self.W // 2
        return self._find_hold(cx, self.H - 1)

    def _find_hold(self, x: int, y: int) -> int:
        """Return hold index closest to (x, y), clamped to wall bounds."""
        x = max(0, min(x, self.W - 1))
        y = max(0, min(y, self.H - 1))
        target = (float(x), float(y))
        return min(range(self.n_holds),
                   key=lambda i: np.linalg.norm(
                       np.array(self.holds[i]) - np.array(target)))

    def _is_action_valid(self, limb_id: int, hold_id: int) -> bool:
        mask = self.get_valid_action_mask()
        return bool(mask[limb_id * self.n_holds + hold_id])

    def _com_y(self) -> float:
        """Current vertical center of mass. Returns 0.0 if no limbs placed."""
        active = [self.holds[h] for h in self.limb_holds if h is not None]
        if not active:
            return 0.0
        com = compute_center_of_mass(active)
        return float(com[1])

    def _fallback_position(self, limb: int) -> tuple:
        """
        Reference position for a limb with no current hold.
        Feet default to bottom row, hands to row 1.
        """
        row = 0 if limb in (self.LF, self.RF) else 1
        col = 0 if limb in (self.LH, self.LF) else self.W - 1
        return (float(col), float(row))


