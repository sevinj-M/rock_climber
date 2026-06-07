import numpy as np

MAX_REACH = 1.5
MIN_SUPPORT_LIMBS = 0

def in_reach(hold_a: tuple, hold_b: tuple) -> bool:
    """True if two holds are within reach of each other."""
    return np.linalg.norm(np.array(hold_a) - np.array(hold_b)) <= MAX_REACH

def compute_center_of_mass(limb_positions: list) -> np.ndarray | None:
    """Mean position of all active limbs. Returns None if no limbs placed."""
    if not limb_positions:
        return None
    return np.mean(limb_positions, axis=0)

def is_stable(limb_positions: list) -> bool:
    """
    Agent is stable if:
      - At least MIN_SUPPORT_LIMBS are on holds
      - Center of mass x lies between leftmost and rightmost contact point
    """
    if len(limb_positions) < MIN_SUPPORT_LIMBS:
        return False
    com = compute_center_of_mass(limb_positions)
    xs = [p[0] for p in limb_positions]
    return float(min(xs)) <= float(com[0]) <= float(max(xs))
    


