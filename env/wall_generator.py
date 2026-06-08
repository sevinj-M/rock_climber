import numpy as np
from collections import deque
from env.physics import in_reach


def generate_wall(
    width:        int   = 8,
    height:       int   = 12,
    density:      float = 0.4,
    dead_end_prob: float = 0.15,
    seed:         int   = None,
) -> dict:
    """
    Generate a sparse climbing wall with a guaranteed solution path.

    Returns:
        {
            "holds":        list of (x, y) tuples,
            "start_holds":  [lf, rf, lh, rh] hold indices for reset(),
            "target_hold":  hold index for the goal,
            "solution_path": list of hold indices forming one valid route,
        }
    """
    rng = np.random.default_rng(seed)

    while True:
        holds = _sample_holds(width, height, density, dead_end_prob, rng)
        result = _verify_solution(holds, width, height)
        if result is not None:
            return result


# ── Hold sampling ──────────────────────────────────────────────────────────────

def _sample_holds(width, height, density, dead_end_prob, rng) -> list:
    holds = set()

    # ── Guaranteed anchor holds at bottom ─────────────────────────────────────
    cx = width // 2
    for x in [cx - 1, cx]:
        holds.add((float(x),     0.0))   # feet row
        holds.add((float(x),     1.0))   # hands row

    # ── Guaranteed target at top ───────────────────────────────────────────────
    holds.add((float(cx), float(height - 1)))

    # ── Random sparse holds ────────────────────────────────────────────────────
    for y in range(2, height - 1):
        for x in range(width):
            if rng.random() < density:
                holds.add((float(x), float(y)))

    # ── Dead-end holds (reachable from main wall but lead nowhere useful) ──────
    # These are holds that the agent might step onto and get stuck
    for y in range(2, height - 2):
        for x in range(width):
            if (float(x), float(y)) not in holds:
                if rng.random() < dead_end_prob:
                    holds.add((float(x), float(y)))

    return sorted(holds, key=lambda h: (h[1], h[0]))


# ── Solution verification via BFS ─────────────────────────────────────────────

def _verify_solution(holds, width, height) -> dict | None:
    """
    BFS over climber states to check a valid route exists.

    State: (lh, rh, lf, rf) — indices into holds list.
    A move: pick one limb, move it to a reachable unoccupied hold.
    Stability: checked at each state (simplified — COM between contacts).

    Returns the wall dict if solvable, None otherwise.
    """
    n      = len(holds)
    h_idx  = {h: i for i, h in enumerate(holds)}

    cx     = width // 2
    target = h_idx[(float(cx), float(height - 1))]

    # Start state — feet on row 0, hands on row 1
    lf = h_idx[(float(cx - 1), 0.0)]
    rf = h_idx[(float(cx),     0.0)]
    lh = h_idx[(float(cx - 1), 1.0)]
    rh = h_idx[(float(cx),     1.0)]
    start = (lh, rh, lf, rf)

    # BFS
    queue    = deque()
    visited  = set()
    parent   = {}          # state → (prev_state, limb_moved, hold_moved_to)

    queue.append(start)
    visited.add(start)
    goal_state = None

    while queue:
        state = queue.popleft()
        lh_i, rh_i, lf_i, rf_i = state

        if target in state:
            goal_state = state
            break

        occupied = set(state)

        for limb in range(4):
            current_hold = state[limb]
            ref_pos      = holds[current_hold]

            for hold_id in range(n):
                if hold_id in occupied:
                    continue
                if not in_reach(ref_pos, holds[hold_id]):
                    continue

                new_state = list(state)
                new_state[limb] = hold_id
                new_state = tuple(new_state)

                if not _stable(new_state, holds):
                    continue
                if new_state in visited:
                    continue

                visited.add(new_state)
                parent[new_state] = (state, limb, hold_id)
                queue.append(new_state)

    if goal_state is None:
        return None   # no solution — regenerate

    # Reconstruct solution path (sequence of hold indices touched)
    path   = []
    state  = goal_state
    while state in parent:
        prev, limb, hold_id = parent[state]
        path.append(hold_id)
        state = prev
    path.reverse()

    return {
        "holds":         holds,
        "start_holds":   list(start),      # [lh, rh, lf, rf]
        "target_hold":   target,
        "solution_path": path,
        "n_holds":       n,
    }


def _stable(state, holds) -> bool:
    """Simplified stability: COM x within [min_x, max_x] of contacts."""
    positions = [holds[i] for i in state]
    xs  = [p[0] for p in positions]
    com_x = sum(xs) / len(xs)
    return min(xs) <= com_x <= max(xs)


def print_wall(wall: dict):
    """ASCII preview of the generated wall."""
    holds  = wall["holds"]
    target = wall["target_hold"]
    starts = set(wall["start_holds"])
    path   = set(wall["solution_path"])

    xs = [int(h[0]) for h in holds]
    ys = [int(h[1]) for h in holds]
    W  = max(xs) + 1
    H  = max(ys) + 1

    h_idx = {h: i for i, h in enumerate(holds)}

    print(f"\nWall {W}×{H} | {len(holds)} holds | "
          f"solution path: {len(wall['solution_path'])} moves\n")

    for y in range(H - 1, -1, -1):
        row = ""
        for x in range(W):
            key = (float(x), float(y))
            if key not in h_idx:
                row += "  ·"
                continue
            i = h_idx[key]
            if i == target:
                row += "  ★"
            elif i in starts:
                row += "  S"
            elif i in path:
                row += "  ○"
            else:
                row += "  ●"
        print(f"{y:2d} {row}")

    print("   " + "".join(f"  {x}" for x in range(W)))
    print("\nLegend: S=start  ○=solution path  ●=hold  ★=target\n")