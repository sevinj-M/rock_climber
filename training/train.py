import os
import time
import yaml
import numpy as np
from collections import deque

from env.climbing_env import ClimbingEnv
from agent.dqn_agent import DQNAgent


def train(config_path: str = "configs/default.yaml"):

    # ── Config ─────────────────────────────────────────────────────────────────
    with open(config_path) as f:
        config = yaml.safe_load(f)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs",        exist_ok=True)

    # ── Setup ──────────────────────────────────────────────────────────────────
    env       = ClimbingEnv(config)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent     = DQNAgent(config, state_dim, n_actions)

    n_episodes     = config.get("n_episodes",      2000)
    save_every     = config.get("save_every",       200)
    log_every      = config.get("log_every",         10)
    reward_window  = config.get("reward_window",     50)  # rolling average window

    # ── Logging state ──────────────────────────────────────────────────────────
    reward_history = deque(maxlen=reward_window)
    best_avg_reward = -float("inf")
    log_rows = []       # accumulated for CSV

    print(f"Training on {agent.device} | "
          f"{n_episodes} episodes | "
          f"wall {config['wall_width']}×{config['wall_height']}")
    print("-" * 60)

    t_start = time.time()

    # ── Episode loop ───────────────────────────────────────────────────────────
    for episode in range(1, n_episodes + 1):

        obs, _          = env.reset()
        done            = False
        truncated       = False
        ep_reward       = 0.0
        ep_loss         = []
        ep_steps        = 0
        reached_target  = False

        # ── Step loop ──────────────────────────────────────────────────────────
        while not (done or truncated):
            mask   = env.get_valid_action_mask()
            action = agent.act(obs, mask)

            next_obs, reward, done, truncated, info = env.step(action)
            next_mask = env.get_valid_action_mask()

            agent.store(obs, action, reward, next_obs,
                        done or truncated, next_mask)
            loss = agent.learn()

            obs        = next_obs
            ep_reward += reward
            ep_steps  += 1
            if loss is not None:
                ep_loss.append(loss)
            if info.get("reached_target"):
                reached_target = True

        # ── Per-episode bookkeeping ─────────────────────────────────────────────
        reward_history.append(ep_reward)
        avg_reward = np.mean(reward_history)
        avg_loss   = np.mean(ep_loss) if ep_loss else 0.0
        elapsed    = time.time() - t_start

        log_rows.append({
            "episode":        episode,
            "reward":         round(ep_reward, 4),
            "avg_reward":     round(avg_reward, 4),
            "loss":           round(avg_loss,   6),
            "epsilon":        round(agent.epsilon, 4),
            "steps":          ep_steps,
            "reached_target": int(reached_target),
            "elapsed_s":      round(elapsed, 1),
        })

        # ── Console log ────────────────────────────────────────────────────────
        if episode % log_every == 0:
            flag = "🏆" if reached_target else "  "
            print(
                f"{flag} ep {episode:>5} | "
                f"reward {ep_reward:>7.2f} | "
                f"avg({reward_window}) {avg_reward:>7.2f} | "
                f"loss {avg_loss:.4f} | "
                f"ε {agent.epsilon:.3f} | "
                f"steps {ep_steps:>3} | "
                f"{elapsed:.0f}s"
            )

        # ── Save best model ─────────────────────────────────────────────────────
        if avg_reward > best_avg_reward and len(reward_history) == reward_window:
            best_avg_reward = avg_reward
            agent.save("checkpoints/best.pt")

        # ── Periodic checkpoint ─────────────────────────────────────────────────
        if episode % save_every == 0:
            agent.save(f"checkpoints/ep_{episode:05d}.pt")
            _flush_log(log_rows, "logs/train_log.csv")
            log_rows = []

    # ── Final save ─────────────────────────────────────────────────────────────
    agent.save("checkpoints/final.pt")
    _flush_log(log_rows, "logs/train_log.csv")
    print("-" * 60)
    print(f"Training complete. Best avg reward: {best_avg_reward:.3f}")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _flush_log(rows: list, path: str) -> None:
    """Append accumulated log rows to a CSV file."""
    if not rows:
        return
    write_header = not os.path.exists(path)
    with open(path, "a") as f:
        if write_header:
            f.write(",".join(rows[0].keys()) + "\n")
        for row in rows:
            f.write(",".join(str(v) for v in row.values()) + "\n")


if __name__ == "__main__":
    train()