import yaml
import numpy as np

from env.climbing_env import ClimbingEnv
from agent.dqn_agent import DQNAgent


def evaluate(
    checkpoint_path: str = "checkpoints/best.pt",
    config_path:     str = "configs/default.yaml",
    n_episodes:      int = 20,
    render:          bool = False,
):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    env       = ClimbingEnv(config)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent     = DQNAgent(config, state_dim, n_actions)
    agent.load(checkpoint_path)

    # Greedy evaluation — no exploration
    agent.epsilon = 0.0

    rewards      = []
    success_rate = 0

    print(f"\nEvaluating {checkpoint_path} over {n_episodes} episodes …\n")

    for ep in range(1, n_episodes + 1):
        obs, _    = env.reset()
        done      = truncated = False
        ep_reward = 0.0
        steps     = 0
        success   = False

        while not (done or truncated):
            mask   = env.get_valid_action_mask()
            action = agent.act(obs, mask)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            steps     += 1
            if info.get("reached_target"):
                success = True

            if render:
                _render_ascii(env)

        rewards.append(ep_reward)
        if success:
            success_rate += 1

        status = "✓" if success else "✗"
        print(f"  {status} ep {ep:>3} | reward {ep_reward:>7.2f} | steps {steps:>3}")

    print(f"\nResults over {n_episodes} episodes:")
    print(f"  Success rate : {success_rate}/{n_episodes} "
          f"({100*success_rate/n_episodes:.1f}%)")
    print(f"  Avg reward   : {np.mean(rewards):.3f}")
    print(f"  Std reward   : {np.std(rewards):.3f}")
    print(f"  Best episode : {max(rewards):.3f}")
    print(f"  Worst episode: {min(rewards):.3f}")

    return rewards


# training/evaluate.py
import yaml
import numpy as np
from env.climbing_env import ClimbingEnv
from agent.dqn_agent import DQNAgent


def evaluate(
    checkpoint_path: str = "checkpoints/best.pt",
    config_path:     str = "configs/default.yaml",
    n_episodes:      int = 100,
    render:          bool = False,
    wall: dict = None
):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    env = ClimbingEnv(config, wall=wall)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    agent     = DQNAgent(config, state_dim, n_actions)
    agent.load(checkpoint_path)
    agent.epsilon = 0.0

    # ── Pygame render path ─────────────────────────────────────────────────────
    if render:
        from visualization.renderer import ClimbingRenderer
        renderer = ClimbingRenderer(env)
        results  = renderer.render(agent, n_episodes=n_episodes, fps=6)
        renderer.close()

        rewards      = [r["reward"]  for r in results]
        success_rate = sum(r["success"] for r in results)
        _print_summary(rewards, success_rate, n_episodes)
        return rewards

    # ── Headless path (unchanged) ──────────────────────────────────────────────
    rewards      = []
    success_rate = 0

    print(f"\nEvaluating {checkpoint_path} over {n_episodes} episodes …\n")
    for ep in range(1, n_episodes + 1):
        obs, _    = env.reset()
        done      = truncated = False
        ep_reward = 0.0
        steps     = 0
        success   = False

        while not (done or truncated):
            mask   = env.get_valid_action_mask()
            action = agent.act(obs, mask)
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            steps     += 1
            if info.get("reached_target"):
                success = True

        rewards.append(ep_reward)
        if success:
            success_rate += 1

        status = "✓" if success else "✗"
        print(f"  {status} ep {ep:>3} | reward {ep_reward:>7.2f} | steps {steps:>3}")

    _print_summary(rewards, success_rate, n_episodes)
    return rewards


def _print_summary(rewards, success_rate, n_episodes):
    print(f"\nResults over {n_episodes} episodes:")
    print(f"  Success rate : {success_rate}/{n_episodes} "
          f"({100*success_rate/n_episodes:.1f}%)")
    print(f"  Avg reward   : {np.mean(rewards):.3f}")
    print(f"  Std reward   : {np.std(rewards):.3f}")
    print(f"  Best episode : {max(rewards):.3f}")
    print(f"  Worst episode: {min(rewards):.3f}")

if __name__ == "__main__":
    evaluate(render=True)