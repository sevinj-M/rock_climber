import argparse
import yaml
from env.wall_generator import generate_wall
from env.climbing_env import ClimbingEnv
from agent.dqn_agent import DQNAgent
from training.train import train
from training.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description="Rock Climber RL")
    parser.add_argument("mode",          choices=["train", "eval", "preview"])
    parser.add_argument("--config",      default="configs/default.yaml")
    parser.add_argument("--checkpoint",  default="checkpoints/best.pt")
    parser.add_argument("--episodes",   type=int,   default=100)
    parser.add_argument("--render",      action="store_true")
    parser.add_argument("--seed",        type=int,   default=3)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # ── Generate wall (shared across all modes) ────────────────────────────────
    wall = generate_wall(
        width         = config["wall_width"],
        height        = config["wall_height"],
        density       = config.get("density",       0.4),
        dead_end_prob = config.get("dead_end_prob", 0.15),
        seed          = args.seed,
    )

    if args.mode == "preview":
        # Just print the wall and exit — useful for picking a seed
        from env.wall_generator import print_wall
        print_wall(wall)
        return

    if args.mode == "train":
        train(config_path=args.config, wall=wall)

    elif args.mode == "eval":
        evaluate(
            checkpoint_path = args.checkpoint,
            config_path     = args.config,
            n_episodes      = args.episodes,
            render          = args.render,
            wall            = wall,
        )


if __name__ == "__main__":
    main()