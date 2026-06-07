import argparse
from training.train    import train
from training.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description="Rock Climber RL")
    parser.add_argument("mode",     choices=["train", "eval"], default="train")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--episodes",   type=int, default=20)
    parser.add_argument("--render",     action="store_true")
    args = parser.parse_args()

    if args.mode == "train":
        train(config_path=args.config)
    else:
        evaluate(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            n_episodes=args.episodes,
            render=args.render,
        )


if __name__ == "__main__":
    main()