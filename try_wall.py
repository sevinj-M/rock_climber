# try_wall.py
from env.wall_generator import generate_wall, print_wall
from env.climbing_env import ClimbingEnv

# Preview a few walls until you like one
for seed in range(5):
    wall = generate_wall(width=8, height=12, density=0.4,
                         dead_end_prob=0.15, seed=seed)
    print(f"Seed {seed}:", end="")
    print_wall(wall)