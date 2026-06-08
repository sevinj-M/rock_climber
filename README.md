<div align="center">

# 🧗 Rock Climber RL

<p>
  A reinforcement learning agent that learns to climb a wall —<br>
  from scratch, with no human demonstrations, using only physics, rewards, and trial and error.
</p>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-009688?style=flat)
![Pygame](https://img.shields.io/badge/Pygame-2.5+-1C1C1C?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

</div>

---

> Built with **Deep Q-Networks (DQN)**, a custom **Gymnasium environment**, and a real-time **Pygame renderer**.
> The agent starts knowing nothing and ends up finding efficient routes up procedurally generated walls
> with a **95%+ success rate**.

---

## 📸 Demo

<!-- Replace with an actual GIF once you have a screen recording -->
<!-- Recommended: record with OBS or ShareX, convert to GIF with ezgif.com -->

```
11   ·  ·  ·  ·  ★  ·  ·  ·
10   ·  ·  ●  ·  ●  ·  ·  ·
 9   ·  ●  ·  ○  ·  ●  ·  ·
 8   ·  ·  ○  ·  ●  ·  ·  ●
 7   ●  ·  ·  ○  ·  ·  ●  ·
 6   ·  ●  ·  ·  ○  ·  ·  ·
 5   ·  ·  ●  ○  ·  ●  ·  ·
 4   ·  ●  ·  ·  ○  ·  ·  ·
 3   ·  ·  ○  ·  ·  ●  ·  ·
 2   ●  ·  ·  ●  ·  ·  ●  ·
 1   ·  ·  S  S  ·  ·  ·  ·
 0   ·  ·  S  S  ·  ·  ·  ·
      0  1  2  3  4  5  6  7

S = start   ○ = solution path   ● = hold   ★ = target
```

**Training results — 8×12 wall, 5000 episodes:**

| Metric | Value |
|:---|:---|
| ✅ Success rate | 95 / 100 **(95%)** |
| 📈 Avg reward | 9.450 |
| 🏆 Best episode | 10.110 |
| ⏱ Training time | ~12 min (CPU) |

---

## 💡 Why I built this

I love rock climbing. I also just learned reinforcement learning. So I asked myself — can an agent figure out how to climb a wall the same way I did when I started: by falling a lot, noticing what works, and slowly building intuition?

The answer turned out to be yes, and watching it happen was one of the most satisfying things I've built.

---

## ⚡ Quickstart

```bash
# Clone and install
git clone https://github.com/your-username/rock-climber-rl.git
cd rock-climber-rl
pip install -e .

# Preview a generated wall before training
python main.py preview --seed 3

# Train
python main.py train --seed 3

# Evaluate with real-time Pygame rendering
python main.py eval --seed 3 --render --episodes 10
```

---

## 🏗 Project structure

```
rock_climber_rl/
│
├── env/
│   ├── climbing_env.py      # Gymnasium environment (holds, state, step, reset)
│   ├── physics.py           # Balance check, reach validation, center of mass
│   └── wall_generator.py    # Procedural wall generation + BFS solution verifier
│
├── agent/
│   ├── dqn_agent.py         # DQN agent (act, store, learn, save/load)
│   ├── network.py           # Q-network with action masking
│   └── replay_buffer.py     # Circular experience replay buffer
│
├── training/
│   ├── train.py             # Training loop with logging + checkpointing
│   └── evaluate.py          # Headless or rendered evaluation
│
├── visualization/
│   └── renderer.py          # Real-time Pygame renderer
│
├── configs/
│   └── default.yaml         # All hyperparameters in one place
│
├── main.py                  # Entry point — train / eval / preview
└── requirements.txt
```

---

## 🧠 How it works

### The environment

The wall is a 2D grid of holds. The agent controls four limbs — left hand, right hand, left foot, right foot — each placed on a hold at any time.

At every step the agent picks one limb and moves it to a new hold. An action is only valid if:

- The target hold is **within physical reach** of that limb
- The hold is **not already occupied** by another limb
- The resulting position is **stable** — the agent's center of mass must fall within the horizontal span of its contact points

If the agent reaches an unstable position, it falls. If it reaches the target hold, it wins.

### The agent

A **Deep Q-Network** with two hidden layers (256 units each) maps the 12-dimensional state vector to Q-values over all possible actions. Invalid actions are masked to `-∞` before argmax — the network never wastes capacity on impossible moves.

Training uses standard DQN techniques:

| Technique | Purpose |
|:---|:---|
| Experience replay | Circular 10k buffer, random sampling breaks temporal correlation |
| Target network | Slowly-updated copy stabilizes Bellman targets |
| ε-greedy exploration | Starts fully random, decays to 5% over 20k steps |
| Curriculum learning | Starts near the top, gradually lowers start position as wins accumulate |

### State vector — 12 dimensions

```
[lh_x, lh_y,  rh_x, rh_y,  lf_x, lf_y,  rf_x, rf_y,  com_x, com_y,  target_x, target_y]
  ──── left hand ────  ─── right hand ───  ──── left foot ───  ─── right foot ──  ── COM ──  ── goal ──
```

### Reward shaping

| Event | Reward |
|:---|:---|
| Reach target | `+10.0` |
| Fall (unstable position) | `−5.0` |
| Upward progress | `+0.5 × Δheight` |
| Each step taken | `−0.001` |

### Wall generation

Walls are **procedurally generated** with a **guaranteed solution path**.

1. Random holds are sampled at a configurable density
2. Dead-end holds are scattered across the wall as traps
3. A **BFS over the full climber state space** verifies at least one valid route exists
4. If no solution is found, the wall is regenerated — so training never starts on an unsolvable layout

```bash
# Preview walls by seed — pick one you like
python main.py preview --seed 0
python main.py preview --seed 3
python main.py preview --seed 7
```

---

## 🔍 Key design decisions

**Action masking over action clipping.**
Rather than letting the network output Q-values for illegal actions and clipping them at execution time, illegal actions are masked to `-∞` before argmax. The network never sees impossible moves as options at all.

**BFS-verified wall generation.**
Random sparse walls can easily be unsolvable. Rather than hoping density is sufficient, a BFS over the full climber state space confirms a solution exists before any training starts. This makes the hardest walls reliably fair.

**Curriculum learning.**
Starting at the bottom of a 12-row sparse wall means thousands of episodes of random falling before the first success — too sparse a signal for the Q-network. Starting near the top, confirming the agent can solve the easy version, then gradually lowering the start position gave clean convergence from episode one.

---

## 📖 What I learned

This project forced me to think carefully about the gap between RL theory and practice. The Bellman equation is straightforward on paper. Getting it to actually work — with the right reward scale, the right exploration schedule, a stable target network, masked actions, and a solvable environment — required debugging each piece in isolation and understanding why each design choice matters.

**The biggest surprise:** reward shaping matters more than network architecture. Changing the step penalty from `-0.01` to `-0.001` was the single change that most improved convergence. The agent had been learning "don't move" as the optimal policy because accumulated step penalties outweighed climbing progress. Fixing that unlocked everything else.

---

## 🗺 What's next

- [ ] **Randomized walls per episode** — force the agent to generalize rather than memorize one layout
- [ ] **Double DQN** — reduce Q-value overestimation for more stable long-term planning
- [ ] **Fatigue mechanic** — holds degrade with use, forcing the agent to plan full routes
- [ ] **3D wall** — extend to a voxel grid with left/right traversal as well as up

---

## 📦 Requirements

```
python    >= 3.10
gymnasium >= 0.29
torch     >= 2.0
pygame    >= 2.5
numpy     >= 1.24
pyyaml    >= 6.0
```

---

<div align="center">
  <i>Built from scratch combining two things I love: climbing and machine learning.</i>
</div>
