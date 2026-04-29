# Pac-Man DQN — Training Notes

## What is this agent doing?

The agent uses **DQN (Deep Q-Network)** — the same algorithm DeepMind used to beat Atari games in 2013. The idea is simple: train a small neural network to predict "how good is it to take action X in situation Y?" Those predictions are called **Q-values**. Whichever action has the highest Q-value gets chosen.

The network never sees the pixels. Instead, every frame we hand it 30 numbers describing the current situation (where Pac-Man is, where each ghost is, where the nearest pellet is, etc.). It outputs 5 numbers — one score per possible action. Pac-Man takes the highest-scoring action.

---

## The 30-number game state (input to the network)

| # | What it is | How it's encoded |
|---|---|---|
| 1–2 | Pac-Man position | x/screen_width, y/screen_height (0–1) |
| 3–6 | Ghost 1 offset + scared + direction | dx/w, dy/h, 0 or 1, direction/4 |
| 7–10 | Ghost 2 | same |
| 11–14 | Ghost 3 | same |
| 15–18 | Ghost 4 | same |
| 19–20 | Nearest pellet position | x/w, y/h (absolute) |
| 21 | Pellets remaining | count/300 |
| 22–26 | Pac-Man's current direction | one-hot (5 values) |
| 27–30 | Which moves are physically possible | 1 if passable, 0 if wall |

**Why normalise everything to 0–1?**
Without normalisation, raw pixel coordinates (0–448) are hundreds of times larger than the 0/1 flags. The network would focus almost entirely on coordinates and basically ignore the boolean features. Normalising puts everything on the same scale so all features get a fair chance.

**Why are ghost positions relative but pellet position absolute?**
Ghost offsets (dx, dy from Pac-Man) are more useful than absolute positions — "ghost is 3 tiles to my left" generalises across the whole map. For the nearest pellet I kept absolute coordinates as minor inconsisteny because I have realised that after 10 hours of training and decided to have it as a possible future improvement.

---

## Reward shaping — what does the agent get points for?

The agent doesn't just learn from the game score. I add extra signals to make learning faster:

| Event | Reward |
|---|---|
| Score increased (ate pellet, etc.) | +score delta (direct game points) |
| Survived a frame | +0.1 |
| Got closer to nearest pellet | +(prev_dist − curr_dist) × 0.1 |
| Power pellet activated (ghosts turned scared) | +30 |
| Ate a ghost (score jumped ≥200) | +50 |
| Changed direction | −0.5 |
| Died | −100 |

The direction-change penalty is small but discourages pointless oscillation. The death penalty (−100) is the strongest signal. The survival bonus (+0.1) encourages staying alive over standing still.

---

## Epsilon — exploration vs exploitation

`epsilon` is the probability of picking a **random** action instead of what the network thinks is best.

- **High epsilon (e.g. 1.0):** fully random — the agent wanders everywhere and collects varied experience
- **Low epsilon (e.g. 0.05):** mostly follows the network — only 5% random

**The bug I made and fixed:** the original code decayed epsilon by `× 0.999` every single frame. At 60fps that meant epsilon hit its minimum (0.05) within the very first game — the agent had barely seen anything. From game 2 onward it was fully greedy with an untrained network, so it kept repeating the same bad behaviour.

**The fix:** decay by `× 0.97` once per game end instead. Epsilon reaches 0.05 after roughly 110 games, giving the agent 100+ games of genuine exploration before it commits to what it's learned.

---

## How training actually works (the Bellman equation)

Every frame:
1. Sample 64 random memories from the buffer
2. For each memory, ask: "what Q-value did the network predict for the action that was taken?"
3. Ask: "what should it have been?" → `reward + 0.99 × (best Q-value in next state)`
4. The gap between 2 and 3 is the error. Update weights to reduce it.

Step 3 is the **Bellman equation** — the core of Q-learning. It says: the value of being in a state is the immediate reward plus a discounted version of the best future value. `gamma = 0.99` means future rewards are almost as valuable as immediate ones, so the agent thinks ahead.

---

## The direction-caching fix

**The problem:** `get_action` was called every single frame. With epsilon=1.0, 20% of frames would randomly pick the direction opposite to current travel, causing an immediate reversal. At 60fps this looked like frantic oscillation.

**The fix:** the agent is only asked for a new direction when Pac-Man reaches a node (intersection) OR every 15 frames — whichever comes first. Between queries the last chosen direction is reused. Mid-tile reversals are still possible (useful for fleeing ghosts) but happen at most once per 15 frames instead of 60 times per second.

---

## Scores list

`agent.scores` is a plain Python list. Every time a game ends, the final score is appended. It's saved inside the `.pth` checkpoint and reloaded on startup, so the full history survives across sessions.

To inspect it:
```python
import torch
data = torch.load("pacman_agent.pth")
print(data["scores"])
```

To plot it (if you have matplotlib):
```python
import matplotlib.pyplot as plt
import numpy as np

scores = data["scores"]
plt.plot(scores, alpha=0.3, label="raw")
window = 20
moving_avg = np.convolve(scores, np.ones(window)/window, mode="valid")
plt.plot(range(window-1, len(scores)), moving_avg, label="20-game avg")
plt.legend()
plt.show()
```
