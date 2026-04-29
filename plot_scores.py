import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import os

SAVE_PATH = os.path.join(os.path.dirname(__file__), "pacman_agent.pth")
WINDOW = 20          # moving average window size
EPSILON_START = 1.0
EPSILON_DECAY = 0.97 # must match on_game_end() in agent.py
EPSILON_MIN = 0.05

# ── load scores ──────────────────────────────────────────────────────────────
data = torch.load(SAVE_PATH)
scores = data["scores"]

if len(scores) < 2:
    print("Not enough games recorded yet.")
    exit()

games = np.arange(1, len(scores) + 1)

# ── moving average ────────────────────────────────────────────────────────────
kernel = np.ones(WINDOW) / WINDOW
moving_avg = np.convolve(scores, kernel, mode="valid")
avg_games = np.arange(WINDOW, len(scores) + 1)  # x positions for moving avg

# ── find game where epsilon first hits the minimum ────────────────────────────
# epsilon_n = EPSILON_START * EPSILON_DECAY^n <= EPSILON_MIN
epsilon_cross = math.ceil(math.log(EPSILON_MIN / EPSILON_START) / math.log(EPSILON_DECAY))

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("#f8f8f8")
ax.grid(color="white", linewidth=1.2)

# raw scores
ax.plot(games, scores, color="#aaaaaa", linewidth=0.8, alpha=0.7, label="score per game")

# 20-game moving average
ax.plot(avg_games, moving_avg, color="#4a4adb", linewidth=2.5, label=f"moving avg ({WINDOW} games)")

# epsilon → 0.05 line (only draw if it falls within the data range)
if epsilon_cross <= len(scores):
    ax.axvline(x=epsilon_cross, color="#2dba7e", linewidth=1.5, linestyle="--", label="ε → 0.05")
    ax.text(epsilon_cross + 1, ax.get_ylim()[1] * 0.95, "ε → 0.05",
            color="#2dba7e", fontsize=9, va="top")

ax.set_xlabel("game", fontsize=11)
ax.set_ylabel("score", fontsize=11)
ax.set_title("Pac-Man DQN — Training Progress", fontsize=13, fontweight="bold")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False, fontsize=10)

ax.set_xlim(1, len(scores))
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig("scores_plot.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Plot saved to scores_plot.png  ({len(scores)} games, best score: {max(scores)})")
