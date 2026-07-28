"""
Visualize the Q-learning results saved by drive.py.

Generates a 2×2 figure (results.png) with:
  Top-left  : Q-table heatmap — value of every (state, action) pair,
               with the greedy policy marked per state (★)
  Top-right : Learned policy — which action the greedy policy takes
               in each state, plotted as an arrow diagram
  Bot-left  : Heading error (yaw) over the full training run
  Bot-right : Rolling-average reward, showing the learning curve

Can be run any time after drive.py has produced results.npz.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from config import STATE_LABELS, ACTION_LABELS, ACTION_DIFFS

data   = np.load("results.npz")
Q      = data["Q"]
yaw    = data["yaw"]
action = data["action"]
reward = data["reward"]

N_STATES, N_ACTIONS = Q.shape
greedy = np.argmax(Q, axis=1)     # greedy action index per state
steps  = np.arange(len(yaw))

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 11))
fig.suptitle("Q-Learning: Straight Drive with IMU Feedback", fontsize=14, fontweight="bold")

ax_q    = fig.add_subplot(2, 2, 1)   # Q-table heatmap
ax_pol  = fig.add_subplot(2, 2, 2)   # policy diagram
ax_yaw  = fig.add_subplot(2, 2, 3)   # heading trace
ax_rew  = fig.add_subplot(2, 2, 4)   # reward curve

# ── Panel 1: Q-table heatmap ──────────────────────────────────────────────────
vmax = max(abs(Q.max()), abs(Q.min()), 0.1)
im   = ax_q.imshow(Q, aspect="auto", cmap="RdYlGn",
                   vmin=-vmax, vmax=vmax, origin="upper")
plt.colorbar(im, ax=ax_q, label="Q-value", shrink=0.8)

ax_q.set_xticks(range(N_ACTIONS))
ax_q.set_xticklabels(ACTION_LABELS, fontsize=8)
ax_q.set_yticks(range(N_STATES))
ax_q.set_yticklabels(STATE_LABELS, fontsize=8)
ax_q.set_xlabel("Action  (right − left differential)")
ax_q.set_ylabel("State  (yaw error from straight)")
ax_q.set_title("Q-Table  (★ = greedy policy)")

for s in range(N_STATES):
    for a in range(N_ACTIONS):
        star  = "★" if a == greedy[s] else ""
        color = "black" if abs(Q[s, a]) < vmax * 0.6 else "white"
        ax_q.text(a, s, f"{Q[s,a]:+.2f}{star}",
                  ha="center", va="center", fontsize=7, color=color)

# ── Panel 2: Learned policy arrow diagram ─────────────────────────────────────
# For each state, show which direction the greedy policy steers.
# Arrow points LEFT if the correction turns left (positive diff → corrects right drift),
# arrow points RIGHT if the correction turns right.
GOAL_STATE = 4  # index of the "-2..+2°" bin

cmap_pol = plt.cm.coolwarm
ax_pol.set_xlim(-1.5, 1.5)
ax_pol.set_ylim(-0.5, N_STATES - 0.5)
ax_pol.set_yticks(range(N_STATES))
ax_pol.set_yticklabels(STATE_LABELS[::-1], fontsize=8)
ax_pol.set_xticks([])
ax_pol.set_title("Greedy Policy (→ turn direction)")
ax_pol.axvline(0, color="gray", lw=0.8, ls="--")
ax_pol.axhline(N_STATES - 1 - GOAL_STATE, color="gold", lw=2, ls="-", alpha=0.4,
               label="goal state")

for s in range(N_STATES):
    a    = greedy[s]
    diff = ACTION_DIFFS[a]
    y    = N_STATES - 1 - s        # flip so large errors are at edges
    norm = diff / max(ACTION_DIFFS) if diff != 0 else 0
    color = cmap_pol(0.5 + 0.5 * norm)

    if diff == 0:
        ax_pol.plot(0, y, "o", color="green", ms=10, zorder=3)
    else:
        ax_pol.annotate("", xy=(norm, y), xytext=(0, y),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=2))
    ax_pol.text(1.1, y, ACTION_LABELS[a], va="center", fontsize=8)

ax_pol.legend(fontsize=7, loc="lower right")

# ── Panel 3: Heading (yaw) over time ─────────────────────────────────────────
ax_yaw.plot(steps, yaw, lw=0.8, color="steelblue", alpha=0.8, label="yaw error")
ax_yaw.axhline(0,  color="green",  lw=1.5, ls="--", label="target (0°)")
ax_yaw.axhline( 2, color="orange", lw=0.8, ls=":",  alpha=0.6)
ax_yaw.axhline(-2, color="orange", lw=0.8, ls=":",  alpha=0.6, label="±2° goal zone")
ax_yaw.fill_between(steps, -2, 2, color="green", alpha=0.08)
ax_yaw.set_xlabel("Step")
ax_yaw.set_ylabel("Yaw error (degrees)")
ax_yaw.set_title("Heading Error During Training")
ax_yaw.legend(fontsize=8)

# ── Panel 4: Reward (rolling average) ────────────────────────────────────────
WINDOW = 20
rolling = np.convolve(reward, np.ones(WINDOW) / WINDOW, mode="valid")
ax_rew.plot(steps[WINDOW - 1:], rolling, lw=1.5, color="darkorange",
            label=f"rolling avg (n={WINDOW})")
ax_rew.scatter(steps, reward, s=4, color="gray", alpha=0.3, label="per-step reward")
ax_rew.axhline(0, color="black", lw=0.5, ls="--")
ax_rew.axhline(1, color="green", lw=0.8, ls=":", alpha=0.6, label="max reward")
ax_rew.set_ylim(-1.3, 1.3)
ax_rew.set_xlabel("Step")
ax_rew.set_ylabel("Reward")
ax_rew.set_title("Learning Curve  (reward over time)")
ax_rew.legend(fontsize=8)

plt.tight_layout()
plt.savefig("results.png", dpi=150)
plt.show()
print("Saved → results.png")
