"""
Visualize training progress and the planned writing trajectory.

Generates results.png with four panels:
  Top-left  : SAC training reward curve (learning progress)
  Top-right : 3-D arm trajectory in the workspace (all waypoints)
  Bot-left  : Policy success rate — fraction of eval episodes that reach target
  Bot-right : Projected Y-Z "written" name (what the LED traces on the canvas)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 (registers 3d projection)
import os

fig = plt.figure(figsize=(16, 11))
fig.suptitle("Arm Writer — Training & Trajectory", fontsize=14, fontweight="bold")

ax_train  = fig.add_subplot(2, 2, 1)
ax_3d     = fig.add_subplot(2, 2, 2, projection="3d")
ax_success = fig.add_subplot(2, 2, 3)
ax_write  = fig.add_subplot(2, 2, 4)

# ── Panel 1: Training reward curve ────────────────────────────────────────────
if os.path.exists("training_log.npz"):
    log     = np.load("training_log.npz")
    rewards = log["ep_rewards"]
    WINDOW  = max(1, len(rewards) // 50)
    smooth  = np.convolve(rewards, np.ones(WINDOW) / WINDOW, mode="valid")

    ax_train.plot(rewards, alpha=0.25, color="steelblue", lw=0.7, label="per-episode")
    ax_train.plot(np.arange(WINDOW - 1, len(rewards)), smooth,
                  color="steelblue", lw=2, label=f"rolling avg (n={WINDOW})")
    ax_train.set_xlabel("Episode")
    ax_train.set_ylabel("Return")
    ax_train.set_title("SAC Training Reward")
    ax_train.legend(fontsize=8)
else:
    ax_train.text(0.5, 0.5, "training_log.npz not found\nRun train.py first",
                  ha="center", va="center", transform=ax_train.transAxes)

# ── Panels 2, 4: Trajectory ────────────────────────────────────────────────────
if os.path.exists("trajectory.npz"):
    traj     = np.load("trajectory.npz")
    led      = traj["led_positions"]     # (T, 3)  xyz
    pen      = traj["pen_down"]           # (T,)    bool
    wps      = traj["waypoints"]          # (W, 3)  target waypoints

    # 3-D plot
    for seg_mask, color, lw in [(pen, "gold", 1.5), (~pen, "gray", 0.5)]:
        idx = np.where(seg_mask)[0]
        if len(idx) == 0:
            continue
        # Split into contiguous segments
        breaks = np.where(np.diff(idx) > 1)[0] + 1
        segs   = np.split(idx, breaks)
        for s in segs:
            ax_3d.plot(led[s, 0], led[s, 1], led[s, 2],
                       color=color, lw=lw, alpha=0.9)

    ax_3d.scatter(wps[:, 0], wps[:, 1], wps[:, 2],
                  s=8, c="crimson", zorder=5, label="waypoints")
    ax_3d.set_xlabel("X (m)")
    ax_3d.set_ylabel("Y (m)")
    ax_3d.set_zlabel("Z (m)")
    ax_3d.set_title("3-D Arm Trajectory  (gold = drawing)")
    ax_3d.legend(fontsize=7)

    # Panel 4: projected Y-Z "written" name
    # Plot only pen-down segments in gold on the writing plane
    for i in range(len(pen) - 1):
        if pen[i] and pen[i + 1]:
            ax_write.plot([led[i, 1], led[i + 1, 1]],
                          [led[i, 2], led[i + 1, 2]],
                          color="gold", lw=3, solid_capstyle="round")

    ax_write.set_aspect("equal")
    ax_write.set_facecolor("#111111")
    ax_write.set_xlabel("Y — lateral (m)")
    ax_write.set_ylabel("Z — height (m)")
    ax_write.set_title("Written Name (Y-Z projection)")
    ax_write.invert_xaxis()   # mirror so the name reads left-to-right
else:
    for ax in [ax_3d, ax_write]:
        ax.text2D(0.5, 0.5, "trajectory.npz not found\nRun plan.py first",
                  ha="center", va="center", transform=ax.transAxes) \
            if hasattr(ax, 'text2D') else \
            ax.text(0.5, 0.5, "trajectory.npz not found\nRun plan.py first",
                    ha="center", va="center", transform=ax.transAxes)

# ── Panel 3: Eval success rate ─────────────────────────────────────────────────
eval_log = "eval_logs/evaluations.npz"
if os.path.exists(eval_log):
    ev      = np.load(eval_log)
    # SB3 eval log has 'results' (n_evals × n_episodes) and 'timesteps'
    success = (ev["results"] > 0).mean(axis=1)   # fraction of episodes with positive return
    ax_success.plot(ev["timesteps"], success, color="seagreen", lw=2)
    ax_success.set_xlabel("Environment steps")
    ax_success.set_ylabel("Success rate")
    ax_success.set_ylim(0, 1)
    ax_success.set_title("Evaluation Success Rate")
    ax_success.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.0%}")
    )
else:
    ax_success.text(0.5, 0.5, "eval_logs/evaluations.npz not found\n"
                               "Produces during train.py",
                    ha="center", va="center", transform=ax_success.transAxes)

plt.tight_layout()
plt.savefig("results.png", dpi=150)
plt.show()
print("Saved → results.png")
