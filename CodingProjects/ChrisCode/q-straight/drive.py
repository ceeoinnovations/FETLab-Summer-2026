"""
Q-learning straight-drive trainer.

Runs the robot forward using the double motor while the IMU tracks yaw.
Each step: observe heading error → choose action (ε-greedy) → apply
motor differential → observe next heading error → compute reward →
update Q-table.

After the run, saves results.npz and prints the learned Q-table.
Run  visualize.py  to generate the full figure.

Press Ctrl+C to stop early — partial results are still saved.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import numpy as np
from config import SERIAL, STEP_DT, RUN_DURATION, ACTION_LABELS, STATE_LABELS
from qlearn import QTable, yaw_to_state, compute_reward, action_to_speeds, N_STATES
from lelib import doubleMotor

# ── Connect ───────────────────────────────────────────────────────────────────
dm = doubleMotor()
print("Connecting to double motor...")
dm.connect(SERIAL)

print("Resetting yaw to 0° — point the robot in the desired direction now.")
time.sleep(1.0)
dm.reset_heading()
time.sleep(0.3)
print(f"Heading zeroed. Running for {RUN_DURATION} s ({int(RUN_DURATION/STEP_DT)} steps).")
print("Ctrl+C to stop early.\n")

# ── Training loop ─────────────────────────────────────────────────────────────
qt = QTable()

history_yaw     = []
history_state   = []
history_action  = []
history_reward  = []
history_epsilon = []

WINDOW = 20  # steps for rolling average reward

print(f"{'Step':>5}  {'ε':>5}  {'Yaw°':>7}  {'State':>3}  {'Action':>6}  {'Rew':>5}  {'Avg':>6}")
print("─" * 52)

step    = 0
t_start = time.time()

try:
    while time.time() - t_start < RUN_DURATION:
        # ── Observe ──────────────────────────────────────────────────────────
        yaw_error = dm.yaw()
        state     = yaw_to_state(yaw_error)

        # ── Act ───────────────────────────────────────────────────────────────
        action      = qt.choose_action(state)
        left, right = action_to_speeds(action)
        dm.movement_move_tank(left, right)

        time.sleep(STEP_DT)

        # ── Observe outcome ───────────────────────────────────────────────────
        next_yaw   = dm.yaw()
        next_state = yaw_to_state(next_yaw)
        reward     = compute_reward(next_yaw)

        # ── Learn ─────────────────────────────────────────────────────────────
        qt.update(state, action, reward, next_state)
        qt.decay_epsilon()

        # ── Log ───────────────────────────────────────────────────────────────
        history_yaw.append(yaw_error)
        history_state.append(state)
        history_action.append(action)
        history_reward.append(reward)
        history_epsilon.append(qt.eps)

        if step % 10 == 0:
            avg = float(np.mean(history_reward[-WINDOW:])) if history_reward else 0.0
            print(f"{step:>5}  {qt.eps:>5.2f}  {yaw_error:>+7.1f}°  "
                  f"{state:>3}  {ACTION_LABELS[action]:>6}  "
                  f"{reward:>+5.1f}  {avg:>+6.2f}")

        step += 1

except KeyboardInterrupt:
    print("\nStopped early.")

finally:
    dm.stop()

# ── Results ───────────────────────────────────────────────────────────────────
elapsed = time.time() - t_start
print(f"\nCompleted {step} steps in {elapsed:.0f} s.\n")

print("Learned Q-table  (★ = greedy action):")
qt.print_table()

visits = np.zeros(N_STATES, dtype=int)
for s in history_state:
    visits[s] += 1
print("\nState visit counts:")
for s in range(N_STATES):
    bar = "█" * min(40, visits[s] // max(1, step // 40))
    print(f"  {STATE_LABELS[s]:<14}  {visits[s]:>4}  {bar}")

np.savez(
    "results.npz",
    Q       = qt.Q,
    yaw     = np.array(history_yaw),
    state   = np.array(history_state),
    action  = np.array(history_action),
    reward  = np.array(history_reward),
    epsilon = np.array(history_epsilon),
)
print("\nSaved → results.npz")
print("Run:    python visualize.py")
