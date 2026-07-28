# ── Hardware ──────────────────────────────────────────────────────────────────
SERIAL     = 2279
BASE_SPEED = 60    # % forward speed applied to both motors before correction

# ── Training run ──────────────────────────────────────────────────────────────
RUN_DURATION = 180   # seconds  (≈ 360 steps at 0.5 s/step)
STEP_DT      = 0.5   # seconds per Q-learning step

# ── State space: yaw error binned into 9 discrete states ─────────────────────
#
#  Bin edges (degrees).  np.digitize gives index 0…8:
#    0: error < -20°   (strong left drift)
#    1: -20° … -10°
#    2: -10° … -5°
#    3:  -5° … -2°
#    4:  -2° … +2°    ← goal: approximately straight
#    5:  +2° … +5°
#    6:  +5° … +10°
#    7: +10° … +20°
#    8: error > +20°   (strong right drift)
#
YAW_EDGES = [-20, -10, -5, -2, 2, 5, 10, 20]

STATE_LABELS = [
    "< -20°",
    "-20..-10°",
    "-10..-5°",
    " -5..-2°",
    " -2..+2°",   # ← goal
    " +2..+5°",
    " +5..+10°",
    "+10..+20°",
    "> +20°",
]

# ── Action space: differential correction added to right vs left motor ────────
#
#  differential > 0 → right motor faster → robot turns LEFT  (corrects right drift)
#  differential < 0 → left motor faster  → robot turns RIGHT (corrects left drift)
#
#  left_speed  = BASE_SPEED - diff // 2
#  right_speed = BASE_SPEED + diff // 2
#
ACTION_DIFFS  = [-40, -20, 0, 20, 40]
ACTION_LABELS = ["-40%", "-20%", "  0%", "+20%", "+40%"]

# ── Reward function ───────────────────────────────────────────────────────────
#  Sparse reward based on absolute yaw error:
#    |error| <  2° → +1.0  (on target)
#    |error| < 10° →  0.0  (acceptable)
#    |error| ≥ 10° → -1.0  (off course)
REWARD_ZONES = [(2, 1.0), (10, 0.0)]
REWARD_DEFAULT = -1.0

# ── Q-learning hyperparameters ────────────────────────────────────────────────
ALPHA     = 0.3     # learning rate
GAMMA     = 0.95    # discount factor (value of future rewards)
EPS_START = 1.0     # initial exploration rate (fully random)
EPS_END   = 0.05    # minimum exploration rate
EPS_DECAY = 0.98    # multiply ε by this each step (≈ greedy after 150 steps)
