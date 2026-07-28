# q-straight

Teach a LEGO robot to drive straight using **Q-learning** — even though its motion is irregular and lurching rather than smooth. The robot's built-in IMU measures heading drift; the Q-learning algorithm figures out, through trial and error, which motor correction to apply in each situation.

No model of the robot's dynamics is needed. The algorithm learns entirely from experience.

---

## The problem

Most robots with two independently driven wheels drift over time. With irregular "lurching" motion (non-wheel LEGO pieces, uneven terrain, mechanical play), the dynamics are chaotic and hard to model analytically. Q-learning sidesteps the modeling problem entirely: it tries things, observes what happens, and improves its strategy based on the outcomes.

---

## Q-learning

Q-learning is a **model-free reinforcement learning** algorithm. It learns a table of values Q(s, a) — the expected cumulative future reward of taking action *a* in state *s* — purely from experience.

**Update rule:**
```
Q(s, a) ← Q(s, a) + α · [r + γ · max_a' Q(s', a') - Q(s, a)]
```

| Symbol | Name | Meaning |
|---|---|---|
| Q(s, a) | Q-value | Expected future reward from state s, action a |
| α | Learning rate | How much to update toward new information (0.3) |
| γ | Discount factor | How much future rewards matter (0.95) |
| r | Reward | Immediate feedback from this step |
| s' | Next state | State observed after taking action a |
| max Q(s', a') | Bootstrap | Best Q-value available in the next state |

The quantity `r + γ · max Q(s', a') - Q(s, a)` is the **TD error** (temporal-difference error) — how wrong our current estimate was. Each update nudges Q(s, a) toward the truth by a fraction α.

**ε-greedy exploration:** early in training, the robot picks random actions (exploration). Over time ε decays and it increasingly picks the best-known action (exploitation). The right balance is crucial: too much exploitation too early locks in a bad policy; too much exploration wastes time on random actions.

---

## State space (9 bins)

The IMU provides yaw in degrees. Yaw error = current yaw − 0 (reset at the start).

| State | Yaw error | Meaning |
|---|---|---|
| 0 | < −20° | Strong left drift |
| 1 | −20° .. −10° | Significant left drift |
| 2 | −10° .. −5° | Moderate left drift |
| 3 | −5° .. −2° | Slight left drift |
| **4** | **−2° .. +2°** | **On target (goal)** |
| 5 | +2° .. +5° | Slight right drift |
| 6 | +5° .. +10° | Moderate right drift |
| 7 | +10° .. +20° | Significant right drift |
| 8 | > +20° | Strong right drift |

---

## Action space (5 actions)

Actions are differential speed corrections applied on top of `BASE_SPEED`:

```
left_speed  = BASE_SPEED - diff // 2
right_speed = BASE_SPEED + diff // 2
```

| Action | Differential | Effect |
|---|---|---|
| 0 | −40% | Left faster → turns right → corrects left drift |
| 1 | −20% | Gentle right turn |
| 2 | 0% | Straight (no correction) |
| 3 | +20% | Gentle left turn |
| 4 | +40% | Right faster → turns left → corrects right drift |

The average speed stays fixed at `BASE_SPEED` regardless of the correction.

---

## Reward function

Sparse reward based on heading error magnitude:

| Condition | Reward |
|---|---|
| \|error\| < 2° | +1.0 — on target |
| \|error\| < 10° | 0.0 — acceptable |
| \|error\| ≥ 10° | −1.0 — off course |

Sparsity makes the problem harder (no gradient to follow) but produces a cleaner learning curve to observe.

---

## Files

| File | Purpose |
|---|---|
| `lelib.py` | Extended SimpleLE: adds `reset_heading()`, `yaw()`, `gyro_z()` to `doubleMotor` |
| `config.py` | All parameters: bins, reward zones, Q-learning hyperparameters, run duration |
| `qlearn.py` | `QTable` class — choose, update, decay ε; also binning helpers and reward |
| `drive.py` | Main RL loop — runs the robot, builds the Q-table, saves `results.npz` |
| `visualize.py` | Loads `results.npz`, generates `results.png` with 4 panels |
| `requirements.txt` | `numpy`, `matplotlib`, `legoeducation` (all already installed) |

---

## Running it

### 1. Set your serial number

Edit `config.py`:
```python
SERIAL = 1128   # ← your Bluetooth card serial
```

### 2. Train

```bash
python drive.py
```

Place the robot on a straight course, pointed in the direction you want it to drive. The script zeroes the yaw on startup, then runs for `RUN_DURATION` seconds (default 3 minutes).

Live output during training:
```
 Step     ε    Yaw°  St  Action    Rew     Avg
────────────────────────────────────────────────────
    0   1.00   +0.3°   4    -40%   +1.0   +1.00
   10   0.82   +4.7°   5    +20%    0.0   +0.40
   20   0.67   +1.1°   4      0%   +1.0   +0.60
   50   0.36   -3.4°   3    +20%    0.0   +0.55
  100   0.13   +0.8°   4      0%   +1.0   +0.72
```

After the run, the learned Q-table is printed to the terminal with greedy policy marked (★):
```
State           -40%    -20%      0%    +20%    +40%
──────────────────────────────────────────────────
 < -20°        +0.12   +0.08   +0.03   +0.61   +0.83★
-20..-10°      +0.05   +0.04   +0.02   +0.54   +0.74★
...
 -2..+2°       +0.01   +0.03   +0.91★  +0.02   +0.01
...
> +20°         +0.87★  +0.61   +0.02   +0.05   +0.04
```

A well-trained policy should show:
- **Goal state (−2..+2°)**: greedy action = "0%" (go straight)
- **Left-drift states (negative yaw)**: greedy action = positive differential (turn left to correct)
- **Right-drift states (positive yaw)**: greedy action = negative differential (turn right to correct)

### 3. Visualize

```bash
python visualize.py
```

Generates `results.png` with four panels:

**Top-left — Q-table heatmap:** Every Q(s, a) value as a color (green = high, red = low). The ★ marks the greedy policy for each state. Ideally: the greedy policy forms a diagonal pattern — states to the left of center favor right-turn actions, states to the right favor left-turn actions.

**Top-right — Learned policy arrows:** One arrow per state showing which direction the policy steers. Converged policy should show arrows pointing toward the goal state from both sides.

**Bottom-left — Heading trace:** Yaw error at each step over the full run. Early training: large swings. Late training: tighter around 0°.

**Bottom-right — Reward curve:** Rolling 20-step average reward. Should trend upward as the policy improves.

---

## What a good result looks like

After 3 minutes (~360 steps), a well-converged policy:
- Average reward in last 50 steps: > 0.6
- Goal state (±2°) visited frequently
- Q-table shows a clear anti-diagonal gradient: left-drift states have high Q on right-correction actions, right-drift states have high Q on left-correction actions

## Tuning

| Issue | Fix |
|---|---|
| Robot barely changes direction | Increase `ACTION_DIFFS` (e.g., `[-60, -30, 0, 30, 60]`) |
| Robot oscillates wildly | Decrease `ACTION_DIFFS` or increase `STEP_DT` |
| Learning too slow | Increase `ALPHA` (try 0.5) or reduce `EPS_DECAY` (try 0.97) |
| Not enough exploitation at end | Reduce `EPS_DECAY` so ε drops faster |
| State 4 rarely visited | The robot drifts too fast — slow it down with `BASE_SPEED` |
