# arm-writer

A 3-DOF robotic arm that writes a name in LED light, trained in MuJoCo simulation with deep reinforcement learning.

## What it does

A LEGO arm (base yaw + shoulder + elbow) holds an LED at its tip.  The arm is trained entirely in simulation using Soft Actor-Critic (SAC) RL to reach any point in a writing canvas.  A letter planner then strings waypoints together to trace each letter of a configurable name, and the joint trajectory is replayed on the real LEGO motors.

## Hardware

| Joint | Motion | Motor |
|-------|--------|-------|
| J1 — base yaw | sweeps the arm left/right | `singleMotor` |
| J2 — shoulder | raises/lowers the whole arm | `doubleMotor` LEFT |
| J3 — elbow | extends/retracts the forearm | `doubleMotor` RIGHT |

The LED is wired to the end of the forearm link.

## Files

| File | Purpose |
|------|---------|
| `arm.xml` | MuJoCo model — links, joints, position-servo actuators, LED site |
| `env.py` | Gymnasium environment wrapping MuJoCo — observation, action, reward |
| `letters.py` | 26 uppercase letter stroke definitions + `name_to_waypoints()` |
| `config.py` | Name to write, hardware serial, gear ratios, RL hyperparameters |
| `train.py` | Step 1 — SAC training (run on laptop, ~10 min) |
| `plan.py` | Step 2 — thread policy through letter waypoints, save trajectory |
| `execute.py` | Step 3 — replay trajectory on real LEGO motors |
| `visualize.py` | Plots training curve, 3-D trajectory, and projected written name |
| `lelib.py` | SimpleLE wrapper (singleMotor, doubleMotor, etc.) |

## How to run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure

Edit `config.py`:
- Set `NAME = "LEGO"` (or any A–Z string)
- Set `SERIAL = 1128` to match your LEGO hub's serial number
- Tune `GEAR_DEG_PER_RAD` to match your physical gearbox ratios — see calibration note below

### 3. Train the reaching policy

```bash
python train.py
```

Trains for 400 k environment steps (~10 minutes on a laptop with 4 parallel envs).  Saves:
- `arm_policy.zip` — the trained SAC policy
- `training_log.npz` — per-episode rewards for visualization

The policy learns to move the LED to any (y, z) position in the writing canvas from any starting configuration, guided only by the dense distance reward and a success bonus.

### 4. Plan the writing trajectory

```bash
python plan.py
```

Loads the trained policy and drives the simulated arm through each waypoint in the name's letter strokes.  Saves `trajectory.npz` with the full joint-angle sequence.

### 5. Visualize

```bash
python visualize.py
```

Generates `results.png` with four panels: training reward curve, 3-D arm trajectory, evaluation success rate, and a projected Y-Z view of what the LED draws on the canvas.

### 6. Execute on real hardware

```bash
python execute.py
```

Streams joint-angle deltas to the LEGO motors one waypoint at a time.  The motors are zeroed at startup and commanded by relative-position moves.

## The AI: Soft Actor-Critic (SAC)

SAC is an off-policy, entropy-regularised actor-critic algorithm designed for continuous control:

- **Actor** — a stochastic neural net (2×256 MLP) that outputs a Gaussian distribution over the 3-D action space (target joint angles).  At inference, the mean is used deterministically.
- **Critic** — two Q-networks that estimate the expected return; the minimum of the two is used to suppress overestimation (twin-critic / "clipped double-Q").
- **Entropy bonus** — SAC automatically tunes a temperature parameter α to maintain a target policy entropy.  This encourages exploration without manual schedule tuning and produces smoother, more robust policies.

Training signal:
```
reward = -distance_to_target  (dense, every step)
       + 5.0                  (bonus when LED arrives within 12 mm)
       - 0.002 * |qvel|²      (velocity regularisation → smooth motion)
```

## The environment (`env.py`)

**Observation (12-D):**
```
[j1, j2, j3,              current joint angles (rad)
 j1_dot, j2_dot, j3_dot,  joint velocities (rad/s)
 tx, ty, tz,              target LED position (m)
 ex, ey, ez]              error vector: LED pos − target (m)
```

**Action (3-D, continuous [-1, 1]):**
Normalised target angles for the three position servos, scaled to each joint's physical range before being passed to MuJoCo.

Each episode randomly samples a new target within the writing canvas.  The policy must therefore generalise across the full workspace, not just memorise a fixed trajectory.

## Sim-to-real gap

The biggest challenge in transferring from simulation to the real arm is that the MuJoCo model uses idealised position servos (instant, perfect position tracking), while the real LEGO motors have:

- Finite speed limits
- Gear-ratio multiplication errors
- Mechanical backlash
- Different link lengths and masses

**Mitigation in this code:**
1. `GEAR_DEG_PER_RAD` in `config.py` — calibrate by commanding each motor to a known angle and measuring the actual joint motion.
2. `MOTOR_SPEED` is set conservatively (30%) so the motors have time to settle.
3. `WAYPOINT_PAUSE` (0.3 s) gives each motor time to reach the target before the next command.
4. For a better sim-to-real bridge, add domain randomisation in `env.py` (randomise link lengths ±5%, actuator gains ±10%) during training.

## Letter stroke library (`letters.py`)

Each letter is defined as a list of strokes, where each stroke is a list of normalised (x, y) points with x, y ∈ [0, 1]:

```
x = 0 → left edge of letter cell,  x = 1 → right edge
y = 0 → bottom of letter cell,     y = 1 → top
```

`name_to_waypoints()` tiles letter cells left-to-right across the canvas, converts each (x, y) to real arm coordinates (canvas_y, canvas_z), and inserts `PEN_UP` sentinels between strokes so the arm lifts between disconnected parts of letters (e.g. the crossbar of H is a separate stroke from the two verticals).

## Calibration procedure

1. Run `execute.py` with `NAME = "I"` (a single vertical stroke — simplest letter).
2. Measure the actual LED path with a ruler.
3. If the path is too short, increase `GEAR_DEG_PER_RAD["j2"]` and `["j3"]`; if too long, decrease.
4. Adjust `CANVAS_Y` and `CANVAS_Z` in `config.py` to centre the writing region on your surface.
