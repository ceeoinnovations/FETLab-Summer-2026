# LEGO Quadruped — Walking Policy (CPU / SB3 track)

How we trained a crank-driven LEGO quadruped to walk forward in simulation and
deploy the policy to the real robot. This documents the model, the reward-design
journey (what failed, what worked, and why), the files needed to reproduce it,
and the sim-to-real gotchas.

---

## 1. The robot and the models

**Hardware.** A LEGO quadruped whose four legs are each driven by a single
rotating crank through a four-bar linkage — one motor per leg, two "Double Motor"
units (front + back), plus an on-board IMU. The feet trace a stance/swing curve
as each crank rotates continuously (like a wind-up walker, not a servo-per-joint
quadruped). It's slow: ~0.02–0.04 m/s.

**Two simulation models** (MuJoCo):
- **Abstract crank model** (`cpu_train/lego_quad_cpu.xml`): each leg is a single
  capsule that pivots ("rows"). Easy to walk, useful for fast iteration, but
  *not* faithful — a policy trained here transfers poorly.
- **Faithful mesh** (`lego_quad_mesh.xml` = `robot.xml` + floor; also
  `lego_walk_scene.xml`): the real four-bar linkage built from CAD STL meshes,
  the leg loop closed with `<weld>` equality constraints. This is the
  deployment-faithful model and the one all the successful training used
  (train6–15). The training XML and the deploy scene compile to the **identical
  robot** (verified) — they differ only in actuator wrapping.

**Control interface (both models).** 4 velocity actuators (one per gear crank,
`kv=0.5`, `ctrlrange ±12` rad/s), 5 Hz policy (`control_dt=0.2`). The policy
outputs 4 crank-velocity commands in [-1, 1] scaled by `MAX_CRANK_SPEED=12`.

**Observation** (grew as we added mechanisms):
- v6 (26-dim): base lin vel (blind/zeroed at deploy), gyro, projected gravity,
  sin/cos of the 4 crank angles, crank velocities/10, previous action, command.
- v8 (28-dim): **+ phase clock** `sin/cos(2π·t/T)`.
- v15 (30-dim): **+ heading error** `sin/cos(yaw − yaw0)`.

---

## 2. Training stack

- **PPO** (Stable-Baselines3 2.9) on **CPU**, 16 parallel envs (`SubprocVecEnv` +
  `VecMonitor`), 3M steps (~50–100 min depending on node).
- **Gymnasium env** `LegoQuadEnv` (obs / reward / termination / reset) with a
  **deploy wrapper** `LegoQuadDeployEnv` that adds domain randomization (floor
  friction, actuator gain/latency, IMU noise) and a hardware-matched masked obs.
- **Custom two-critic PPO** (`two_critic_ppo.py`) for the barrier reward (§4).
- **HPC / Slurm**, Python env via `uv` (`module load uv`), headless rendering via
  `MUJOCO_GL=egl`.

---

## 3. The reward-design journey

The abstract model walked early (it "rows" trivially). The **faithful mesh would
not walk** — the policy stood still — and fixing that drove the whole project.
Each `train<N>` folder is one experiment; the reward changed one variable at a
time so we could attribute cause.

| ver | change | result |
|-----|--------|--------|
| 6.2 | `track + progress + alive + lateral + upright + smooth + balance + heading`; raised `alive` to stop suicide-collapse | **stands still** — `alive`(1.5) + `track`-at-0 subsidize standing; `progress` too weak |
| 7 / 7.1 | phase-clock gait *reward* (crank-angle phase match) | reward-hackable (spin-in-place farms it); forward-gated fix; still stands |
| 8 | **phase clock in the OBSERVATION** (paper concept 1) + contact-barrier gait reward; **phase-aligned assembled reset** | stable now (no early falls) but **ignores clock**, stands |
| 9 | **multiplicative reward** `R = r_pos·exp(0.2·r_neg)` (paper concept 3) — kills the standing subsidy, and `R ≥ 0` makes falling-to-exit never optimal (no `alive` needed) | stands — subsidy gone but stuck |
| 9.1 | strong *additive* barrier | ineffective (stands, penalty-dragged) |
| 10 | **two-critic PPO** (paper concept 2) — separate value fns for task vs barrier | stands + mild fall drift → **the wall is discovery, not reward** |
| 11 | **reference-state init (RSI)** — pre-roll cranks into a moving trot on ~85% of resets; + realistic command range + narrow track | **trots in place** (clock-match 69%, ~0 net travel) |
| **12** | **translation-dominant reward** `r_pos = 50·clip(v_b[0], 0, cmd)`, **zero at standing** | **WALKS** — ~0.04 m/s, ~1 m/30 s, cold-starts, clean trot, never falls |
| 13 | same, reversed to the robot's **original design direction** (`FWD_SIGN=-1`, RSI flipped) | walks, **much straighter** (~25° vs train12's 65–84°) |
| 14 | stronger heading penalty (weight 1→4) | **negative result** — doesn't straighten a heading-*blind* policy; costs speed |
| **15** | **heading error in the OBS** (`sin/cos(yaw−yaw0)`, 30-dim) + moderate heading penalty → closed-loop steering | net yaw ~7° (vs 25°), speed held ~0.036 m/s, cold-starts, 73% trot. Actively corrects drift. Straight, but **only two legs drive** (back-drive gait). |
| 16 | per-leg net-rotation *floor* (force every crank to spin) | **negative** — the floor kills steering: all four legs forced equal → can't turn. Front legs must be free to slow down in order to steer. |
| 17 | **one-sided** front-back balance `−0.4·max(0, back−front)` + grippy rubber feet (foot μ=2.0) | four-leg-ish but **front-biased** (front \|drive\| 8.7 ≫ back 5.0); steers ~9–11°. One-sided leaves a front-bias loophole. |
| 17.1 | 17 + foot μ→1.4, **surface friction randomized** (0.6–1.8), 48 envs | still front-biased (8.7 vs 4.3) — one-sided allows it; confirms low/varied μ costs no speed (sim-to-real safe). |
| **18** | **two-sided** balance `−0.4·\|back−front\|` + μ=1.4 surface-rand | **EVEN four-leg trot (front 8.2 ≈ back 7.9), fastest (~0.033 m/s — all four push), ~8–10° veer, surface-robust.** Best in sim; on hardware 17.1/19.1 sometimes felt better (see §5). |
| 19 / 19.1 | **clock-FREE gait coordination** (reward the diagonal-trot *structure*, not clock pace; gated off when slow) to stop the hardware front/back phase-lock stall. 19 strong (`barrier_adv_coef=2.0`), 19.1 flexible (0.6) + recovery training | 19 **trots in place** (~0 travel) — strong coordination killed translation. 19.1 walks dead-straight (4° veer) + best heading recovery, but **front-only** (back idle). Coordination strength is the knob. |
| 20 / 20.1 | **LINE-FOLLOWING**: dead-reckoned cross-track in obs (31-dim) + *multiplicative* true-cross penalty (`XT_WEIGHT=8`); 20 front-bias, 20.1 two-sided | **forward walk stalled** (~0 m/s) — the strong cross-track penalty over-constrained gait discovery; and **never returned to the line** (0/3 recovery). |
| 21 | retune: `XT_WEIGHT 8→3`, heading `0.5→1.0`, coord `1.0→0.6` | **forward recovered (~0.02 m/s)** but still **front-only** and still **0/3 line recovery** — the multiplicative cross-track penalty is too weak a driver (25 cm off = only 14% reward cut) and the heading penalty fights the return. |
| **22** | **line-following redesign**: *potential-based* cross-track shaping `+K·(\|X_prev\|−\|X_now\|)` (strong, non-farmable return gradient), **heading penalty dropped**, two-sided balance, fall penalty −10 | even four-leg drive ✅, but forward only ~0.012 m/s and **2/5** return-to-line — the shaping halved the drift (vs 0/3 for 20/21) but wasn't reliable. **Verdict: in-policy line-following isn't worth the speed cost → do it deploy-side (§5).** |
| 23 / 23.1 | **surgical anti-front-sync retrain of 17.1**: FORWARD-GATED diagonal coordination (`barrier_adv_coef=1.5`, only paid while translating → no trot-in-place) + balance that keeps front-bias *but forbids idle back* (`0.5·front ≤ back ≤ front`). 23.1 adds recovery training (broken-phase RSI seeds + mid-episode phase kicks) | **more balanced four-leg gait** (back/front 0.63–0.64 vs 17.1's 0.51); 23.1 **fastest of the project (0.031 m/s)**. BUT front-foot antiphase **unchanged** (corr −0.44 / −0.50 ≈ 17.1's −0.49) — sim did **not** confirm the sync fix. Both are strong walkers; whether the hardware front-sync improves is a hardware-only question. Deploy bundles built for both. |

### Decisive fixes (found by probing the physics, not the paper)

The paper concepts (phase clock, two-critic, multiplicative reward) were
**necessary scaffolding but not sufficient**. The breakthroughs came from
measuring what the robot could actually do:

1. **The reward's "forward" axis was flipped.** The linkage travels in `−v_b[0]`
   for one crank direction and `+v_b[0]` for the other; the reward rewarded
   `+v_b[0]` while the seeded gait went `−v_b[0]`, so moving forward earned ~0.
   Fixed by aligning RSI drive direction with the rewarded direction.
2. **Commanded speeds were physically unreachable.** The linkage tops out ~0.02
   m/s, but the command range was 0.05–0.35 m/s — 2–17× too fast — and the
   velocity-tracking Gaussian (width 0.15) was wider than the achievable speed,
   so it *could not distinguish standing from crawling*. Fixed: `cmd_range =
   (0.01, 0.04)`, narrower tracking.
3. **Reward the outcome, not a proxy.** The velocity-tracking Gaussian paid for
   standing at low commands. Replacing it with a **linear forward-velocity
   reward that is exactly 0 at standing** made forward translation the only way
   to score — the single change (train11→12) that produced walking.
4. **Discovery, not incentive, was the wall.** Even with a correct, standing-free
   reward, PPO never *found* the coordinated gait from a standing start. **RSI**
   (start most episodes already trotting) let it refine a gait instead of
   discovering one cold.
5. **A policy can only control what it observes.** It had yaw *rate* (gyro) but
   no absolute-heading reference, so it could not self-correct drift — more
   heading *penalty* (train14) just slowed it. Adding **heading error to the
   observation** (train15) is the real fix.
6. **An even four-leg gait needs a *two-sided* penalty, not a floor.** train15's
   straight walk used only two legs (back-drive). A per-leg rotation *floor*
   (train16) over-constrains — forcing all four equal removes the left/right
   asymmetry the policy needs to steer. A *one-sided* balance (penalize only
   front-drives-less-than-back, train17) leaves a front-bias loophole (it
   over-drives the front, coasts the back). A **two-sided** `−|back−front|`
   (train18) forces symmetric front/back drive while leaving left/right free, so
   all four legs push *and* it can still steer — and pushing with all four is
   also faster (0.033 vs 0.028 m/s).
7. **Match contact friction to the real feet, then randomize it.** Rubber feet
   grip; bare plastic slides. Setting the foot geoms to μ≈1.4 (contact μ = the
   element-wise max of the two touching geoms) and randomizing the surface μ per
   reset (0.6–1.8) closed a sim-to-real gap that had let back-drive-only gaits
   "cheat" on a too-slippery floor, and makes the policy robust across floors.
8. **A gait reward must not fight what you want to allow.** The original gait
   term rewarded on-clock foot *contact* against the global free-running clock —
   which silently punished slowing, pausing, and steering (all desync the feet
   from the clock). Rewarding the diagonal-trot **coordination** *structure*
   instead (diagonals together, other pairs opposed), clock-free and gated off
   when the cranks are slow (train19), keeps the anti-sync benefit without
   penalising pauses/steering. Its *strength* is the two-critic advantage
   coefficient (`barrier_adv_coef`), not `GAIT_WEIGHT` (which washes out under
   per-stream normalization): 2.0 kills translation, 0.6 walks (front-only), ~1.0
   is the middle.
9. **Position control needs a shaping reward, not a state penalty.** To make the
   robot return to a *line* (not just hold heading), a *multiplicative* penalty
   on cross-track is far too weak (25 cm off cost only ~14% of reward; cranking
   it up crushed forward discovery — train20/21 never returned, 0/3). The fix is
   **potential-based shaping** `+K·(|X_prev|−|X_now|)` that directly rewards
   *closing* the offset — a strong, non-farmable (telescoping) return gradient —
   plus **dropping the heading penalty**, which otherwise fights the heading
   excursion a return maneuver requires (train22). Deployability comes from a
   **dead-reckoned** cross-track estimate in the obs (`∫ crank-progress ·
   sin(heading_err)`), computed identically in sim and on the robot since the
   real robot has no absolute position; the *reward* uses sim ground-truth.

### Reward principles worth reusing

- **Multiplicative task reward** `r_pos·exp(k·r_neg)`: `r_pos ≥ 0` = the thing you
  want (forward translation), `r_neg ≤ 0` = style penalties. Standing → `r_pos≈0`
  → reward≈0; penalties can only *scale down*, never be farmed; every surviving
  step is `≥ 0` so "fall to end the episode" is never optimal. Elegant and it
  removed the need for a hand-tuned `alive` bonus.
- **Two-critic** (paper 2409.15780): route a sharp barrier penalty to its own
  value function; combine `sum(normalized advantage_task, normalized advantage_
  barrier)`. Keeps the barrier from corrupting the task value / triggering
  fall-to-exit.
- **Phase-clock gait prior**: clock `sin/cos(2π·t/T)`, `T=0.72 s`, in the obs;
  reward on-clock foot **contact** (barrier `f_i ≥ −0.6`), not crank angle.
- **Reference-state initialization** for exploration walls.
- **Check the survival floor** before every reward change (`check_reward_floor.py`)
  — a per-step-negative reward for an upright policy causes suicide-collapse.

---

## 4. Files to reproduce a similar project

**Core training (per version `N`):**
- `lego_env<N>.py` — Gymnasium env: obs, reward, termination, RSI reset.
- `lego_env_deploy<N>.py` — deploy wrapper: domain randomization + masked,
  hardware-matched obs.
- `train_deploy<N>.py` — SB3 PPO / `TwoCriticPPO` training script.
- `two_critic_ppo.py` — custom two-critic PPO (policy + rollout buffer + PPO
  subclass) for SB3 2.9.
- `lego_quad_mesh.xml` + `assets/*.stl` — the faithful model.
- `submit_train<N>.sbatch` — Slurm job (17 CPUs = 16 envs + main).

**Diagnostics (built as needed — invaluable):**
- `check_reward_floor.py` — is an upright policy's per-step reward positive?
- `reward_breakdown.py` — per-term reward contribution of a trained policy.
- `symmetry_test.py`, `pin_kinematics.py` — is the mesh left/right symmetric?
- `open_loop_sweep7.py`, `diag_contact.py` — does any open-loop crank pattern
  walk / do the feet cycle contact?
- `probe_phaseclock.py`, `calib_gaitphase.py` — is the trot clock realizable;
  what is the natural contact phase?

**Export / evaluation:**
- `export_policy.py` — SB3 `.zip` → NumPy weights `.npz` (with SB3↔NumPy
  verification). Note: two-critic models need loading via `TwoCriticPPO.load`.
- `numpy_policy.py` — dependency-free NumPy forward pass (used on hardware).
- `eval_straightness.py`, `action_stats.py`, `analyze_final.py` — forward speed,
  yaw drift, per-leg action, clock-match; renders mp4/gif.

**Deployment (hardware, `legoeducation` library):**
- `run_on_robot<N>.py` — the deploy loop: reads IMU + motor encoders, builds the
  obs (incl. phase clock and, for v15, heading error), runs the NumPy policy,
  maps actions to motors, heading-hold, tilt safety.
- `calibrate_motors.py`, `orient_robot.py` — map motors↔legs, find crank
  directions / verify encoders (fixes the flail).
- `scripted_trot_test.py` — open-loop trot; the ground-truth "can the hardware
  walk at all / which direction" check.
- `watch_train<N>.py`, `play_on_lego.py` — live MuJoCo viewer (needs a display).

---

## 5. Deployment & sim-to-real notes

- **Two Double Motor units**, front (red, serial 1779) + back (purple, 6040);
  IMU read via notification callbacks (pitch/roll/yaw in decidegrees, gyro in
  decidegrees/s, accel in milli-g → gravity unit vector).
- **`LEG_MAP`** (which motor is which leg) and **`LEG_SIGNS`** (crank direction).
  The **back unit is mounted flipped**, so its LEFT/RIGHT ports and API CW/CCW
  are reversed vs the robot — the single biggest mapping gotcha.
- **The flail bug:** `LEG_SIGNS` must not cancel the policy's dominant crank
  pattern. The trained trot is `[+,−,+,−]` (train12) / `[−,+,−,+]` (train13);
  `LEG_SIGNS = [+1,+1,+1,+1]` passes it through. An earlier `[-1,+1,-1,+1]`
  multiplied it to all-one-direction → the robot flailed instead of trotting.
- **Direction lives in the policy's actions**, not `LEG_SIGNS` — so the two
  directions (train12 / train13) share the same `LEG_MAP`/`LEG_SIGNS` and differ
  only in the weights file.
- **Phase clock** is reproduced on wall-clock time from start (`phase_t += dt`).
- **Steering: two approaches.** (a) *Heading-hold* — an external differential
  leg-speed trim on IMU yaw error (in run_on_robot12/13); crude, barely helped
  because the policy was heading-blind. (b) *Heading-aware policy* (train15+,
  `run_on_robot15.py`/`run_on_robot18.py`) — feed `sin/cos(yaw−yaw0)` into the obs
  so the policy steers itself; this is the deploy pick. On the real robot, verify
  the **`HEADING_ERR_SIGN`** on a slow first run: sim yaw increases CCW, so if the
  IMU increases the other way the policy steers the wrong way — flip the sign.
- **Deploy scripts, by policy:** `run_on_robot12/13/15/18.py` and
  `run_on_robot17_1.py`. All the v15+ scripts share the identical 30-dim obs /
  `LEG_MAP` / `LEG_SIGNS` and differ only in the weights file; each needs its
  `policy_weights<N>.npz` + `numpy_policy.py`.
- **Hardware walk-off ≠ sim ranking — trust the robot.** In sim, train18 (even
  four-leg) then train19.1 (straightest) looked best. On the *real* robot,
  **train17.1 walks best** (front-biased but both pairs drive; beats 19.1, and 18
  didn't win either). The sim metrics we optimised — veer angle, artificial-
  perturbation recovery — didn't predict real walking, which is governed by
  sustained propulsion from all four legs under real friction/motor-lag. 17.1 is
  the deploy base and `run_on_robot17_1.py` carries all the deploy-side fixes
  below.
- **Bring-up order on hardware:** `orient_robot.py` (motors alive + `LEG_MAP`) →
  `scripted_trot_test.py` (does open-loop walk, which direction) → `run_on_robot`
  starting at low `MAX_SPEED_PCT` to confirm direction before full speed.
- **The robot walks straighter in its original design direction** (train13,
  ~25° veer) than reversed (train12, 65–84°) — foot geometry rectifies better the
  way it was built.
- **Path-following is a deploy-side outer loop, not an in-policy objective.**
  In-policy line-following (train20–22) always cost forward speed and never
  reliably returned to the line. Instead, `run_on_robot17_1.py`'s `PATH_FOLLOW`
  wraps the heading-hold walker with pure-pursuit: dead-reckon the cross-track
  (`xt += fwd-progress·sin(heading-err)` — the robot has no absolute position)
  and bend the heading target toward the line (`he_eff = he + K·xt`). The policy
  nulls `he_eff`, so it returns to the line. No retrain; tune `K` and flip
  `PATH_SIGN` live. (See decisive-fix #9.)
- **Bring-up order on hardware:** `orient_robot.py` (motors alive + `LEG_MAP`) →
  `scripted_trot_test.py` (does open-loop walk, which direction) → `run_on_robot`
  starting at low `MAX_SPEED_PCT` to confirm direction before full speed.
- **The robot walks straighter in its original design direction** (train13,
  ~25° veer) than reversed (train12, 65–84°) — foot geometry rectifies better the
  way it was built.

### Deploy-side fixes for hardware gait pathologies (in `run_on_robot17_1`+)

Three failure modes showed up only on the real robot; all three are fixed in the
control loop, behind flags, without retraining:

- **Launch from a trot, not all-feet-down (`PREROLL_ENABLE`).** The anchor step
  puts every crank at the same phase (all feet lowest) = the degenerate all-in-
  phase state. A short open-loop pre-roll in the trot pattern (`RSI_SIGNS`) spins
  the cranks into a moving trot before the policy engages, launching on-
  distribution (mimics training RSI).
- **Front/back phase-lock watchdog (`WATCHDOG_ENABLE`).** A front (or back) pair
  can lock into *co-rotation* — both feet doing the same thing at once — and the
  robot bobs without translating. The loop watches each pair's crank-angle deltas
  and, on sustained co-rotation, injects a few open-loop trot-drive steps to break
  it. (Detects crank *rotation* co-rotation; a *foot-phase* sync where the cranks
  still counter-rotate slips past it — that's a training-side gait-quality issue.)
- **Slave the gait clock to the cranks (`CLOCK_FROM_CRANKS`).** The obs clock
  free-runs on wall-clock time, but the sim cranks tracked it perfectly while
  hardware cranks lag — so the clock and the real cadence **beat**, producing a
  walk-a-few-steps / stall-a-while rhythm (diagnosed by `wd` staying 0 through the
  stalls). Advancing the clock by *measured* crank progress locks it to the
  cranks. **Clamp it** to `[0.4, 1.0]·dt`: the policy trained on a clock that
  always advanced ~dt and never froze, so a frozen/lurching clock is off-
  distribution and triggers erratic full-speed bursts.

The remaining hardware issue — the front pair intermittently syncing so it stops
stepping forward — is **not** deploy-fixable: a sim probe showed 17.1's front feet
are only weakly antiphase (foot-z correlation −0.49 vs ~−1 for a clean trot), so
the sync is baked into the gait, and the watchdog's crank-rotation detector can't
see the foot-phase collapse. That is what **train23 / 23.1** retrain to fix at the
source (forward-gated diagonal coordination, so the front feet are held antiphase)
— see the reward-journey table.

---

## 6. Environment / infra

- Python 3.13, `mujoco` 3.10, `stable-baselines3` 2.9, `gymnasium`, `numpy`.
- Hardware: `legoeducation` (its own Python env on the control machine).
- `uv` for envs; on HPC `module load uv/0.9.18` and set `UV_CACHE_DIR`.
- Headless rendering: `MUJOCO_GL=egl uv run --with imageio --with imageio-ffmpeg …`.

---

## 7. Running it — HPC (Slurm) and local CPU

Everything runs on **CPU** via `uv` (no GPU). A run lives in one `train<N>/`
directory and is fully self-contained (env, deploy env, `two_critic_ppo.py`,
train script, `.sbatch`, the mesh XML, `numpy_policy.py`). Always launch from
inside that directory.

### 7a. On the HPC (Slurm)

One-time per shell: `module load uv/0.9.18`.

```sh
cd cpu-projects/tasks/walk-straight/train<N>/
sbatch submit_train<N>.sbatch          # queue the job
squeue -u $USER                        # is it PENDING / RUNNING, on which node?
tail -f slurm_<jobid>.out              # live progress (fps, total_timesteps, ep_rew_mean)
sacct -j <jobid> --format=JobID,State,Elapsed   # COMPLETED / RUNNING / TIMEOUT?
scancel <jobid>                        # kill it
```

The `.sbatch` header is the whole config; the body just does
`module load uv` → export cache vars → `cd "$SLURM_SUBMIT_DIR"` →
`uv run python train_deploy<N>.py`. Key `#SBATCH` lines and why:

- `--cpus-per-task=49` — 48 parallel envs (`N_ENVS` in the train script) + 1 main
  process. Match these two: `cpus-per-task = N_ENVS + 1`.
- `--exclusive` — take a **whole clean node** (64-core sapphirerapids). Removes the
  memory-bandwidth contention that otherwise drops fps from ~1000 to ~300. Costs a
  short wait for a free node. **Drop `--exclusive`** (keep `--cpus-per-task=49`) to
  start immediately on any node with 49 free cores, at the cost of possible
  contention — good for a quick run, bad for a timed comparison.
- `--time=05:00:00`, `--mem=16G`, `--exclude=pax166,pax017` (flaky nodes),
  `--output=slurm_%j.out`.
- The body exports `UV_CACHE_DIR=/tmp/uv_cache_${SLURM_JOB_ID}` (keeps uv's cache
  off the home quota), `UV_LINK_MODE=copy`, `PYTHONUNBUFFERED=1` (so the progress
  table streams to the log), and `cd "$SLURM_SUBMIT_DIR"` (Slurm copies the script
  to a spool dir, so `$0`-relative paths break — always cd back).

A clean 3M-step run is ~35–40 min on an exclusive node (~1000 fps), longer if
contended. Checkpoints land in `checkpoints/` every 100k steps, and the final
policy is `lego_quad_deploy_ppo<N>.zip`.

### 7b. On a local CPU (no Slurm)

The training is just a Python script — Slurm only schedules it. To run anywhere
with `uv` installed (laptop/workstation), from inside `train<N>/`:

```sh
export UV_CACHE_DIR=/tmp/uv_cache          # optional, avoids home-dir cache spam
uv run python train_deploy<N>.py           # trains, writes lego_quad_deploy_ppo<N>.zip
```

`uv run` auto-creates the env from the project's lockfile on first use (mujoco,
SB3, gymnasium, numpy) — no manual `pip install`. The one thing to change for a
smaller machine is **`N_ENVS`** at the top of `train_deploy<N>.py`: set it to
about **(physical cores − 1)**. It's 48 for the HPC nodes; on an 8-core laptop use
~7. Fewer envs → proportionally slower wall-clock (same learning), so a 3M-step
run that's ~40 min on 48 envs is a few hours on a laptop — drop `TOTAL_STEPS` to
~1M for a quick smoke test.

After training, export and evaluate the same way in either environment:

```sh
uv run python export_policy.py             # .zip -> policy_weights<N>.npz (+ SB3<->NumPy check)
MUJOCO_GL=egl uv run --with imageio python <render script>   # optional GIF (headless)
```

The deploy bundle to copy to the robot is always the three files
`run_on_robot<N>.py`, `numpy_policy.py`, `policy_weights<N>.npz` — and the robot
side needs its own `pip install legoeducation numpy`, not `uv`.
