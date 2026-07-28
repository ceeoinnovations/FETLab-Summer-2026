import numpy as np

# ── Hardware ──────────────────────────────────────────────────────────────────
SERIAL = 2279

# Motor to joint mapping:
#   Joint 1 (base yaw)       → singleMotor
#   Joint 2 (shoulder)       → doubleMotor LEFT  motor
#   Joint 3 (elbow flexion)  → doubleMotor RIGHT motor
#
# Gear ratio: motor shaft degrees per radian of joint angle.
# Measure your physical build — these are typical starting values.
GEAR_DEG_PER_RAD = {
    "j1": np.degrees(1) * 3.0,   # 3:1 gearbox on base
    "j2": np.degrees(1) * 5.0,   # 5:1 on shoulder (heavy link)
    "j3": np.degrees(1) * 3.0,   # 3:1 on elbow
}

# ── Simulation ────────────────────────────────────────────────────────────────
ARM_XML     = "arm.xml"
MAX_STEPS   = 150          # max environment steps per episode
SUBSTEPS    = 5            # MuJoCo physics steps per RL action step
SUCCESS_THR = 0.012        # m — LED within this distance = success

# Writing canvas boundaries in 3-D arm workspace (metres).
# Targets are sampled uniformly from this box during training.
CANVAS_X    = 0.23                   # fixed depth (roughly constant during writing)
CANVAS_Y    = (-0.14,  0.14)         # lateral sweep (controlled by j1)
CANVAS_Z    = ( 0.08,  0.20)         # vertical range (controlled by j2+j3)

# ── Name to write ─────────────────────────────────────────────────────────────
NAME = "LEGO"   # change to any string using A–Z

# ── SAC hyperparameters ───────────────────────────────────────────────────────
TOTAL_TIMESTEPS = 400_000
N_ENVS          = 4        # parallel environments
LEARNING_RATE   = 3e-4
BUFFER_SIZE     = 200_000
LEARNING_STARTS = 5_000
BATCH_SIZE      = 256
TAU             = 0.005
GAMMA           = 0.99
POLICY_KWARGS   = dict(net_arch=[256, 256])   # 2-layer MLP

# ── Execution speed ───────────────────────────────────────────────────────────
MOTOR_SPEED     = 30    # % speed for real hardware
WAYPOINT_PAUSE  = 0.3   # seconds to hold each waypoint before moving on
