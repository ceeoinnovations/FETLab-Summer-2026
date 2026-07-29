# LEGO quadruped — CPU training bundle (train17.1 + train23.1)

Self-contained CPU/SB3 training for the two deployable policies. See
WALKING_POLICY.md section 7 for full run instructions. Quick start (needs `uv`):

    cd train17.1            # or train23.1
    uv run python train_deploy17.py     # trains -> lego_quad_deploy_ppo17.zip

`uv` builds the env from ../pyproject.toml + ../uv.lock (CPU torch, no GPU).
Lower N_ENVS in train_deploy*.py to ~(cores-1) on a small machine.
Deploy to the robot: copy run_on_robot<N>.py + numpy_policy.py +
policy_weights<N>.npz (see WALKING_POLICY.md section 5). Meshes are in assets/.
