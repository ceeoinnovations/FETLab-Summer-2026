"""
Train the offline RL actor (offline_rl_actor_model.py) via TD3+BC on
collected data. Run these first:

    python collect_data.py                                     (human joystick!)
    python calibrate_color.py                                  (tune HSV)
    python generate_pseudo_labels.py data/images data/pseudo_labels.csv
    python check_data_diversity.py                             (worth reading before this)
    python train_offline_rl.py

Builds (state, action, reward, next_state, done) transitions from
labels.csv + pseudo_labels.csv, respecting session_id boundaries (a
transition never crosses from the end of one drive into the start of the
next — those aren't causally connected).

TD3+BC = a twin-critic actor-critic (TD3) plus a behavior-cloning
regularization term keeping the actor from straying too far from the
data's own actions. See config.ALPHA's comment for why that
regularization isn't optional here — an early attempt with a higher ALPHA
diverged (unbounded Q-values, actor collapsing to near-constant extreme
actions). Checkpoints are selected by validation BC error (closest match
to held-out real actions), not just "last step trained," and the final
printed action-std comparison is the most important sanity check to read
before trusting the result — check BOTH directions against the logged
std: much larger means divergence (the critic overestimated something
out-of-distribution), much smaller means collapse (the actor gave up and
settled near one "safe" average action).
"""

import sys
import csv
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

from config import (
    GAMMA, TAU, POLICY_NOISE, NOISE_CLIP, ALPHA, CRITIC_LR, ACTOR_LR,
    CRITIC_WEIGHT_DECAY, BATCH_SIZE, TRAIN_STEPS, EVAL_EVERY, VAL_FRACTION,
    TRAIN_TEST_SPLIT_SEED, ACTION_SCALE, REWARD_NOT_VISIBLE,
)
from offline_rl_actor_model import build_offline_rl_actor, Critic

MODEL_OUT = "offline_rl_actor.pt"


def build_transitions(labels_csv, pseudo_csv):
    with open(labels_csv) as f:
        labels = list(csv.DictReader(f))
    with open(pseudo_csv) as f:
        pseudo = list(csv.DictReader(f))
    if len(labels) != len(pseudo):
        raise SystemExit(f"Row count mismatch between {labels_csv} ({len(labels)}) and "
                          f"{pseudo_csv} ({len(pseudo)}) — regenerate pseudo-labels.")

    compact = np.array([[float(p["cx_norm"]), float(p["cy_norm"]),
                          float(p["area_frac"]), float(p["visible"])] for p in pseudo], dtype=np.float32)
    actions = np.array([[float(l["left_speed"]), float(l["right_speed"])]
                        for l in labels], dtype=np.float32) / 100.0

    by_session = defaultdict(list)
    for i, l in enumerate(labels):
        by_session[l["session_id"]].append(i)

    S, A, R, NS, D = [], [], [], [], []
    for sid, idxs in by_session.items():
        for k in range(len(idxs)):
            i = idxs[k]
            cx, visible = compact[i, 0], compact[i, 3]
            reward = (1.0 - abs(cx)) if visible > 0.5 else REWARD_NOT_VISIBLE
            S.append(compact[i]); A.append(actions[i]); R.append(reward)
            if k < len(idxs) - 1:
                j = idxs[k + 1]
                NS.append(compact[j]); D.append(0.0)
            else:
                NS.append(compact[i]); D.append(1.0)

    return (torch.tensor(np.array(S, dtype=np.float32)),
            torch.tensor(np.array(A, dtype=np.float32)),
            torch.tensor(np.array(R, dtype=np.float32)),
            torch.tensor(np.array(NS, dtype=np.float32)),
            torch.tensor(np.array(D, dtype=np.float32)),
            len(by_session))


def soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tp.data * (1 - tau) + sp.data * tau)


def main(labels_csv, pseudo_csv):
    S, A, R, NS, D, n_sessions = build_transitions(labels_csv, pseudo_csv)
    n = len(S)
    print(f"{n} transitions across {n_sessions} session(s)\n")

    idx = np.random.RandomState(TRAIN_TEST_SPLIT_SEED).permutation(n)
    split = int(n * (1 - VAL_FRACTION))
    tr, val = idx[:split], idx[split:]
    print(f"train: {len(tr)}  val: {len(val)}")

    actor, actor_targ = build_offline_rl_actor(), build_offline_rl_actor()
    q1, q1_targ = Critic(), Critic()
    q2, q2_targ = Critic(), Critic()
    actor_targ.load_state_dict(actor.state_dict())
    q1_targ.load_state_dict(q1.state_dict())
    q2_targ.load_state_dict(q2.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=ACTOR_LR)
    critic_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()),
                                   lr=CRITIC_LR, weight_decay=CRITIC_WEIGHT_DECAY)

    S_tr, A_tr, R_tr, NS_tr, D_tr = S[tr], A[tr], R[tr], NS[tr], D[tr]
    action_clip = ACTION_SCALE  # target-policy-smoothing noise must stay within the actor's own output range

    best_val_mae, best_state, best_step = float("inf"), None, 0
    for step in range(TRAIN_STEPS):
        b = np.random.randint(0, len(S_tr), BATCH_SIZE)
        s, a, r, ns, d = S_tr[b], A_tr[b], R_tr[b].unsqueeze(1), NS_tr[b], D_tr[b].unsqueeze(1)

        with torch.no_grad():
            noise = (torch.randn_like(a) * POLICY_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP)
            na = (actor_targ(ns) + noise).clamp(-action_clip, action_clip)
            target_q = torch.min(q1_targ(ns, na), q2_targ(ns, na)).clamp(-20, 20)
            y = r + GAMMA * (1 - d) * target_q

        critic_loss = F.mse_loss(q1(s, a), y) + F.mse_loss(q2(s, a), y)
        critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), 1.0)
        critic_opt.step()

        if step % 2 == 0:
            pred_a = actor(s)
            q_val = q1(s, pred_a)
            lam = ALPHA / q_val.abs().mean().detach().clamp(min=1e-3)
            actor_loss = -lam * q_val.mean() + F.mse_loss(pred_a, a)
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()
            soft_update(actor_targ, actor, TAU)
            soft_update(q1_targ, q1, TAU)
            soft_update(q2_targ, q2, TAU)

        if (step + 1) % EVAL_EVERY == 0:
            with torch.no_grad():
                val_bc_mae = F.l1_loss(actor(S[val]), A[val]).item()
                val_q = q1(S[val], A[val]).mean().item()
            if val_bc_mae < best_val_mae:
                best_val_mae, best_step = val_bc_mae, step + 1
                best_state = {k: v.clone() for k, v in actor.state_dict().items()}
            print(f"step {step + 1:5d}: critic_loss={critic_loss.item():.4f} "
                  f"actor_bc_mae(val)={val_bc_mae:.4f} avgQ(val)={val_q:.3f}")

    print(f"\nBest checkpoint: step {best_step}, val_bc_mae={best_val_mae:.4f}")
    torch.save({"state_dict": best_state}, MODEL_OUT)
    print(f"Saved -> {MODEL_OUT}")

    actor.load_state_dict(best_state)
    with torch.no_grad():
        pred = actor(S[val]).numpy()
        true = A[val].numpy()
    print(f"\naction std, logged (real) data:  {true.std(axis=0)}")
    print(f"action std, learned actor:      {pred.std(axis=0)}")
    print("\nIMPORTANT — check both directions against the logged std above:")
    print("  much LARGER  -> divergence: the critic overestimated actions outside")
    print("                  the data and the actor is chasing them. Don't deploy.")
    print("  much SMALLER -> collapse: the actor gave up and settled near a single")
    print("                  'safe average' action regardless of state. Don't deploy.")
    print("  reasonably close (same order of magnitude) -> healthy, proceed.")


if __name__ == "__main__":
    default_pseudo = Path(__file__).parent / "data" / "pseudo_labels.csv"
    if len(sys.argv) == 2:
        pseudo_csv = Path(sys.argv[1])
    elif len(sys.argv) == 1:
        pseudo_csv = default_pseudo
        print(f"No arguments given — using default:\n  pseudo_csv = {pseudo_csv}\n")
    else:
        raise SystemExit("Usage: python train_offline_rl.py <pseudo_labels_csv>")

    labels_csv = pseudo_csv.parent / "labels.csv"
    if not labels_csv.exists():
        raise SystemExit(f"Could not find {labels_csv} next to {pseudo_csv}.")
    main(labels_csv, pseudo_csv)
