"""
Q-table, state/action binning, reward, and motor speed helpers.

Q-learning update rule:
    Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') - Q(s,a)]

The term in brackets is the TD error (temporal-difference error):
    δ = r + γ · max_a' Q(s',a') - Q(s,a)
                 └──── bootstrapped value of next state ────┘

α scales how far we move toward the new estimate each step.
γ controls how much future rewards matter relative to immediate ones.
"""

import numpy as np
from config import (YAW_EDGES, ACTION_DIFFS, ACTION_LABELS, STATE_LABELS,
                    ALPHA, GAMMA, EPS_START, EPS_END, EPS_DECAY,
                    REWARD_ZONES, REWARD_DEFAULT, BASE_SPEED)

N_STATES  = len(YAW_EDGES) + 1   # 9
N_ACTIONS = len(ACTION_DIFFS)     # 5


def yaw_to_state(yaw_deg: float) -> int:
    """Bin a yaw error in degrees to a state index 0–8."""
    return int(np.digitize(yaw_deg, YAW_EDGES))


def compute_reward(yaw_deg: float) -> float:
    """Sparse reward: +1 straight, 0 acceptable, -1 off course."""
    ae = abs(yaw_deg)
    for threshold, reward in REWARD_ZONES:
        if ae < threshold:
            return reward
    return REWARD_DEFAULT


def action_to_speeds(action_idx: int):
    """
    Convert an action index to (left_speed, right_speed) percentages.
    The differential is split evenly so the average speed stays at BASE_SPEED.
    """
    diff  = ACTION_DIFFS[action_idx]
    left  = BASE_SPEED - diff // 2
    right = BASE_SPEED + diff // 2
    return int(left), int(right)


class QTable:
    def __init__(self):
        self.Q   = np.zeros((N_STATES, N_ACTIONS))
        self.eps = EPS_START

    def choose_action(self, state: int) -> int:
        """ε-greedy: explore with probability ε, exploit otherwise."""
        if np.random.random() < self.eps:
            return np.random.randint(N_ACTIONS)
        return int(np.argmax(self.Q[state]))

    def update(self, state: int, action: int, reward: float, next_state: int):
        """Apply one Q-learning update."""
        td_target = reward + GAMMA * float(np.max(self.Q[next_state]))
        td_error  = td_target - self.Q[state, action]
        self.Q[state, action] += ALPHA * td_error

    def decay_epsilon(self):
        self.eps = max(EPS_END, self.eps * EPS_DECAY)

    def greedy_policy(self) -> np.ndarray:
        """Return the greedy action index for every state."""
        return np.argmax(self.Q, axis=1)

    def print_table(self):
        """Pretty-print the Q-table to the terminal."""
        header = f"{'State':<14}" + "".join(f"{a:>8}" for a in ACTION_LABELS)
        print(header)
        print("─" * len(header))
        greedy = self.greedy_policy()
        for s in range(N_STATES):
            row = f"{STATE_LABELS[s]:<14}"
            for a in range(N_ACTIONS):
                mark = "★" if a == greedy[s] else " "
                row += f"{self.Q[s, a]:>+7.2f}{mark}"
            print(row)
