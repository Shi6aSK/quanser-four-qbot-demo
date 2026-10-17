"""Minimal tabular Q-learning agent with peer-to-peer policy sharing."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np


# QBot body-mode commands: [forward velocity (m/s), turn rate (rad/s)].
ACTIONS = np.array(
    [
        [0.00, 0.00],
        [0.13, 0.00],
        [0.08, 0.65],
        [0.08, -0.65],
        [0.00, 0.75],
        [0.00, -0.75],
    ],
    dtype=np.float64,
)
ACTION_NAMES = ("stop", "forward", "forward-left", "forward-right", "left", "right")


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def state_key(
    position: Sequence[float],
    yaw: float,
    target: Sequence[float],
    obstacle_circles: Iterable[Tuple[float, float, float]],
    peer_positions: Iterable[Sequence[float]],
) -> tuple[str, float, float]:
    """Discretize local goal and hazard geometry into a compact state."""
    dx = float(target[0]) - float(position[0])
    dy = float(target[1]) - float(position[1])
    distance = math.hypot(dx, dy)
    bearing = wrap_angle(math.atan2(dy, dx) - yaw)

    if distance < 0.25:
        distance_bin = 0
    elif distance < 0.75:
        distance_bin = 1
    elif distance < 1.5:
        distance_bin = 2
    else:
        distance_bin = 3
    bearing_bin = int(((bearing + math.pi) / (2.0 * math.pi)) * 8.0) % 8

    front_blocked = 0
    nearest_clearance = float("inf")
    hazards = list(obstacle_circles) + [
        (float(peer[0]), float(peer[1]), 0.18) for peer in peer_positions
    ]
    for hazard_x, hazard_y, radius in hazards:
        hx = hazard_x - float(position[0])
        hy = hazard_y - float(position[1])
        center_distance = math.hypot(hx, hy)
        clearance = center_distance - radius - 0.18
        nearest_clearance = min(nearest_clearance, clearance)
        relative = wrap_angle(math.atan2(hy, hx) - yaw)
        if clearance < 0.42 and abs(relative) < math.radians(65):
            front_blocked = 1

    return f"{distance_bin}:{bearing_bin}:{front_blocked}", distance, nearest_clearance


class DistributedQAgent:
    def __init__(
        self,
        bot_id: int,
        learning_rate: float = 0.25,
        discount: float = 0.92,
        epsilon: float = 0.35,
        consensus: float = 0.15,
        seed: int = 0,
    ) -> None:
        self.bot_id = bot_id
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.consensus = consensus
        self.rng = random.Random(seed + bot_id)
        self.q_table: Dict[str, np.ndarray] = {}

    def values(self, state: str) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(ACTIONS), dtype=np.float64)
        return self.q_table[state]

    def choose_action(self, state: str, training: bool = True) -> int:
        values = self.values(state)
        if training and self.rng.random() < self.epsilon:
            return self.rng.randrange(len(ACTIONS))
        maximum = float(np.max(values))
        candidates = np.flatnonzero(np.isclose(values, maximum)).tolist()
        return self.rng.choice(candidates)

    def update(
        self,
        state: str,
        action: int,
        reward: float,
        next_state: str,
        terminal: bool,
    ) -> None:
        values = self.values(state)
        future = 0.0 if terminal else float(np.max(self.values(next_state)))
        td_target = reward + self.discount * future
        values[action] += self.learning_rate * (td_target - values[action])

    def blend_peer_values(self, state: str, peer_values: Sequence[float]) -> None:
        if len(peer_values) != len(ACTIONS):
            return
        local = self.values(state)
        peer = np.asarray(peer_values, dtype=np.float64)
        local[:] = (1.0 - self.consensus) * local + self.consensus * peer

    def decay_exploration(self, minimum: float = 0.05, factor: float = 0.94) -> None:
        self.epsilon = max(minimum, self.epsilon * factor)

    def save(self, path: Path) -> None:
        serializable = {state: values.tolist() for state, values in self.q_table.items()}
        path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
