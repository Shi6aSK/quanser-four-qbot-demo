"""Four-QBot distributed tabular RL experiment for Quanser QLabs."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
from qvl.basic_shape import QLabsBasicShape

DEMO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO_DIR))

from distributed_rl import ACTION_NAMES, ACTIONS, DistributedQAgent, state_key
from metrics import LiveMetrics, MetricsRecorder
from run_four_qbots import BOT_IDS, connect_drivers, spawn_bots
from v2v import RLMessage, RLV2VNode


STARTS = {
    0: (-1.35, -1.20, 0.0),
    1: (-1.35, -0.40, 0.0),
    2: (-1.35, 0.40, 0.0),
    3: (-1.35, 1.20, 0.0),
}
TARGETS = {
    0: (1.35, -1.20, 0.0),
    1: (1.35, -0.40, 0.0),
    2: (1.35, 0.40, 0.0),
    3: (1.35, 1.20, 0.0),
}
# Each tuple is x, y, radius. The same geometry feeds the observation function.
OBSTACLES = (
    (-0.35, -1.20, 0.22),
    (0.30, -0.40, 0.22),
    (-0.10, 0.40, 0.22),
    (0.45, 1.20, 0.22),
)


def spawn_obstacles_and_targets(qlabs) -> None:
    shape = QLabsBasicShape(qlabs)
    for index, (x, y, radius) in enumerate(OBSTACLES):
        shape.spawn_id(
            actorNumber=100 + index,
            location=[x, y, 0.20],
            rotation=[0, 0, 0],
            scale=[2.0 * radius, 2.0 * radius, 0.40],
            configuration=shape.SHAPE_CYLINDER,
        )
        shape.set_material_properties(color=[0.12, 0.12, 0.12], roughness=0.85)
        shape.set_enable_dynamics(False)
        shape.set_enable_collisions(True)

    colors = (
        [0.15, 0.45, 1.0],
        [1.0, 0.45, 0.05],
        [0.10, 0.80, 0.25],
        [0.90, 0.10, 0.20],
    )
    for bot_id in BOT_IDS:
        x, y, _ = TARGETS[bot_id]
        shape.spawn_id(
            actorNumber=200 + bot_id,
            location=[x, y, 0.015],
            rotation=[0, 0, 0],
            scale=[0.34, 0.34, 0.03],
            configuration=shape.SHAPE_CYLINDER,
        )
        shape.set_material_properties(color=colors[bot_id], roughness=0.45)
        shape.set_enable_dynamics(False)
        shape.set_enable_collisions(False)


class PoseTracker:
    """Fast wheel odometry with low-rate QLabs ground-truth corrections.

    QLabs transform requests are synchronous. Polling all four actors at every
    control step can saturate that channel, so the control loop uses measured
    wheel speeds and corrects one robot at a time in round-robin order.
    """

    def __init__(self, correction_period: float) -> None:
        self.correction_period = correction_period
        self.poses: Dict[int, tuple[np.ndarray, float]] = {}
        self.query_misses = {bot_id: 0 for bot_id in BOT_IDS}
        self._last_update = time.monotonic()
        self._next_correction = self._last_update
        self._next_bot_index = 0

    def reset(self) -> None:
        self.poses = {
            bot_id: (np.asarray(STARTS[bot_id], dtype=np.float64), 0.0)
            for bot_id in BOT_IDS
        }
        now = time.monotonic()
        self._last_update = now
        self._next_correction = now
        self._next_bot_index = 0

    def update_odometry(self, drivers: Dict[int, object]) -> float:
        now = time.monotonic()
        dt = min(0.5, max(0.0, now - self._last_update))
        self._last_update = now
        radius = float(drivers[BOT_IDS[0]].WHEEL_RADIUS)
        wheel_base = float(drivers[BOT_IDS[0]].WHEEL_BASE)

        for bot_id in BOT_IDS:
            position, yaw = self.poses[bot_id]
            wheel_speeds = np.asarray(drivers[bot_id].wheelSpeeds, dtype=np.float64)
            left, right = float(wheel_speeds[0]), float(wheel_speeds[1])
            forward = radius * (left + right) / 2.0
            turn = radius * (right - left) / wheel_base
            midpoint_yaw = yaw + 0.5 * turn * dt
            updated = position.copy()
            updated[0] += forward * math.cos(midpoint_yaw) * dt
            updated[1] += forward * math.sin(midpoint_yaw) * dt
            self.poses[bot_id] = (updated, (yaw + turn * dt + math.pi) % (2 * math.pi) - math.pi)
        return dt

    def correct_one(self, actors: Dict[int, object]) -> set[int]:
        """Correct at most one robot; a missing QLabs reply is non-fatal."""
        now = time.monotonic()
        if now < self._next_correction:
            return set()

        bot_id = BOT_IDS[self._next_bot_index]
        self._next_bot_index = (self._next_bot_index + 1) % len(BOT_IDS)
        spacing = self.correction_period / len(BOT_IDS)
        self._next_correction = now + spacing

        success, location, rotation, _ = actors[bot_id].get_world_transform()
        if success and np.all(np.isfinite(location)) and np.all(np.isfinite(rotation)):
            self.poses[bot_id] = (
                np.asarray(location, dtype=np.float64),
                float(rotation[2]),
            )
            return {bot_id}

        self.query_misses[bot_id] += 1
        print(
            f"Warning: missed QLabs pose correction for bot {bot_id}; "
            "continuing with wheel odometry."
        )
        return set()


def reset_robots(
    actors: Dict[int, object],
    drivers: Dict[int, object],
    pose_tracker: PoseTracker,
) -> None:
    for bot_id in BOT_IDS:
        drivers[bot_id].read_write_std(
            timestamp=0.0, arm=0, commands=np.zeros(2, dtype=np.float64)
        )
        success = False
        for _ in range(3):
            success, _, _, _ = actors[bot_id].set_transform_degrees(
                location=list(STARTS[bot_id]),
                rotation=[0, 0, 0],
                scale=[1, 1, 1],
                enableDynamics=True,
            )
            if success:
                break
            time.sleep(0.05)
        if not success:
            raise RuntimeError(f"QLabs could not reset bot {bot_id} for the next episode")
    time.sleep(0.5)
    pose_tracker.reset()


def collision_detected(
    bot_id: int,
    position: Sequence[float],
    poses: Dict[int, tuple[np.ndarray, float]],
) -> bool:
    x, y = float(position[0]), float(position[1])
    if abs(x) > 1.85 or abs(y) > 1.65:
        return True
    for obstacle_x, obstacle_y, radius in OBSTACLES:
        if math.hypot(x - obstacle_x, y - obstacle_y) < radius + 0.17:
            return True
    for peer_id, (peer_position, _) in poses.items():
        if peer_id != bot_id and math.hypot(x - peer_position[0], y - peer_position[1]) < 0.31:
            return True
    return False


def run_experiment(args: argparse.Namespace) -> Path:
    agents = spawn_bots()
    actors = {actor.actorNumber: actor for actor in agents.robotActors}
    drivers: Dict[int, object] = {}
    nodes: Dict[int, RLV2VNode] = {}
    recorder = None
    plotter = None
    started = time.monotonic()

    metadata = {
        "algorithm": "independent tabular Q-learning with peer Q-row consensus",
        "episodes": args.episodes,
        "steps_per_episode": args.steps,
        "rate_hz": args.rate,
        "seed": args.seed,
        "starts": STARTS,
        "targets": TARGETS,
        "obstacles": OBSTACLES,
        "actions": ACTIONS.tolist(),
        "v2v_ports": {bot_id: 21000 + bot_id for bot_id in BOT_IDS},
        "pose_estimator": "wheel odometry with round-robin QLabs corrections",
        "ground_truth_correction_period_per_bot_s": args.ground_truth_period,
    }

    try:
        spawn_obstacles_and_targets(agents.qlabs)
        time.sleep(2.0)
        drivers = connect_drivers(agents)
        nodes = {bot_id: RLV2VNode(bot_id, BOT_IDS) for bot_id in BOT_IDS}
        learners = {
            bot_id: DistributedQAgent(bot_id, seed=args.seed) for bot_id in BOT_IDS
        }
        pose_tracker = PoseTracker(args.ground_truth_period)
        recorder = MetricsRecorder(DEMO_DIR / "runs", metadata)
        plotter = LiveMetrics(
            BOT_IDS, STARTS, TARGETS, OBSTACLES, enabled=not args.no_plot
        )
        sequence = 0
        next_plot = time.monotonic()
        print(f"Recording experiment in: {recorder.run_dir}")
        print("RL V2V ports: 21000-21003. Press Ctrl+C for a safe stop.")

        for episode in range(args.episodes):
            reset_robots(actors, drivers, pose_tracker)
            previous_state: Dict[int, str] = {}
            previous_action: Dict[int, int] = {}
            previous_distance: Dict[int, float] = {}
            episode_reward = {bot_id: 0.0 for bot_id in BOT_IDS}
            finished = {bot_id: False for bot_id in BOT_IDS}
            next_tick = time.monotonic()
            previous_tick = next_tick

            for step in range(args.steps):
                tick_started = time.monotonic()
                control_period = tick_started - previous_tick
                previous_tick = tick_started
                pose_tracker.update_odometry(drivers)
                corrected_bots = pose_tracker.correct_one(actors)
                poses = pose_tracker.poses
                incoming = {bot_id: nodes[bot_id].receive_latest() for bot_id in BOT_IDS}

                # Mix the newest peer rows into the local table before acting.
                for bot_id in BOT_IDS:
                    for message in incoming[bot_id].values():
                        learners[bot_id].blend_peer_values(message.state, message.q_values)

                for bot_id in BOT_IDS:
                    position, yaw = poses[bot_id]
                    peers = [pose[0] for peer_id, pose in poses.items() if peer_id != bot_id]
                    current_state, distance, clearance = state_key(
                        position, yaw, TARGETS[bot_id], OBSTACLES, peers
                    )
                    reached = distance < 0.25
                    collision = collision_detected(bot_id, position, poses)

                    reward = 0.0
                    if bot_id in previous_state:
                        progress = previous_distance[bot_id] - distance
                        reward = 8.0 * progress - 0.02
                        if clearance < 0.20:
                            reward -= 0.25
                        if collision:
                            reward -= 5.0
                        if reached:
                            reward += 10.0
                        learners[bot_id].update(
                            previous_state[bot_id],
                            previous_action[bot_id],
                            reward,
                            current_state,
                            terminal=collision or reached,
                        )
                        episode_reward[bot_id] += reward

                    finished[bot_id] = finished[bot_id] or reached or collision
                    action = 0 if finished[bot_id] else learners[bot_id].choose_action(current_state)
                    command = ACTIONS[action]
                    driver = drivers[bot_id]
                    driver.read_write_std(
                        timestamp=time.monotonic() - started,
                        arm=0 if finished[bot_id] else 1,
                        commands=command,
                    )

                    nodes[bot_id].broadcast(
                        RLMessage(
                            bot_id=bot_id,
                            sequence=sequence,
                            timestamp=time.monotonic(),
                            episode=episode,
                            step=step,
                            state=current_state,
                            q_values=learners[bot_id].values(current_state).tolist(),
                            action=action,
                            reward=reward,
                            distance_to_target=distance,
                            reached_target=reached,
                            collision=collision,
                        )
                    )

                    message_ages = [
                        max(0.0, time.monotonic() - message.timestamp) * 1000.0
                        for message in incoming[bot_id].values()
                    ]
                    row = {
                        "wall_time": time.monotonic() - started,
                        "episode": episode,
                        "step": step,
                        "bot_id": bot_id,
                        "x": position[0],
                        "y": position[1],
                        "yaw": yaw,
                        "pose_source": (
                            "ground_truth" if bot_id in corrected_bots else "odometry"
                        ),
                        "pose_query_misses": pose_tracker.query_misses[bot_id],
                        "target_x": TARGETS[bot_id][0],
                        "target_y": TARGETS[bot_id][1],
                        "distance_to_target": distance,
                        "nearest_clearance": clearance,
                        "state": current_state,
                        "action": action,
                        "action_name": ACTION_NAMES[action],
                        "forward_command": command[0],
                        "turn_command": command[1],
                        "reward": reward,
                        "episode_reward": episode_reward[bot_id],
                        "reached_target": int(reached),
                        "collision": int(collision),
                        "epsilon": learners[bot_id].epsilon,
                        "left_wheel_speed": driver.wheelSpeeds[0],
                        "right_wheel_speed": driver.wheelSpeeds[1],
                        "battery_voltage": float(np.asarray(driver.battVoltage).reshape(-1)[0]),
                        "packets_sent": nodes[bot_id].sent,
                        "packets_received": nodes[bot_id].received,
                        "packets_lost": nodes[bot_id].inferred_lost,
                        "active_peers": len(incoming[bot_id]),
                        "mean_message_age_ms": (
                            sum(message_ages) / len(message_ages) if message_ages else float("nan")
                        ),
                        "control_period_ms": 1000.0 * control_period,
                    }
                    recorder.write(row)
                    plotter.add(bot_id, row)
                    previous_state[bot_id] = current_state
                    previous_action[bot_id] = action
                    previous_distance[bot_id] = distance

                sequence += 1
                if time.monotonic() >= next_plot:
                    plotter.refresh()
                    next_plot = time.monotonic() + 0.5
                if all(finished.values()):
                    break
                next_tick += 1.0 / args.rate
                time.sleep(max(0.0, next_tick - time.monotonic()))

            print(
                f"episode {episode + 1}/{args.episodes}: "
                + ", ".join(
                    f"bot {bot_id} reward={episode_reward[bot_id]:+.2f}"
                    for bot_id in BOT_IDS
                )
            )
            for learner in learners.values():
                learner.decay_exploration()
    except KeyboardInterrupt:
        print("\nStopping on user request.")
    finally:
        for bot_id, driver in drivers.items():
            try:
                driver.read_write_std(
                    timestamp=time.monotonic() - started,
                    arm=0,
                    commands=np.zeros(2, dtype=np.float64),
                )
            finally:
                driver.terminate()
        for node in nodes.values():
            node.close()
        if recorder is not None:
            if "learners" in locals():
                for bot_id, learner in learners.items():
                    learner.save(recorder.run_dir / f"q_table_bot_{bot_id}.json")
            if plotter is not None:
                try:
                    plotter.save(recorder.run_dir / "metrics.png")
                except Exception as error:
                    print(f"Could not save the final dashboard image: {error}")
            recorder.close()
        if plotter is not None:
            plotter.close()
        agents.qlabs.close()

    if recorder is None:
        raise RuntimeError("Experiment stopped before the recorder was initialized")
    print(f"Saved {recorder.rows} telemetry rows to {recorder.run_dir}")
    return recorder.run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--steps", type=int, default=220, help="maximum steps per episode")
    parser.add_argument("--rate", type=float, default=10.0, help="control frequency in Hz")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--ground-truth-period",
        type=float,
        default=2.0,
        help="seconds between QLabs pose corrections for each robot",
    )
    parser.add_argument("--no-plot", action="store_true", help="record without live plotting")
    args = parser.parse_args()
    if (
        args.episodes <= 0
        or args.steps <= 0
        or args.rate <= 0
        or args.ground_truth_period <= 0
    ):
        parser.error("episodes, steps, rate, and ground-truth-period must be positive")
    return args


if __name__ == "__main__":
    run_experiment(parse_args())
