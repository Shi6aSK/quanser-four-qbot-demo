"""Spawn and control four communicating QBot Platforms in QLabs.

Open QLabs in the QBot Platform Warehouse/Open World before running this file.
Bot 0 is the leader. Bots 1-3 receive its body-velocity command through UDP.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
import sys

import numpy as np
from pal.products.qbot_platform import QBotPlatformDriver
from qvl.multi_agent import MultiAgent

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v2v import BotMessage, V2VNode, message_is_fresh


BOT_IDS = (0, 1, 2, 3)
SPAWN_CONFIG = [
    {"RobotType": "QBP", "ActorNumber": 0, "Location": [-1.3, -1.2, 0.0], "Rotation": [0, 0, 0]},
    {"RobotType": "QBP", "ActorNumber": 1, "Location": [-1.3, -0.4, 0.0], "Rotation": [0, 0, 0]},
    {"RobotType": "QBP", "ActorNumber": 2, "Location": [-1.3,  0.4, 0.0], "Rotation": [0, 0, 0]},
    {"RobotType": "QBP", "ActorNumber": 3, "Location": [-1.3,  1.2, 0.0], "Rotation": [0, 0, 0]},
]


def spawn_bots() -> MultiAgent:
    """Spawn all four actors and launch their workspace/driver RT models."""
    agents = MultiAgent([config.copy() for config in SPAWN_CONFIG])
    print("\nQBot driver assignments:")
    for bot_id in BOT_IDS:
        info = agents.robotsDict[f"QBP_{bot_id}"]
        print(
            f"  bot {bot_id}: driver={info['driverPort']}, "
            f"HIL={info['hilPort']}, lidar={info['lidarPort']}"
        )
    return agents


def connect_drivers(agents: MultiAgent, timeout: float = 8.0) -> dict[int, QBotPlatformDriver]:
    """Connect one Python client to each RT driver started by MultiAgent."""
    deadline = time.monotonic() + timeout
    drivers = {
        bot_id: QBotPlatformDriver(
            mode=1,
            ip="localhost",
            driverPort=agents.robotsDict[f"QBP_{bot_id}"]["driverPort"],
        )
        for bot_id in BOT_IDS
    }

    while time.monotonic() < deadline:
        disconnected = [
            bot_id for bot_id, driver in drivers.items() if not driver._handle.connected
        ]
        if not disconnected:
            break
        for bot_id in disconnected:
            drivers[bot_id]._handle.checkConnection()
        time.sleep(0.05)

    missing = [bot_id for bot_id, driver in drivers.items() if not driver._handle.connected]
    if missing:
        for driver in drivers.values():
            driver.terminate()
        raise RuntimeError(
            f"Could not connect to QBot driver(s) {missing}. "
            "Wait for all RT-model windows to start, then run again."
        )
    for bot_id in BOT_IDS:
        port = agents.robotsDict[f"QBP_{bot_id}"]["driverPort"]
        print(f"Connected to bot {bot_id} on tcpip://localhost:{port}")
    return drivers


def run_demo(agents: MultiAgent, duration: float, rate: float) -> None:
    drivers = connect_drivers(agents)
    nodes = {bot_id: V2VNode(bot_id, BOT_IDS) for bot_id in BOT_IDS}
    last_messages: dict[int, dict[int, BotMessage]] = {bot_id: {} for bot_id in BOT_IDS}
    sequence = 0
    started = time.monotonic()
    next_tick = started

    print("\nV2V online: UDP ports 20000-20003. Press Ctrl+C to stop.")
    try:
        while time.monotonic() - started < duration:
            now = time.monotonic()
            elapsed = now - started

            # Use a gentle S-curve so copied commands are easy to see in QLabs.
            leader_command = np.array(
                [0.10, 0.18 * math.sin(2.0 * math.pi * elapsed / 8.0)],
                dtype=np.float64,
            )

            commands: dict[int, np.ndarray] = {0: leader_command}
            for bot_id in BOT_IDS[1:]:
                received = nodes[bot_id].receive_latest()
                last_messages[bot_id].update(received)
                leader_state = last_messages[bot_id].get(0)
                if leader_state is not None and message_is_fresh(leader_state):
                    commands[bot_id] = np.array(
                        [leader_state.forward, leader_state.turn], dtype=np.float64
                    )
                else:
                    # Stop rather than continue on an old leader command.
                    commands[bot_id] = np.zeros(2, dtype=np.float64)

            # Drain follower telemetry on the leader side as well.
            last_messages[0].update(nodes[0].receive_latest())

            for bot_id in BOT_IDS:
                command = commands[bot_id]
                driver = drivers[bot_id]
                driver.read_write_std(
                    timestamp=elapsed,
                    arm=1,
                    commands=command,
                )
                nodes[bot_id].broadcast(
                    BotMessage(
                        bot_id=bot_id,
                        sequence=sequence,
                        timestamp=time.monotonic(),
                        forward=float(command[0]),
                        turn=float(command[1]),
                        left_wheel_speed=float(driver.wheelSpeeds[0]),
                        right_wheel_speed=float(driver.wheelSpeeds[1]),
                        battery_voltage=float(driver.battVoltage),
                    )
                )

            if sequence % max(1, int(rate)) == 0:
                peers = sorted(last_messages[0])
                print(
                    f"t={elapsed:5.1f}s  leader=({leader_command[0]:+.2f}, "
                    f"{leader_command[1]:+.2f})  leader sees bots={peers}"
                )

            sequence += 1
            next_tick += 1.0 / rate
            time.sleep(max(0.0, next_tick - time.monotonic()))
    except KeyboardInterrupt:
        print("\nStopping on user request.")
    finally:
        # Leave every driver with an explicit zero/disarm command.
        for bot_id, driver in drivers.items():
            try:
                driver.read_write_std(
                    timestamp=time.monotonic() - started,
                    arm=0,
                    commands=np.zeros(2, dtype=np.float64),
                )
            finally:
                driver.terminate()
                nodes[bot_id].close()
        agents.qlabs.close()
        print("All four QBot drivers disarmed and closed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=20.0, help="run time in seconds")
    parser.add_argument("--rate", type=float, default=50.0, help="control/V2V rate in Hz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.duration <= 0 or args.rate <= 0:
        raise SystemExit("--duration and --rate must be positive")
    agents = spawn_bots()
    try:
        # RT models start asynchronously, so let their streams open first.
        time.sleep(2.0)
        run_demo(agents, duration=args.duration, rate=args.rate)
    except Exception:
        # This also handles failures that happen before run_demo enters its loop.
        agents.qlabs.close()
        raise


if __name__ == "__main__":
    main()
