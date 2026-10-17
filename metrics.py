"""CSV recording and live plotting for the distributed QBot experiment."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence


FIELDS = (
    "wall_time",
    "episode",
    "step",
    "bot_id",
    "x",
    "y",
    "yaw",
    "pose_source",
    "pose_query_misses",
    "target_x",
    "target_y",
    "distance_to_target",
    "nearest_clearance",
    "state",
    "action",
    "action_name",
    "forward_command",
    "turn_command",
    "reward",
    "episode_reward",
    "reached_target",
    "collision",
    "epsilon",
    "left_wheel_speed",
    "right_wheel_speed",
    "battery_voltage",
    "packets_sent",
    "packets_received",
    "packets_lost",
    "active_peers",
    "mean_message_age_ms",
    "control_period_ms",
)


class MetricsRecorder:
    def __init__(self, root: Path, metadata: Mapping[str, object]) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_dir = root / f"run_{stamp}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        (self.run_dir / "config.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        self._file = (self.run_dir / "telemetry.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDS)
        self._writer.writeheader()
        self.rows = 0

    def write(self, row: Mapping[str, object]) -> None:
        self._writer.writerow({field: row.get(field, "") for field in FIELDS})
        self.rows += 1
        if self.rows % 40 == 0:
            self._file.flush()

    def close(self) -> None:
        self._file.flush()
        self._file.close()


class LiveMetrics:
    def __init__(
        self,
        bot_ids: Iterable[int],
        starts: Mapping[int, Sequence[float]],
        targets: Mapping[int, Sequence[float]],
        obstacles: Iterable[Sequence[float]],
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.bot_ids = tuple(bot_ids)
        self.starts = starts
        self.targets = targets
        self.obstacles = tuple(obstacles)
        self.history = defaultdict(lambda: defaultdict(list))
        if not enabled:
            return

        import matplotlib.pyplot as plt

        self.plt = plt
        plt.ion()
        self.figure, axes = plt.subplots(2, 2, figsize=(11, 8))
        self.map_axis, self.reward_axis, self.distance_axis, self.network_axis = axes.flat
        manager = self.figure.canvas.manager
        if hasattr(manager, "set_window_title"):
            manager.set_window_title("Four-QBot distributed RL metrics")
        self.figure.tight_layout(pad=2.5)
        plt.show(block=False)

    def add(self, bot_id: int, row: Mapping[str, object]) -> None:
        data = self.history[bot_id]
        for field in (
            "wall_time",
            "x",
            "y",
            "episode_reward",
            "distance_to_target",
            "packets_received",
            "packets_lost",
        ):
            data[field].append(float(row[field]))

    def refresh(self) -> None:
        if not self.enabled:
            return
        colors = ("tab:blue", "tab:orange", "tab:green", "tab:red")
        for axis in (
            self.map_axis,
            self.reward_axis,
            self.distance_axis,
            self.network_axis,
        ):
            axis.clear()

        for obstacle in self.obstacles:
            circle = self.plt.Circle(
                (obstacle[0], obstacle[1]), obstacle[2], color="black", alpha=0.45
            )
            self.map_axis.add_patch(circle)

        received, lost = [], []
        for index, bot_id in enumerate(self.bot_ids):
            color = colors[index % len(colors)]
            data = self.history[bot_id]
            self.map_axis.plot(data["x"], data["y"], color=color, label=f"bot {bot_id}")
            target = self.targets[bot_id]
            self.map_axis.scatter(target[0], target[1], marker="*", s=130, color=color)
            self.reward_axis.plot(data["wall_time"], data["episode_reward"], color=color)
            self.distance_axis.plot(
                data["wall_time"], data["distance_to_target"], color=color
            )
            received.append(data["packets_received"][-1] if data["packets_received"] else 0)
            lost.append(data["packets_lost"][-1] if data["packets_lost"] else 0)

        self.map_axis.set(title="Trajectories, obstacles and targets", xlabel="x (m)", ylabel="y (m)")
        self.map_axis.set_aspect("equal", adjustable="box")
        self.map_axis.set_xlim(-2.0, 2.0)
        self.map_axis.set_ylim(-1.8, 1.8)
        self.map_axis.legend(loc="upper left", fontsize=8)
        self.map_axis.grid(True, alpha=0.25)
        self.reward_axis.set(title="Episode reward", xlabel="time (s)", ylabel="reward")
        self.distance_axis.set(title="Distance to target", xlabel="time (s)", ylabel="m")
        x = list(range(len(self.bot_ids)))
        self.network_axis.bar([v - 0.18 for v in x], received, width=0.36, label="received")
        self.network_axis.bar([v + 0.18 for v in x], lost, width=0.36, label="inferred lost")
        self.network_axis.set(title="V2V packets", xticks=x, xticklabels=[str(i) for i in self.bot_ids])
        self.network_axis.legend(fontsize=8)
        self.figure.tight_layout(pad=2.0)
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        self.plt.pause(0.001)

    def close(self) -> None:
        if self.enabled:
            self.plt.ioff()

    def save(self, path: Path) -> None:
        if self.enabled:
            self.refresh()
            self.figure.savefig(path, dpi=150, bbox_inches="tight")
