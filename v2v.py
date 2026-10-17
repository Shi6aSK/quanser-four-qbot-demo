"""Small localhost UDP bus used by the four-QBot digital-twin demo."""

from __future__ import annotations

import json
import socket
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class BotMessage:
    """State shared by one QBot with its peers."""

    bot_id: int
    sequence: int
    timestamp: float
    forward: float
    turn: float
    left_wheel_speed: float
    right_wheel_speed: float
    battery_voltage: float


@dataclass(frozen=True)
class RLMessage:
    """Policy information and metrics exchanged by distributed RL agents."""

    bot_id: int
    sequence: int
    timestamp: float
    episode: int
    step: int
    state: str
    q_values: List[float]
    action: int
    reward: float
    distance_to_target: float
    reached_target: bool
    collision: bool


class V2VNode:
    """Non-blocking peer-to-peer UDP endpoint for one simulated robot."""

    def __init__(
        self,
        bot_id: int,
        peer_ids: Iterable[int],
        host: str = "127.0.0.1",
        base_port: int = 20000,
    ) -> None:
        self.bot_id = bot_id
        self.host = host
        self.base_port = base_port
        self.peer_ids = tuple(peer_id for peer_id in peer_ids if peer_id != bot_id)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((host, base_port + bot_id))
        self.socket.setblocking(False)

    def broadcast(self, message: BotMessage) -> None:
        payload = json.dumps(asdict(message), separators=(",", ":")).encode("utf-8")
        for peer_id in self.peer_ids:
            self.socket.sendto(payload, (self.host, self.base_port + peer_id))

    def receive_latest(self) -> Dict[int, BotMessage]:
        """Drain queued datagrams and return the newest message from each peer."""
        latest: Dict[int, BotMessage] = {}
        while True:
            try:
                payload, _ = self.socket.recvfrom(4096)
            except BlockingIOError:
                break

            try:
                message = BotMessage(**json.loads(payload.decode("utf-8")))
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
                continue

            previous = latest.get(message.bot_id)
            if previous is None or message.sequence > previous.sequence:
                latest[message.bot_id] = message
        return latest

    def close(self) -> None:
        self.socket.close()


class RLV2VNode:
    """UDP peer used by an RL agent, including basic link-quality counters."""

    def __init__(
        self,
        bot_id: int,
        peer_ids: Iterable[int],
        host: str = "127.0.0.1",
        base_port: int = 21000,
    ) -> None:
        self.bot_id = bot_id
        self.host = host
        self.base_port = base_port
        self.peer_ids = tuple(peer_id for peer_id in peer_ids if peer_id != bot_id)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((host, base_port + bot_id))
        self.socket.setblocking(False)
        self.sent = 0
        self.received = 0
        self.inferred_lost = 0
        self._last_sequence: Dict[int, int] = {}

    def broadcast(self, message: RLMessage) -> None:
        payload = json.dumps(asdict(message), separators=(",", ":")).encode("utf-8")
        for peer_id in self.peer_ids:
            self.socket.sendto(payload, (self.host, self.base_port + peer_id))
            self.sent += 1

    def receive_latest(self) -> Dict[int, RLMessage]:
        latest: Dict[int, RLMessage] = {}
        while True:
            try:
                payload, _ = self.socket.recvfrom(16384)
            except BlockingIOError:
                break

            try:
                message = RLMessage(**json.loads(payload.decode("utf-8")))
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
                continue

            if message.bot_id == self.bot_id:
                continue
            previous_sequence = self._last_sequence.get(message.bot_id)
            if previous_sequence is not None and message.sequence > previous_sequence + 1:
                self.inferred_lost += message.sequence - previous_sequence - 1
            if previous_sequence is None or message.sequence > previous_sequence:
                self._last_sequence[message.bot_id] = message.sequence
                self.received += 1
                latest[message.bot_id] = message
        return latest

    def close(self) -> None:
        self.socket.close()


def message_is_fresh(message: BotMessage, max_age: float = 0.5) -> bool:
    return time.monotonic() - message.timestamp <= max_age
