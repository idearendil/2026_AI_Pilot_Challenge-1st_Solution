"""Organizer-protocol adapter for the downloaded Unreal cutoff BT client.

The cutoff binary is a UDP client.  Training workers therefore host one local
UDP server each, stream the JSBSim states at the simulator's native 60 Hz, and
consume one command per frame.  ``action_repeat=6`` keeps the embedded BT at
its validated 10 Hz policy frequency while its last command is applied at
every FDM substep.
"""
from __future__ import annotations

import os
import socket
import struct
import subprocess
from pathlib import Path

import numpy as np

from dogfight.ai.action_provider import (
    ActionContext,
    ActionProvider,
    ActionResult,
    clip_action,
)


MESSAGE_GAME_CONTROL = 0
MESSAGE_INIT = 1
MESSAGE_PLANE_INFO = 2
MESSAGE_SIM_STATE = 4
MESSAGE_CMD = 6
MESSAGE_CLIENT_INFO = 8
MESSAGE_SET_PLANE_ID = 9

MESSAGE_TYPE_STRUCT = struct.Struct("<i")
GAME_CONTROL_STRUCT = struct.Struct("<ib")
INIT_STRUCT = struct.Struct("<i3f3ff3f3ff")
PLANE_INFO_STRUCT = struct.Struct("<iQb3f3f3f")
CLIENT_INFO_STRUCT = struct.Struct("<i30sib")
SET_PLANE_ID_STRUCT = struct.Struct("<ib")
CMD_STRUCT = struct.Struct("<ibQffff")


def _speed(state: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(state[6:9], dtype=np.float64)))


def _pack_plane_info(frame: int, plane_id: int, state: np.ndarray) -> bytes:
    values = [float(value) for value in state]
    return PLANE_INFO_STRUCT.pack(
        MESSAGE_PLANE_INFO,
        int(frame),
        int(plane_id),
        *values[0:3],
        *values[3:6],
        *values[6:9],
    )


def _pack_init(learner_state: np.ndarray, cutoff_state: np.ndarray) -> bytes:
    learner = [float(value) for value in learner_state]
    cutoff = [float(value) for value in cutoff_state]
    return INIT_STRUCT.pack(
        MESSAGE_INIT,
        *learner[0:3],
        *learner[3:6],
        _speed(learner_state),
        *cutoff[0:3],
        *cutoff[3:6],
        _speed(cutoff_state),
    )


class CutoffUDPActionProvider(ActionProvider):
    """Expose one external cutoff-client process as a target provider."""

    def __init__(
        self,
        executable: str | Path,
        output_dir: str | Path,
        worker_id: str | int,
        *,
        action_repeat: int = 6,
        handshake_timeout_sec: float = 15.0,
        command_timeout_sec: float = 5.0,
    ):
        self.executable = Path(executable).resolve()
        if not self.executable.is_file():
            raise FileNotFoundError(
                f"cutoff client executable not found: {self.executable}"
            )
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.worker_id = str(worker_id)
        self.action_repeat = int(action_repeat)
        if self.action_repeat <= 0:
            raise ValueError("cutoff action_repeat must be positive")
        self.handshake_timeout_sec = float(handshake_timeout_sec)
        self.command_timeout_sec = float(command_timeout_sec)
        self._socket: socket.socket | None = None
        self._process: subprocess.Popen | None = None
        self._client_addr: tuple[str, int] | None = None
        self._stdout = None
        self._stderr = None
        self._frame = 0
        self._closed = False
        self._start_client()

    def opponent_kind(self) -> str:
        return "cutoff_unreal_bt_udp"

    def _start_client(self) -> None:
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind(("127.0.0.1", 0))
        self._socket.settimeout(self.handshake_timeout_sec)
        server_port = int(self._socket.getsockname()[1])
        log_stem = f"cutoff_{self.worker_id}_pid{os.getpid()}"
        self._stdout = (self.output_dir / f"{log_stem}.stdout.log").open(
            "a", encoding="utf-8", errors="replace"
        )
        self._stderr = (self.output_dir / f"{log_stem}.stderr.log").open(
            "a", encoding="utf-8", errors="replace"
        )
        creationflags = 0
        if os.name == "nt":
            creationflags = (
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                | getattr(subprocess, "CREATE_NO_WINDOW", 0)
            )
        command = [
            str(self.executable),
            "--server-ip", "127.0.0.1",
            "--server-port", str(server_port),
            "--team-name", f"CutoffTrain{self.worker_id}",
            "--server-timeout-sec", "0",
            "--heartbeat-sec", "0.5",
            "--recv-timeout-sec", "0.2",
            "--action-repeat", str(self.action_repeat),
            "--ownship-force-side", "2",
            "--target-force-side", "1",
        ]
        self._process = subprocess.Popen(
            command,
            cwd=str(self.executable.parent),
            stdin=subprocess.DEVNULL,
            stdout=self._stdout,
            stderr=self._stderr,
            creationflags=creationflags,
        )
        self._await_client()

    def _await_client(self) -> None:
        assert self._socket is not None
        assert self._process is not None
        while True:
            if self._process.poll() is not None:
                raise RuntimeError(
                    "cutoff client exited during handshake: "
                    f"code={self._process.returncode}"
                )
            data, addr = self._socket.recvfrom(2048)
            if len(data) < MESSAGE_TYPE_STRUCT.size:
                continue
            if MESSAGE_TYPE_STRUCT.unpack_from(data)[0] != MESSAGE_CLIENT_INFO:
                continue
            if len(data) < CLIENT_INFO_STRUCT.size:
                continue
            CLIENT_INFO_STRUCT.unpack(data[: CLIENT_INFO_STRUCT.size])
            self._client_addr = (str(addr[0]), int(addr[1]))
            self._send(SET_PLANE_ID_STRUCT.pack(MESSAGE_SET_PLANE_ID, 1))
            self._socket.settimeout(self.command_timeout_sec)
            return

    def _send(self, payload: bytes) -> None:
        if self._socket is None or self._client_addr is None:
            raise RuntimeError("cutoff client has not completed its handshake")
        self._socket.sendto(payload, self._client_addr)

    def _drain(self) -> None:
        assert self._socket is not None
        self._socket.setblocking(False)
        try:
            while True:
                self._socket.recvfrom(2048)
        except BlockingIOError:
            pass
        finally:
            self._socket.settimeout(self.command_timeout_sec)

    def reset(self, context: ActionContext | None = None) -> None:
        if context is None or context.ownship_state is None or context.target_state is None:
            return
        self._drain()
        self._frame = 0
        self._send(SET_PLANE_ID_STRUCT.pack(MESSAGE_SET_PLANE_ID, 1))
        # Provider-ownship is cutoff plane 1; provider-target is learner plane 0.
        self._send(_pack_init(context.target_state, context.ownship_state))
        self._send(GAME_CONTROL_STRUCT.pack(MESSAGE_GAME_CONTROL, 1))

    def compute_action(self, context: ActionContext) -> ActionResult:
        if context.ownship_state is None or context.target_state is None:
            raise ValueError("cutoff UDP provider requires both aircraft states")
        assert self._socket is not None
        assert self._process is not None
        frame = self._frame
        self._frame += 1
        self._send(_pack_plane_info(frame, 1, context.ownship_state))
        self._send(_pack_plane_info(frame, 0, context.target_state))
        while True:
            if self._process.poll() is not None:
                raise RuntimeError(
                    "cutoff client exited during episode: "
                    f"code={self._process.returncode}"
                )
            data, _ = self._socket.recvfrom(2048)
            if len(data) < MESSAGE_TYPE_STRUCT.size:
                continue
            message_type = MESSAGE_TYPE_STRUCT.unpack_from(data)[0]
            if message_type in (MESSAGE_SIM_STATE, MESSAGE_CLIENT_INFO):
                continue
            if message_type != MESSAGE_CMD or len(data) < CMD_STRUCT.size:
                continue
            _, plane_id, command_frame, roll, pitch, yaw, throttle = (
                CMD_STRUCT.unpack(data[: CMD_STRUCT.size])
            )
            if int(plane_id) != 1 or int(command_frame) != frame:
                continue
            return ActionResult(
                action=clip_action([roll, pitch, yaw, throttle]),
                source="cutoff_unreal_udp",
                info={"frame": frame, "action_repeat": self.action_repeat},
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._client_addr is not None:
                try:
                    self._send(GAME_CONTROL_STRUCT.pack(MESSAGE_GAME_CONTROL, 0))
                except OSError:
                    pass
        finally:
            if self._socket is not None:
                self._socket.close()
            if self._process is not None and self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=3.0)
            for stream in (self._stdout, self._stderr):
                if stream is not None:
                    stream.close()


__all__ = ["CutoffUDPActionProvider"]
