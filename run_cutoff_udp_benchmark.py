"""Evaluate a lineage checkpoint against the organizer-style Unreal BT client.

The downloaded cutoff executable is a UDP *client*, not a local ActionProvider.
This tool supplies the minimal organizer-server side of the public protocol and
feeds the client PlaneInfo packets from the same JSBSim environment used for
training.  The target provider is called at 60 Hz by DogFightWrapper; the
client's action-repeat=6 therefore preserves its intended 10 Hz BT update rate.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import struct
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np


HERE = Path(__file__).resolve()
LINEAGE = HERE.parents[1]
SOURCE = LINEAGE / "source"
RELEASE = LINEAGE.parents[2]
for _path in (RELEASE / "src", RELEASE, SOURCE):
    _value = str(_path)
    while _value in sys.path:
        sys.path.remove(_value)
    sys.path.insert(0, _value)
os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(SOURCE), str(RELEASE), str(RELEASE / "src"), os.environ.get("PYTHONPATH", "")]
)
os.environ.setdefault("DOGFIGHT_RELEASE_ROOT", str(RELEASE))

from claude_code import evaluation  # noqa: E402
from claude_code.env_utils import make_env  # noqa: E402
from claude_code.model import make_actor_critic  # noqa: E402
from dogfight.ai.action_provider import (  # noqa: E402
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


def _pack_init(rl_state: np.ndarray, bt_state: np.ndarray) -> bytes:
    # Organizer plane IDs are 0=RL ownship and 1=cutoff target in this test.
    rl = [float(value) for value in rl_state]
    bt = [float(value) for value in bt_state]
    return INIT_STRUCT.pack(
        MESSAGE_INIT,
        *rl[0:3],
        *rl[3:6],
        _speed(rl_state),
        *bt[0:3],
        *bt[3:6],
        _speed(bt_state),
    )


class CutoffUDPActionProvider(ActionProvider):
    """Server-side adapter for one external cutoff-client process."""

    def __init__(self, executable: Path, output_dir: Path, worker_id: int):
        self.executable = executable.resolve()
        self.output_dir = output_dir.resolve()
        self.worker_id = int(worker_id)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind(("127.0.0.1", 0))
        self._socket.settimeout(10.0)
        self._server_port = int(self._socket.getsockname()[1])
        self._client_addr: tuple[str, int] | None = None
        self._frame = 0
        self._stdout = (self.output_dir / f"cutoff_worker{worker_id}.stdout.log").open(
            "w", encoding="utf-8", errors="replace"
        )
        self._stderr = (self.output_dir / f"cutoff_worker{worker_id}.stderr.log").open(
            "w", encoding="utf-8", errors="replace"
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
            "--server-port", str(self._server_port),
            "--team-name", f"CutoffEval{worker_id}",
            "--server-timeout-sec", "0",
            "--heartbeat-sec", "0.5",
            "--recv-timeout-sec", "0.2",
            "--action-repeat", "6",
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

    def opponent_kind(self) -> str:
        return "bt_cutoff_unreal_udp"

    def _await_client(self) -> None:
        while True:
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"cutoff client exited during handshake: code={self._process.returncode}"
                )
            data, addr = self._socket.recvfrom(2048)
            if len(data) < MESSAGE_TYPE_STRUCT.size:
                continue
            message_type = MESSAGE_TYPE_STRUCT.unpack_from(data)[0]
            if message_type != MESSAGE_CLIENT_INFO:
                continue
            CLIENT_INFO_STRUCT.unpack(data[: CLIENT_INFO_STRUCT.size])
            self._client_addr = (str(addr[0]), int(addr[1]))
            self._send(SET_PLANE_ID_STRUCT.pack(MESSAGE_SET_PLANE_ID, 1))
            return

    def _send(self, payload: bytes) -> None:
        if self._client_addr is None:
            raise RuntimeError("cutoff client has not completed its UDP handshake")
        self._socket.sendto(payload, self._client_addr)

    def _drain(self) -> None:
        self._socket.setblocking(False)
        try:
            while True:
                self._socket.recvfrom(2048)
        except BlockingIOError:
            pass
        finally:
            self._socket.settimeout(5.0)

    def reset(self, context: ActionContext | None = None) -> None:
        if context is None or context.ownship_state is None or context.target_state is None:
            return
        self._drain()
        self._frame = 0
        # Re-assert plane identity, reset the embedded tree through Init, and
        # announce a running round before the first PlaneInfo pair.
        self._send(SET_PLANE_ID_STRUCT.pack(MESSAGE_SET_PLANE_ID, 1))
        self._send(_pack_init(context.target_state, context.ownship_state))
        self._send(GAME_CONTROL_STRUCT.pack(MESSAGE_GAME_CONTROL, 1))

    def compute_action(self, context: ActionContext) -> ActionResult:
        if context.ownship_state is None or context.target_state is None:
            raise ValueError("cutoff UDP provider requires both aircraft states")
        frame = self._frame
        self._frame += 1
        # From the provider's perspective ownship is cutoff plane 1 and target
        # is the learner's plane 0.
        self._send(_pack_plane_info(frame, 1, context.ownship_state))
        self._send(_pack_plane_info(frame, 0, context.target_state))
        while True:
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"cutoff client exited during episode: code={self._process.returncode}"
                )
            data, _ = self._socket.recvfrom(2048)
            if len(data) < MESSAGE_TYPE_STRUCT.size:
                continue
            message_type = MESSAGE_TYPE_STRUCT.unpack_from(data)[0]
            if message_type in (MESSAGE_SIM_STATE, MESSAGE_CLIENT_INFO):
                continue
            if message_type != MESSAGE_CMD or len(data) < CMD_STRUCT.size:
                continue
            _, plane_id, command_frame, roll, pitch, yaw, throttle = CMD_STRUCT.unpack(
                data[: CMD_STRUCT.size]
            )
            if int(plane_id) != 1 or int(command_frame) != frame:
                continue
            action = clip_action([roll, pitch, yaw, throttle])
            return ActionResult(
                action=action,
                source="cutoff_unreal_udp",
                info={"frame": frame, "server_port": self._server_port},
            )

    def close(self) -> None:
        try:
            if self._client_addr is not None:
                try:
                    self._send(GAME_CONTROL_STRUCT.pack(MESSAGE_GAME_CONTROL, 0))
                except OSError:
                    pass
        finally:
            try:
                self._socket.close()
            finally:
                if self._process.poll() is None:
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=3.0)
                    except subprocess.TimeoutExpired:
                        self._process.kill()
                        self._process.wait(timeout=3.0)
                self._stdout.close()
                self._stderr.close()


def _run_shard(payload: dict) -> list[dict]:
    worker_id = int(payload["worker_id"])
    checkpoint = Path(payload["checkpoint"])
    executable = Path(payload["executable"])
    output_dir = Path(payload["output_dir"])
    seeds = [int(seed) for seed in payload["seeds"]]
    run_config = json.loads(Path(payload["run_config"]).read_text(encoding="utf-8"))

    state, model_kwargs, rms_dict = evaluation.load_snapshot(checkpoint)
    model = make_actor_critic(**model_kwargs)
    model.load_state_dict(state)
    model.eval()
    rms = evaluation.rms_from_dict(rms_dict)

    provider = CutoffUDPActionProvider(executable, output_dir, worker_id)
    scenario = dict(run_config["initial_scenario"])
    env = make_env(
        overrides={
            "target_mode": "rl",
            "randomize_start_side": False,
            "initial_scenario": scenario,
        },
        reward_module=run_config["arguments"]["reward_module"],
        observation_module=run_config["arguments"]["observation_module"],
        reward_overrides=dict(run_config["reward"]),
        replace_reward_config=True,
        runner_index=f"cutoff_eval_{worker_id}",
        env_index=worker_id,
    )
    env._target_action_provider = provider
    results: list[dict] = []
    try:
        for index, seed in enumerate(seeds, 1):
            one = evaluation.play_games(
                env,
                model,
                None if rms is None else rms.mean,
                None if rms is None else rms.var,
                [seed],
                stochastic=False,
                reconstruct=True,
                device="cpu",
            )
            results.extend(one)
            outcome = evaluation._game_outcome(one[0])
            print(
                f"[cutoff-eval w{worker_id}] {index}/{len(seeds)} seed={seed} "
                f"{outcome} hp={one[0]['own_hp']:.3f}/{one[0]['tgt_hp']:.3f} "
                f"end={one[0]['end_condition']}",
                flush=True,
            )
        return results
    finally:
        env.close()


def _split(values: list[int], count: int) -> Iterable[list[int]]:
    return [values[index::count] for index in range(count)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=LINEAGE / "checkpoints" / "iter_2350.pt")
    parser.add_argument("--cutoff-exe", type=Path, default=Path.home() / "Downloads" / "unreal_bt_client.exe")
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=930000)
    parser.add_argument("--output-dir", type=Path, default=LINEAGE / "evaluations" / "cutoff_bt_vs_iter2350_50")
    args = parser.parse_args()
    if args.games <= 0 or args.workers <= 0:
        raise ValueError("games and workers must be positive")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.cutoff_exe.is_file():
        raise FileNotFoundError(args.cutoff_exe)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [args.base_seed + index for index in range(args.games)]
    workers = min(args.workers, args.games)
    payloads = [
        {
            "worker_id": index,
            "checkpoint": str(args.checkpoint.resolve()),
            "executable": str(args.cutoff_exe.resolve()),
            "output_dir": str(args.output_dir.resolve()),
            "run_config": str((LINEAGE / "logs" / "run_config.json").resolve()),
            "seeds": shard,
        }
        for index, shard in enumerate(_split(seeds, workers))
        if shard
    ]
    results: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_run_shard, payload) for payload in payloads]
        for future in as_completed(futures):
            results.extend(future.result())
    summary = evaluation.summarize(results)
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "cutoff_executable": str(args.cutoff_exe.resolve()),
        "cutoff_sha256": "084226A5B41E0DFEE4A3BFF827B0CC1613D6D58CAA41F995D78F569F9C42ABC2",
        "scenario": "official_3_9_uniform",
        "base_seed": args.base_seed,
        "games": args.games,
        "workers": workers,
        "policy": "deterministic_argmax",
        "cutoff_action_repeat": 6,
        "summary": summary,
        "results": results,
    }
    report_path = args.output_dir / "results.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[cutoff-eval] complete", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"[cutoff-eval] report={report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
