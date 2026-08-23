# -*- coding: utf-8 -*-
"""`unreal_bt_client.exe`(외부 UDP 클라이언트)를 env 의 target(상대) ActionProvider 로
쓰기 위한 로컬 UDP 서버 브리지.

배경
----
`unreal_bt_client.exe` 는 `run_unreal_inference.py --mode bt` 를 C++ 로 다시 구현한
**UDP 클라이언트**다(BT 룰이 exe 안에 내장). 혼자서는 못 싸우고, PlaneInfo 를 뿌리고
CMD 를 받아주는 **서버**(=실제 언리얼 게임 서버 역할)가 필요하다.

이 모듈은 그 서버 역할을 아주 얇게 흉내 내되, **물리/데미지/WEZ/종료 판정은 우리
JSBSim env** 가 그대로 담당하게 한다. 즉 exe 는 오직 "상대기 조종 커맨드"만 공급하는
`ActionProvider` 로 동작한다(= 학습/파워테스트의 BTActionProvider 와 같은 자리). 그래서
`power_test` 의 env·판정·통계 코드를 그대로 재사용할 수 있다.

동작
----
1. (127.0.0.1, port) 에 UDP 서버 소켓을 bind 하고 exe 를 그 port 로 접속시킨다(worker
   마다 다른 port → 병렬 가능).
2. handshake: exe 의 heartbeat 로 exe 주소를 알아낸 뒤 SetPlaneID 를 보내 exe 가
   자기 plane_id 를 인지하게 한다(ClientJoinInfo echo 로 확인).
3. env 가 substep 마다 `compute_action(context)` 를 부르면:
     - context.ownship_state = 상대기(=exe 가 조종할 기체), context.target_state = 본 기체.
     - 두 기체의 PlaneInfo 를 만들어 exe 에 보낸다(own=상대기, enemy=본기체).
     - exe 가 내장 BT 로 계산해 돌려준 CMD 를 4-D action 으로 반환한다.

좌표/단위 규약은 `dogfight.unreal.policies.plane_info_to_state` 의 **역변환**이다:
  position: N,E,D(state) → x=N, y=E, z=-D(Up+)   (plane_info_to_state 가 z 를 뒤집으므로 역)
  rotation: roll,pitch,yaw(deg) 그대로
  velocity: state[6:9] 그대로 (exe BT 는 |velocity| 만 사용하므로 프레임 무관)

CMD 규약: exe 는 native BT 의 (RollCMD, PitchCMD, RudderCMD, Throttle) 을 그대로 CMD 로
보낸다. throttle 은 [0,1], 나머지는 [-1,1] — env 의 target action 규약과 동일하다
(claude_code.self_play.SelfPlayProvider 가 policy_action_to_command 로 만드는 값과 같은 규약).
"""
from __future__ import annotations

import atexit
import socket
import subprocess
import time
from pathlib import Path

import numpy as np

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult, clip_action
from dogfight.sim.state_schema import StateIndex
from dogfight.unreal.protocol import (
    CLIENT_JOIN_INFO_STRUCT,
    CMD_STRUCT,
    INIT_STRUCT,
    MESSAGE_TYPE_STRUCT,
    PLANE_INFO_STRUCT,
    SET_PLANE_ID_STRUCT,
    SIMULATION_STATE_STRUCT,
    MessageType,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXE = ROOT / "unreal_bt_client.exe"


# ── 패킷 pack/unpack (protocol.py 는 클라이언트 방향만 제공 → 서버 방향을 여기서 정의) ──
def _pack_set_plane_id(plane_id: int) -> bytes:
    return SET_PLANE_ID_STRUCT.pack(int(MessageType.MT_SetPlaneID), int(plane_id))


def _pack_plane_info(index: int, plane_id: int, pos, rot, vel) -> bytes:
    return PLANE_INFO_STRUCT.pack(
        int(MessageType.MT_PlaneInfo),
        int(index),
        int(plane_id),
        float(pos[0]), float(pos[1]), float(pos[2]),
        float(rot[0]), float(rot[1]), float(rot[2]),
        float(vel[0]), float(vel[1]), float(vel[2]),
    )


def _pack_init(p1_loc, p1_rot, p1_speed, p2_loc, p2_rot, p2_speed) -> bytes:
    return INIT_STRUCT.pack(
        int(MessageType.MT_Init),
        float(p1_loc[0]), float(p1_loc[1]), float(p1_loc[2]),
        float(p1_rot[0]), float(p1_rot[1]), float(p1_rot[2]),
        float(p1_speed),
        float(p2_loc[0]), float(p2_loc[1]), float(p2_loc[2]),
        float(p2_rot[0]), float(p2_rot[1]), float(p2_rot[2]),
        float(p2_speed),
    )


def _msg_type(buffer: bytes):
    if len(buffer) < MESSAGE_TYPE_STRUCT.size:
        return None
    return MessageType(MESSAGE_TYPE_STRUCT.unpack_from(buffer)[0])


def _unpack_cmd(buffer: bytes):
    # CMD_STRUCT = "<ibQffff" → (msg, plane_id, index, roll, pitch, yaw, throttle)
    u = CMD_STRUCT.unpack(buffer[: CMD_STRUCT.size])
    return {"plane_id": int(u[1]), "index": int(u[2]),
            "roll": float(u[3]), "pitch": float(u[4]),
            "yaw": float(u[5]), "throttle": float(u[6])}


def _unpack_client_plane_id(buffer: bytes):
    # CLIENT_JOIN_INFO_STRUCT = "<i30sib" → (msg, team[30s], ai_type, plane_id)
    u = CLIENT_JOIN_INFO_STRUCT.unpack(buffer[: CLIENT_JOIN_INFO_STRUCT.size])
    return int(u[3])


# ── state → PlaneInfo 필드 ────────────────────────────────────────────────────
def _state_to_plane_fields(state):
    s = np.asarray(state, dtype=np.float64)
    pos = (float(s[StateIndex.N]), float(s[StateIndex.E]), float(-s[StateIndex.D]))  # z=Up+
    rot = (float(s[StateIndex.ROLL]), float(s[StateIndex.PITCH]), float(s[StateIndex.YAW]))
    vel = (float(s[6]), float(s[7]), float(s[8]))  # exe BT 는 |vel| 만 사용
    return pos, rot, vel


_SAFE_ACTION = np.array([0.0, 0.0, 0.0, 0.7], dtype=np.float32)  # 실패 시 직진·중간추력


class UnrealExeProvider(ActionProvider):
    """실행 중인 unreal_bt_client.exe 를 상대기 조종기로 브리지하는 provider."""

    def __init__(
        self,
        exe_path: str | Path = DEFAULT_EXE,
        port: int = 9999,
        *,
        own_plane_id: int = 1,       # exe 가 조종하는 기체(=상대기) 의 plane_id
        enemy_plane_id: int = 0,     # 본 기체(RL) 의 plane_id
        ownship_force_side: int = 1,  # exe 관점: 자기(상대기) force side
        target_force_side: int = 2,   # exe 관점: 적(본기체) force side
        team_name: str = "UnrealBT",
        launch: bool = True,
        cwd: str | Path = ROOT,
        step_timeout_sec: float = 0.5,
        handshake_timeout_sec: float = 20.0,
        recv_chunk_sec: float = 0.1,
        max_resends: int = 3,
        max_relaunches: int = 5,
        quiet: bool = True,
        confidence: float = 0.85,
    ):
        self.exe_path = str(Path(exe_path))
        self.port = int(port)
        self.own_plane_id = int(own_plane_id)
        self.enemy_plane_id = int(enemy_plane_id)
        self.ownship_force_side = int(ownship_force_side)
        self.target_force_side = int(target_force_side)
        self.team_name = team_name
        self.cwd = str(cwd)
        self.step_timeout_sec = float(step_timeout_sec)
        self.handshake_timeout_sec = float(handshake_timeout_sec)
        self.recv_chunk_sec = float(recv_chunk_sec)
        self.max_resends = int(max_resends)
        self.max_relaunches = int(max_relaunches)
        self.quiet = bool(quiet)
        self.confidence = float(confidence)

        self._sock: socket.socket | None = None
        self._exe_addr: tuple[str, int] | None = None
        self._proc: subprocess.Popen | None = None
        self._idx = 0
        self._fail_count = 0
        self._relaunch_count = 0     # 지금까지 exe 를 다시 띄운 횟수(상한 max_relaunches)
        self._dead = False           # 재실행 상한 초과 → 영구 포기(이후 즉시 SAFE_ACTION, 스톨 없음)
        self._closed = False

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind(("127.0.0.1", self.port))
        self._sock.settimeout(self.recv_chunk_sec)
        atexit.register(self.close)

        if launch:
            self._launch_exe()
        self._handshake()

    # ── 프로세스/handshake ────────────────────────────────────────────────────
    def _launch_exe(self) -> None:
        args = [
            self.exe_path,
            "--server-ip", "127.0.0.1",
            "--server-port", str(self.port),
            "--team-name", self.team_name,
            "--ai-type", "rule",
            "--action-repeat", "1",       # env 가 substep 마다 부르므로 exe 는 매 pair 재계산
            "--heartbeat-sec", "0.5",
            "--recv-timeout-sec", "0.05",
            "--command-delay-sec", "0",
            "--ownship-force-side", str(self.ownship_force_side),
            "--target-force-side", str(self.target_force_side),
        ]
        out = subprocess.DEVNULL if self.quiet else None
        self._proc = subprocess.Popen(args, cwd=self.cwd, stdout=out, stderr=out)

    def _handshake(self) -> None:
        """exe heartbeat 로 주소를 얻고 SetPlaneID 를 인지시킨다(ClientJoinInfo echo 확인)."""
        assert self._sock is not None
        deadline = time.time() + self.handshake_timeout_sec
        last_setid = 0.0
        while time.time() < deadline:
            # 주소를 알면 SetPlaneID 를 주기적으로 재전송(exe 가 늦게 떠도 잡히게).
            if self._exe_addr is not None and time.time() - last_setid > 0.2:
                self._sock.sendto(_pack_set_plane_id(self.own_plane_id), self._exe_addr)
                last_setid = time.time()
            try:
                buf, addr = self._sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                break
            if self._exe_addr is None:
                self._exe_addr = addr
                continue
            mt = _msg_type(buf)
            if mt == MessageType.MT_ClientInfo:
                if _unpack_client_plane_id(buf) == self.own_plane_id:
                    if not self.quiet:
                        print(f"[UnrealExeProvider] handshake ok (port={self.port}, "
                              f"plane_id={self.own_plane_id})", flush=True)
                    return
        # exe 가 죽었는지 확인해 더 명확한 오류를 낸다.
        rc = self._proc.poll() if self._proc is not None else None
        raise TimeoutError(
            f"unreal_bt_client.exe handshake 실패 (port={self.port}, "
            f"exe_addr={self._exe_addr}, proc_returncode={rc}). exe 경로/DLL/포트를 확인하세요.")

    def _alive(self) -> bool:
        """exe 프로세스가 살아 있는지."""
        return self._proc is not None and self._proc.poll() is None

    def _recover(self) -> bool:
        """exe 가 죽었으면 재실행+handshake 로 복구한다.

        성공하면 True(같은 port 소켓 재사용, 새 exe 가 접속·handshake). 재실행 상한
        (max_relaunches) 초과·복구 실패 시 self._dead=True 로 두고 False 를 반환한다
        (이후 compute_action 은 즉시 SAFE_ACTION → 2초 타임아웃 스톨/hang 방지).
        """
        if self._dead:
            return False
        if self._relaunch_count >= self.max_relaunches:
            self._dead = True
            if not self.quiet:
                print(f"[UnrealExeProvider] exe 재실행 상한({self.max_relaunches}) 초과 "
                      f"(port={self.port}) → 이후 SAFE_ACTION(직진) 상대로 대체", flush=True)
            return False
        self._relaunch_count += 1
        # 죽은(또는 좀비) 프로세스 정리 후 재실행.
        try:
            if self._proc is not None and self._proc.poll() is None:
                self._proc.terminate()
        except Exception:
            pass
        self._proc = None
        self._exe_addr = None
        try:
            self._launch_exe()
            self._handshake()          # 새 exe 주소 재취득 + SetPlaneID 인지
            if not self.quiet:
                print(f"[UnrealExeProvider] exe 재실행 성공 "
                      f"(port={self.port}, {self._relaunch_count}/{self.max_relaunches})", flush=True)
            return True
        except Exception as e:
            if not self.quiet:
                print(f"[UnrealExeProvider] exe 재실행 실패({e}, port={self.port})", flush=True)
            return False

    # ── ActionProvider 인터페이스 ─────────────────────────────────────────────
    def reset(self, context: ActionContext | None = None) -> None:
        # exe 내장 BT 는 SetPlaneID 처리 시 command_policy.reset 을 부르지만 BTActionProvider.reset
        # 은 no-op(BT 를 episode 간 유지) 이라 in-process BT 와 동일하게 상태가 이어진다.
        # 새 episode 시작을 알려 exe 의 own/enemy received 플래그만 리셋한다(index 는 유지 → 드롭 방지).
        if self._exe_addr is not None and self._sock is not None:
            try:
                self._sock.sendto(_pack_set_plane_id(self.own_plane_id), self._exe_addr)
            except OSError:
                pass

    def compute_action(self, context: ActionContext) -> ActionResult:
        own_state = context.ownship_state   # exe 가 조종할 기체(상대기)
        opp_state = context.target_state    # 본 기체(RL)
        if own_state is None or opp_state is None:
            return ActionResult(action=_SAFE_ACTION.copy(), source="unreal_exe",
                                confidence=0.0, info={"reason": "missing_state"})

        # exe 가 죽었으면 재실행 시도. 복구 실패(상한 초과)면 즉시 SAFE_ACTION 으로 빠르게 빠져
        # 나온다 — 매 substep ~2초 타임아웃 누적으로 워커(→학습 전체)가 hang 되는 것을 막는다.
        if not self._alive():
            if not self._recover():
                self._fail_count += 1
                return ActionResult(action=_SAFE_ACTION.copy(), source="unreal_exe",
                                    confidence=0.0, info={"reason": "exe_dead",
                                                          "fail_count": self._fail_count})

        own_pos, own_rot, own_vel = _state_to_plane_fields(own_state)
        opp_pos, opp_rot, opp_vel = _state_to_plane_fields(opp_state)

        cmd = self._exchange(own_pos, own_rot, own_vel, opp_pos, opp_rot, opp_vel)
        if cmd is None:
            self._fail_count += 1
            return ActionResult(action=_SAFE_ACTION.copy(), source="unreal_exe",
                                confidence=0.0, info={"reason": "no_cmd",
                                                      "fail_count": self._fail_count})

        action = clip_action([cmd["roll"], cmd["pitch"], cmd["yaw"], cmd["throttle"]])
        return ActionResult(action=action, source="unreal_exe", confidence=self.confidence,
                            info={"plane_id": cmd["plane_id"], "index": cmd["index"]})

    def _exchange(self, own_pos, own_rot, own_vel, opp_pos, opp_rot, opp_vel):
        """PlaneInfo 한 쌍을 보내고 그에 대응(index 일치) 하는 CMD 를 받아 온다."""
        assert self._sock is not None and self._exe_addr is not None
        for _attempt in range(self.max_resends + 1):
            self._idx += 1
            idx = self._idx
            # own(상대기) → enemy(본기체) 순서로 두 패킷 전송(같은 index).
            self._sock.sendto(
                _pack_plane_info(idx, self.own_plane_id, own_pos, own_rot, own_vel), self._exe_addr)
            self._sock.sendto(
                _pack_plane_info(idx, self.enemy_plane_id, opp_pos, opp_rot, opp_vel), self._exe_addr)

            deadline = time.time() + self.step_timeout_sec
            while time.time() < deadline:
                # exe 가 대기 중 죽으면 남은 타임아웃을 다 기다리지 말고 즉시 빠져나온다
                # (compute_action 이 다음 호출에서 _recover 로 재실행을 시도).
                if self._proc is not None and self._proc.poll() is not None:
                    return None
                try:
                    buf, _addr = self._sock.recvfrom(2048)
                except socket.timeout:
                    continue
                except OSError:
                    return None
                if _msg_type(buf) != MessageType.MT_CMD:
                    continue  # heartbeat(SimState/ClientInfo) 등은 무시
                cmd = _unpack_cmd(buf)
                if cmd["index"] == idx:      # 이번 pair 에 대한 응답만 채택(구 패킷 버림)
                    return cmd
                # index 가 더 낮은 stale CMD → 계속 기다림
        return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._proc is not None:
            try:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
            except Exception:
                pass
            self._proc = None
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None


__all__ = ["UnrealExeProvider", "DEFAULT_EXE"]
