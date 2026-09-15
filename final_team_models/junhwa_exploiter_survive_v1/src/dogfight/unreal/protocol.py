from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import struct


class MessageType(IntEnum):
    MT_GameControl = 0
    MT_Init = 1
    MT_PlaneInfo = 2
    MT_Damage = 3
    MT_SimState = 4
    MT_VP = 5
    MT_CMD = 6
    MT_StatgeInfo = 7
    MT_ClientInfo = 8
    MT_SetPlaneID = 9


class AIType(IntEnum):
    RuleBased = 0
    ReinforcementLearning = 1
    SupervisedLearning = 2
    Fusion = 3
    etc = 4


@dataclass
class Vector3D:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Rotation3D:
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


@dataclass
class GameControl:
    command: int


@dataclass
class Init:
    plane1_location: Vector3D
    plane1_rotation: Rotation3D
    plane1_speed: float
    plane2_location: Vector3D
    plane2_rotation: Rotation3D
    plane2_speed: float


@dataclass
class PlaneInfo:
    index: int
    plane_id: int
    position: Vector3D
    rotation: Rotation3D
    velocity: Vector3D


@dataclass
class SimulationState:
    state: int


@dataclass
class SetPlaneID:
    plane_id: int


@dataclass
class ClientJoinInfo:
    team_name: str
    ai_type: AIType
    plane_id: int


@dataclass
class CMD:
    plane_id: int
    index: int
    roll_cmd: float
    pitch_cmd: float
    yaw_cmd: float
    throttle_cmd: float


GAME_CONTROL_STRUCT = struct.Struct("<ib")
INIT_STRUCT = struct.Struct("<i3f3ff3f3ff")
PLANE_INFO_STRUCT = struct.Struct("<iQb3f3f3f")
SIMULATION_STATE_STRUCT = struct.Struct("<ib")
SET_PLANE_ID_STRUCT = struct.Struct("<ib")
CLIENT_JOIN_INFO_STRUCT = struct.Struct("<i30sib")
CMD_STRUCT = struct.Struct("<ibQffff")
MESSAGE_TYPE_STRUCT = struct.Struct("<i")


def pack_simulation_state(state: SimulationState) -> bytes:
    return SIMULATION_STATE_STRUCT.pack(int(MessageType.MT_SimState), int(state.state))


def pack_client_join_info(info: ClientJoinInfo) -> bytes:
    team_name = info.team_name.encode("utf-8")[:29]
    team_name = team_name + b"\x00" * (30 - len(team_name))
    return CLIENT_JOIN_INFO_STRUCT.pack(
        int(MessageType.MT_ClientInfo),
        team_name,
        int(info.ai_type),
        int(info.plane_id),
    )


def pack_cmd(cmd: CMD) -> bytes:
    return CMD_STRUCT.pack(
        int(MessageType.MT_CMD),
        int(cmd.plane_id),
        int(cmd.index),
        float(cmd.roll_cmd),
        float(cmd.pitch_cmd),
        float(cmd.yaw_cmd),
        float(cmd.throttle_cmd),
    )


def unpack_message_type(buffer: bytes) -> MessageType:
    return MessageType(MESSAGE_TYPE_STRUCT.unpack_from(buffer)[0])


def unpack_game_control(buffer: bytes) -> GameControl:
    _, command = GAME_CONTROL_STRUCT.unpack(buffer[: GAME_CONTROL_STRUCT.size])
    return GameControl(command=command)


def unpack_set_plane_id(buffer: bytes) -> SetPlaneID:
    _, plane_id = SET_PLANE_ID_STRUCT.unpack(buffer[: SET_PLANE_ID_STRUCT.size])
    return SetPlaneID(plane_id=plane_id)


def unpack_init(buffer: bytes) -> Init:
    unpacked = INIT_STRUCT.unpack(buffer[: INIT_STRUCT.size])
    return Init(
        plane1_location=Vector3D(*unpacked[1:4]),
        plane1_rotation=Rotation3D(*unpacked[4:7]),
        plane1_speed=unpacked[7],
        plane2_location=Vector3D(*unpacked[8:11]),
        plane2_rotation=Rotation3D(*unpacked[11:14]),
        plane2_speed=unpacked[14],
    )


def unpack_plane_info(buffer: bytes) -> PlaneInfo:
    unpacked = PLANE_INFO_STRUCT.unpack(buffer[: PLANE_INFO_STRUCT.size])
    return PlaneInfo(
        index=unpacked[1],
        plane_id=unpacked[2],
        position=Vector3D(*unpacked[3:6]),
        rotation=Rotation3D(*unpacked[6:9]),
        velocity=Vector3D(*unpacked[9:12]),
    )
