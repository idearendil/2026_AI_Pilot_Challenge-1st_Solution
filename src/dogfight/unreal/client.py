from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, is_dataclass
import math
import socket
import sys
import threading
import time
from typing import Any, Protocol

from dogfight.unreal.protocol import (
    AIType,
    CMD,
    ClientJoinInfo,
    GameControl,
    Init,
    MessageType,
    PlaneInfo,
    SetPlaneID,
    SimulationState,
    pack_client_join_info,
    pack_cmd,
    pack_simulation_state,
    unpack_game_control,
    unpack_init,
    unpack_message_type,
    unpack_plane_info,
    unpack_set_plane_id,
)


class CommandPolicy(Protocol):
    def reset(self, context: "RemoteClientContext") -> None: ...
    def compute_command(self, context: "RemoteClientContext") -> CMD: ...


@dataclass
class PlaneSnapshot:
    is_valid: bool = False
    plane_id: int = -1
    frame_index: int = 0
    plane_info: PlaneInfo | None = None

    def update(self, plane_info: PlaneInfo) -> None:
        self.is_valid = True
        self.plane_id = plane_info.plane_id
        self.frame_index = plane_info.index
        self.plane_info = plane_info


@dataclass
class RemoteClientContext:
    plane_id: int = -1
    frame_index: int = 0
    initial_state: Init | None = None
    own_plane: PlaneSnapshot = field(default_factory=PlaneSnapshot)
    enemy_plane: PlaneSnapshot = field(default_factory=PlaneSnapshot)
    game_control: GameControl | None = None


@dataclass
class PacketTrace:
    direction: str = ""
    message_type: str = ""
    timestamp: float = 0.0
    size_bytes: int = 0
    endpoint: str = ""
    fields: dict[str, Any] = field(default_factory=dict)


class UnrealAIPilotUDPClient:
    def __init__(
        self,
        command_policy: CommandPolicy,
        server_ip: str = "221.151.77.208",
        server_port: int = 9999,
        team_name: str = "ASDF",
        ai_type: AIType = AIType.ReinforcementLearning,
        simulation_state: int = 1,
        heartbeat_interval_sec: float = 1.0,
        command_delay_sec: float = 0.06,
        recv_timeout_sec: float = 0.2,
        enable_terminal_monitor: bool = False,
        terminal_monitor_interval_sec: float = 0.2,
    ):
        self.command_policy = command_policy
        self.server_ip = server_ip
        self.server_port = server_port
        self.team_name = team_name
        self.ai_type = ai_type
        self.simulation_state = simulation_state
        self.heartbeat_interval_sec = heartbeat_interval_sec
        self.command_delay_sec = command_delay_sec
        self.recv_timeout_sec = recv_timeout_sec
        self.enable_terminal_monitor = enable_terminal_monitor
        self.terminal_monitor_interval_sec = terminal_monitor_interval_sec

        self._socket: socket.socket | None = None
        self._server_addr: tuple[str, int] | None = None
        self._local_endpoint = ""
        self._udp_mode = "connected"
        self._running = False
        self._heartbeat_thread: threading.Thread | None = None
        self._terminal_monitor_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.context = RemoteClientContext()
        self._own_info_received = False
        self._enemy_info_received = False
        self._start_time = time.time()
        self._last_received_packet = PacketTrace()
        self._last_sent_packet = PacketTrace()
        self._last_plane_info_by_id: dict[int, PacketTrace] = {}
        self._rx_counts = {message_type.name: 0 for message_type in MessageType}
        self._tx_counts = {message_type.name: 0 for message_type in MessageType}

    def connect(self) -> None:
        if self._socket is not None:
            return
        server_host = self._normalize_server_host(self.server_ip)
        self._server_addr = (server_host, self.server_port)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if self._is_loopback_host(self.server_ip):
            self._socket.bind(("127.0.0.1", 0))
            self._udp_mode = "local-unconnected"
        else:
            self._socket.connect(self._server_addr)
            self._udp_mode = "connected"
        self._local_endpoint = self._format_endpoint(self._socket.getsockname())
        self._socket.settimeout(self.recv_timeout_sec)

    def run(self) -> None:
        self.connect()
        self._running = True
        self._start_terminal_monitor()
        self._start_heartbeat()
        try:
            self._receive_loop()
        finally:
            self.stop()

    def stop(self) -> None:
        self._running = False
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=1.0)
        if self._terminal_monitor_thread and self._terminal_monitor_thread.is_alive():
            self._terminal_monitor_thread.join(timeout=1.0)
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def enable_packet_monitor(self, refresh_interval_sec: float | None = None) -> None:
        with self._lock:
            self.enable_terminal_monitor = True
            if refresh_interval_sec is not None:
                self.terminal_monitor_interval_sec = refresh_interval_sec
        if self._running and not (self._terminal_monitor_thread and self._terminal_monitor_thread.is_alive()):
            self._start_terminal_monitor()

    def render_terminal_packet_monitor(self) -> str:
        with self._lock:
            context_copy = RemoteClientContext(
                plane_id=self.context.plane_id,
                frame_index=self.context.frame_index,
                initial_state=copy.deepcopy(self.context.initial_state),
                own_plane=copy.deepcopy(self.context.own_plane),
                enemy_plane=copy.deepcopy(self.context.enemy_plane),
                game_control=copy.deepcopy(self.context.game_control),
            )
            last_rx = copy.deepcopy(self._last_received_packet)
            last_tx = copy.deepcopy(self._last_sent_packet)
            plane_info_by_id = copy.deepcopy(self._last_plane_info_by_id)
            rx_counts = dict(self._rx_counts)
            tx_counts = dict(self._tx_counts)

        lines = [
            "=== Unreal UDP Packet Monitor ===",
            f"server={self.server_ip}:{self.server_port} team={self.team_name} ai_type={self.ai_type.name}",
            f"udp_mode={self._udp_mode} local={self._local_endpoint}",
            f"running={self._running} uptime_sec={time.time() - self._start_time:.1f} plane_id={context_copy.plane_id} frame_index={context_copy.frame_index}",
            "",
            "[Context]",
            f"own_plane.valid={context_copy.own_plane.is_valid} own_plane.id={context_copy.own_plane.plane_id} own_plane.frame={context_copy.own_plane.frame_index}",
            f"enemy_plane.valid={context_copy.enemy_plane.is_valid} enemy_plane.id={context_copy.enemy_plane.plane_id} enemy_plane.frame={context_copy.enemy_plane.frame_index}",
            f"game_control.command={getattr(context_copy.game_control, 'command', 'n/a')}",
            "",
            "[Counters]",
            "RX: " + self._format_counts(rx_counts),
            "TX: " + self._format_counts(tx_counts),
            "",
            "[PlaneInfo Latest]",
        ]
        lines.extend(self._format_plane_info_summary(plane_info_by_id))
        lines.extend([
            "",
            "[Last RX]",
        ])
        lines.extend(self._format_packet_trace(last_rx))
        lines.extend(["", "[Last TX]"])
        lines.extend(self._format_packet_trace(last_tx))
        return "\n".join(lines)

    def refresh_terminal_packet_monitor(self) -> None:
        snapshot = self.render_terminal_packet_monitor()
        sys.stdout.write("\r\x1b[2J\x1b[H")
        sys.stdout.write(snapshot)
        if not snapshot.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()

    def _start_heartbeat(self) -> None:
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    def _start_terminal_monitor(self) -> None:
        if not self.enable_terminal_monitor:
            return
        self._terminal_monitor_thread = threading.Thread(target=self._terminal_monitor_loop, daemon=True)
        self._terminal_monitor_thread.start()

    def _heartbeat_loop(self) -> None:
        while self._running:
            self.send_simulation_state()
            self.send_client_join_info()
            time.sleep(self.heartbeat_interval_sec)

    def _terminal_monitor_loop(self) -> None:
        while self._running:
            self.refresh_terminal_packet_monitor()
            time.sleep(self.terminal_monitor_interval_sec)

    def _receive_loop(self) -> None:
        assert self._socket is not None
        while self._running:
            try:
                if self._udp_mode == "local-unconnected":
                    buffer, remote_addr = self._socket.recvfrom(1024)
                    remote_endpoint = self._format_endpoint(remote_addr)
                else:
                    buffer = self._socket.recv(1024)
                    remote_endpoint = self._format_endpoint(self._server_addr)
            except socket.timeout:
                continue
            except OSError:
                break
            self._process_packet(buffer, remote_endpoint)

    def send_simulation_state(self) -> None:
        assert self._socket is not None
        simulation_state = SimulationState(state=self.simulation_state)
        packet = pack_simulation_state(simulation_state)
        self._send_packet(packet)
        self._record_packet(
            "TX",
            MessageType.MT_SimState,
            simulation_state,
            len(packet),
            self._format_endpoint(self._server_addr),
        )

    def send_client_join_info(self) -> None:
        assert self._socket is not None
        with self._lock:
            plane_id = self.context.plane_id
        join_info = ClientJoinInfo(team_name=self.team_name, ai_type=self.ai_type, plane_id=plane_id)
        packet = pack_client_join_info(join_info)
        self._send_packet(packet)
        self._record_packet(
            "TX",
            MessageType.MT_ClientInfo,
            join_info,
            len(packet),
            self._format_endpoint(self._server_addr),
        )

    def send_command(self, cmd: CMD) -> None:
        assert self._socket is not None
        packet = pack_cmd(cmd)
        self._send_packet(packet)
        self._record_packet(
            "TX",
            MessageType.MT_CMD,
            cmd,
            len(packet),
            self._format_endpoint(self._server_addr),
        )

    def _send_packet(self, packet: bytes) -> None:
        assert self._socket is not None
        assert self._server_addr is not None
        if self._udp_mode == "local-unconnected":
            self._socket.sendto(packet, self._server_addr)
        else:
            self._socket.send(packet)

    def _process_packet(self, buffer: bytes, remote_endpoint: str = "") -> None:
        if len(buffer) < 4:
            return

        message_type = unpack_message_type(buffer)
        if message_type == MessageType.MT_SetPlaneID:
            packet = unpack_set_plane_id(buffer)
            self._record_packet("RX", message_type, packet, len(buffer), remote_endpoint)
            self._handle_set_plane_id(packet)
        elif message_type == MessageType.MT_Init:
            packet = unpack_init(buffer)
            self._record_packet("RX", message_type, packet, len(buffer), remote_endpoint)
            self._handle_init(packet)
        elif message_type == MessageType.MT_GameControl:
            packet = unpack_game_control(buffer)
            self._record_packet("RX", message_type, packet, len(buffer), remote_endpoint)
            self._handle_game_control(packet)
        elif message_type == MessageType.MT_PlaneInfo:
            packet = unpack_plane_info(buffer)
            self._record_packet("RX", message_type, packet, len(buffer), remote_endpoint)
            self._handle_plane_info(packet)

    def _handle_set_plane_id(self, packet: SetPlaneID) -> None:
        with self._lock:
            self.context.plane_id = packet.plane_id
            self._own_info_received = False
            self._enemy_info_received = False
            self.command_policy.reset(self.context)

    def _handle_init(self, packet: Init) -> None:
        with self._lock:
            self.context.initial_state = packet
            self.context.own_plane = PlaneSnapshot()
            self.context.enemy_plane = PlaneSnapshot()
            self._own_info_received = False
            self._enemy_info_received = False
            self.command_policy.reset(self.context)

    def _handle_game_control(self, packet: GameControl) -> None:
        with self._lock:
            self.context.game_control = packet

    def _handle_plane_info(self, packet: PlaneInfo) -> None:
        with self._lock:
            if self.context.plane_id != -1 and packet.plane_id == self.context.plane_id:
                if self.context.own_plane.is_valid and packet.index < self.context.own_plane.frame_index:
                    return
                self.context.own_plane.update(packet)
                self._own_info_received = True
            else:
                if self.context.enemy_plane.is_valid and packet.index < self.context.enemy_plane.frame_index:
                    return
                self.context.enemy_plane.update(packet)
                self._enemy_info_received = True

            self.context.frame_index = packet.index
            should_send = self._own_info_received and self._enemy_info_received

            if not should_send:
                return

            context_copy = RemoteClientContext(
                plane_id=self.context.plane_id,
                frame_index=self.context.frame_index,
                initial_state=copy.deepcopy(self.context.initial_state),
                own_plane=copy.deepcopy(self.context.own_plane),
                enemy_plane=copy.deepcopy(self.context.enemy_plane),
                game_control=copy.deepcopy(self.context.game_control),
            )
            self._own_info_received = False
            self._enemy_info_received = False

        if self.command_delay_sec > 0:
            time.sleep(self.command_delay_sec)
        cmd = self.command_policy.compute_command(context_copy)
        self.send_command(cmd)

    def _record_packet(
        self,
        direction: str,
        message_type: MessageType,
        packet: Any,
        size_bytes: int,
        endpoint: str = "",
    ) -> None:
        trace = PacketTrace(
            direction=direction,
            message_type=message_type.name,
            timestamp=time.time(),
            size_bytes=size_bytes,
            endpoint=endpoint,
            fields=self._packet_to_fields(packet),
        )
        with self._lock:
            if direction == "RX":
                self._last_received_packet = trace
                self._rx_counts[message_type.name] += 1
                if message_type == MessageType.MT_PlaneInfo and isinstance(packet, PlaneInfo):
                    self._last_plane_info_by_id[int(packet.plane_id)] = trace
            else:
                self._last_sent_packet = trace
                self._tx_counts[message_type.name] += 1

    def _packet_to_fields(self, packet: Any) -> dict[str, Any]:
        if is_dataclass(packet):
            return self._normalize_value(asdict(packet))
        return {"value": self._normalize_value(packet)}

    def _normalize_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._normalize_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._normalize_value(item) for item in value]
        if hasattr(value, "name") and hasattr(value, "value"):
            return value.name
        return value

    def _format_counts(self, counts: dict[str, int]) -> str:
        visible_counts = [f"{name}={count}" for name, count in counts.items() if count]
        if not visible_counts:
            return "no packets yet"
        return ", ".join(visible_counts)

    def _format_packet_trace(self, trace: PacketTrace) -> list[str]:
        if not trace.message_type:
            return ["no packet yet"]
        lines = [
            f"type={trace.message_type} size_bytes={trace.size_bytes} age_sec={time.time() - trace.timestamp:.2f}",
        ]
        if trace.endpoint:
            label = "from" if trace.direction == "RX" else "to"
            lines.append(f"{label}={trace.endpoint}")
        lines.extend(self._flatten_mapping(trace.fields))
        return lines

    def _format_plane_info_summary(self, traces: dict[int, PacketTrace]) -> list[str]:
        if not traces:
            return ["plane_id=0 no packet yet", "plane_id=1 no packet yet"]
        plane_ids = [0, 1]
        plane_ids.extend(sorted(plane_id for plane_id in traces if plane_id not in {0, 1}))
        return [
            self._format_plane_info_row(plane_id, traces.get(plane_id))
            for plane_id in plane_ids
        ]

    def _format_plane_info_row(self, plane_id: int, trace: PacketTrace | None) -> str:
        if trace is None or not trace.fields:
            return f"plane_id={plane_id} no packet yet"
        fields = trace.fields
        position = fields.get("position", {})
        rotation = fields.get("rotation", {})
        velocity = fields.get("velocity", {})
        vx = self._as_float(velocity.get("x"))
        vy = self._as_float(velocity.get("y"))
        vz = self._as_float(velocity.get("z"))
        speed = math.sqrt(vx * vx + vy * vy + vz * vz)
        return (
            f"plane_id={plane_id} age={time.time() - trace.timestamp:.2f}s "
            f"frame={fields.get('index', 'n/a')} "
            f"pos=({self._as_float(position.get('x')):.2f},"
            f"{self._as_float(position.get('y')):.2f},"
            f"{self._as_float(position.get('z')):.2f}) "
            f"rot=({self._as_float(rotation.get('roll')):.1f},"
            f"{self._as_float(rotation.get('pitch')):.1f},"
            f"{self._as_float(rotation.get('yaw')):.1f}) "
            f"vel=({vx:.2f},{vy:.2f},{vz:.2f}) speed={speed:.2f}"
        )

    def _flatten_mapping(self, value: Any, prefix: str = "") -> list[str]:
        if isinstance(value, dict):
            lines: list[str] = []
            for key, item in value.items():
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                lines.extend(self._flatten_mapping(item, next_prefix))
            return lines
        if isinstance(value, list):
            lines: list[str] = []
            for index, item in enumerate(value):
                next_prefix = f"{prefix}[{index}]"
                lines.extend(self._flatten_mapping(item, next_prefix))
            return lines
        return [f"{prefix}={value}"]

    @staticmethod
    def _as_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _is_loopback_host(host: str) -> bool:
        value = host.strip().lower()
        return value in {"localhost", "127.0.0.1", "::1"}

    @staticmethod
    def _normalize_server_host(host: str) -> str:
        if UnrealAIPilotUDPClient._is_loopback_host(host):
            return "127.0.0.1"
        return host

    @staticmethod
    def _format_endpoint(endpoint: Any) -> str:
        if not endpoint:
            return ""
        try:
            host, port = endpoint[:2]
        except Exception:
            return str(endpoint)
        return f"{host}:{port}"
