from __future__ import annotations

import ctypes as ct
from pathlib import Path
from typing import Iterable

import numpy as np


class PublicState(ct.Structure):
    _pack_ = 8
    _fields_ = [(name, ct.c_double) for name in (
        "north_m", "east_m", "down_m", "roll_deg", "pitch_deg", "yaw_deg",
        "u_mps", "v_mps", "w_mps", "p_radps", "q_radps", "r_radps",
        "sim_time_s", "last_roll_cmd", "last_pitch_cmd", "last_rudder_cmd",
        "last_throttle_cmd",
    )]


class TargetSample(ct.Structure):
    _pack_ = 8
    _fields_ = [(name, ct.c_double) for name in (
        "north_m", "east_m", "down_m", "vel_n_mps", "vel_e_mps", "vel_d_mps",
        "roll_deg", "pitch_deg", "yaw_deg",
    )]

    @classmethod
    def from_values(cls, position_ned_m, velocity_ned_mps, euler_deg) -> "TargetSample":
        p = np.asarray(position_ned_m, dtype=np.float64)
        v = np.asarray(velocity_ned_mps, dtype=np.float64)
        r = np.asarray(euler_deg, dtype=np.float64)
        return cls(*p.tolist(), *v.tolist(), *r.tolist())


class Control(ct.Structure):
    _pack_ = 8
    _fields_ = [("roll", ct.c_double), ("pitch", ct.c_double),
                ("rudder", ct.c_double), ("throttle", ct.c_double)]


class NativeCostWeights(ct.Structure):
    _pack_ = 8
    _fields_ = [(name, ct.c_double) for name in (
        "damage_dealt", "damage_taken", "attack_geometry", "control_zone",
        "closure", "threat_geometry", "nose_advantage", "far_range",
        "overshoot", "ground", "envelope", "terminal_geometry",
        "control_slew",
    )]


class RolloutDiagnostic(ct.Structure):
    _pack_ = 8
    _fields_ = [
        ("score", ct.c_double),
        ("predicted_damage_dealt", ct.c_double),
        ("predicted_damage_taken", ct.c_double),
        ("min_altitude_ft", ct.c_double),
        ("min_range_ft", ct.c_double),
        ("final_ata_deg", ct.c_double),
        ("final_enemy_ata_deg", ct.c_double),
        ("final_speed_mps", ct.c_double),
        ("valid", ct.c_int32),
    ]


class DebugState(ct.Structure):
    _pack_ = 8
    _fields_ = [
        ("public_state", PublicState),
        ("aileron_deg", ct.c_double),
        ("elevator_deg", ct.c_double),
        ("rudder_deg", ct.c_double),
        ("engine_n2_percent", ct.c_double),
        ("alpha_deg", ct.c_double),
        ("beta_deg", ct.c_double),
    ]


class NativePredictor:
    def __init__(self, dll_path: str | Path, asset_root: str | Path, max_candidates: int):
        self.dll_path = Path(dll_path).resolve()
        self.asset_root = Path(asset_root).resolve()
        if not self.dll_path.is_file():
            raise FileNotFoundError(f"MPC predictor DLL not found: {self.dll_path}")
        if not (self.asset_root / "aircraft" / "f16" / "f16.xml").is_file():
            raise FileNotFoundError(f"MPC asset root is incomplete: {self.asset_root}")
        self.lib = ct.CDLL(str(self.dll_path))
        self._bind()
        self.handle = self.lib.MPC_Create(str(self.asset_root).encode("utf-8"), int(max_candidates))
        if not self.handle:
            raise RuntimeError("MPC_Create failed; verify JSBSim assets and native build")

    def _bind(self) -> None:
        self.lib.MPC_Create.argtypes = [ct.c_char_p, ct.c_int]
        self.lib.MPC_Create.restype = ct.c_void_p
        self.lib.MPC_Destroy.argtypes = [ct.c_void_p]
        self.lib.MPC_Destroy.restype = None
        self.lib.MPC_LastError.argtypes = [ct.c_void_p]
        self.lib.MPC_LastError.restype = ct.c_char_p
        self.lib.MPC_MaxCandidates.argtypes = [ct.c_void_p]
        self.lib.MPC_MaxCandidates.restype = ct.c_int
        self.lib.MPC_EvaluateBatch.argtypes = [
            ct.c_void_p, ct.POINTER(PublicState), ct.POINTER(TargetSample), ct.c_int,
            ct.POINTER(Control), ct.c_int, ct.c_int, ct.c_int,
            ct.POINTER(NativeCostWeights),
            ct.POINTER(RolloutDiagnostic),
        ]
        self.lib.MPC_EvaluateBatch.restype = ct.c_int
        self.lib.MPC_RolloutOne.argtypes = [
            ct.c_void_p, ct.POINTER(PublicState), ct.POINTER(Control), ct.c_int,
            ct.c_int, ct.POINTER(PublicState),
        ]
        self.lib.MPC_RolloutOne.restype = ct.c_int
        self.lib.MPC_RolloutDebug.argtypes = [
            ct.c_void_p, ct.POINTER(PublicState), ct.POINTER(Control), ct.c_int,
            ct.c_int, ct.POINTER(DebugState),
        ]
        self.lib.MPC_RolloutDebug.restype = ct.c_int
        self.lib.MPC_Version.argtypes = []
        self.lib.MPC_Version.restype = ct.c_char_p

    @property
    def version(self) -> str:
        return self.lib.MPC_Version().decode("utf-8", errors="replace")

    @property
    def max_candidates(self) -> int:
        return int(self.lib.MPC_MaxCandidates(self.handle))

    def _last_error(self) -> str:
        value = self.lib.MPC_LastError(self.handle)
        return value.decode("utf-8", errors="replace") if value else "unknown native error"

    def evaluate_batch(
        self,
        initial_state: PublicState,
        target_trajectory: Iterable[TargetSample],
        controls: np.ndarray,
        steps_per_knot: int,
        weights: NativeCostWeights,
    ) -> list[RolloutDiagnostic]:
        array = np.asarray(controls, dtype=np.float64)
        if array.ndim != 3 or array.shape[2] != 4:
            raise ValueError("controls must have shape [candidate, knot, 4]")
        candidate_count, knot_count, _ = array.shape
        if candidate_count > self.max_candidates:
            raise ValueError("candidate count exceeds native predictor capacity")
        flat = array.reshape(-1, 4)
        native_controls = (Control * len(flat))(*(Control(*row) for row in flat))
        target_values = list(target_trajectory)
        native_targets = (TargetSample * len(target_values))(*target_values)
        diagnostics = (RolloutDiagnostic * candidate_count)()
        success = self.lib.MPC_EvaluateBatch(
            self.handle,
            ct.byref(initial_state),
            native_targets,
            len(target_values),
            native_controls,
            candidate_count,
            knot_count,
            int(steps_per_knot),
            ct.byref(weights),
            diagnostics,
        )
        if not success:
            raise RuntimeError(f"MPC_EvaluateBatch failed: {self._last_error()}")
        return list(diagnostics)

    def rollout_one(
        self,
        initial_state: PublicState,
        controls: np.ndarray,
        steps_per_control: int,
    ) -> PublicState:
        values = np.asarray(controls, dtype=np.float64).reshape(-1, 4)
        native_controls = (Control * len(values))(*(Control(*row) for row in values))
        final_state = PublicState()
        success = self.lib.MPC_RolloutOne(
            self.handle,
            ct.byref(initial_state),
            native_controls,
            len(values),
            int(steps_per_control),
            ct.byref(final_state),
        )
        if not success:
            raise RuntimeError(f"MPC_RolloutOne failed: {self._last_error()}")
        return final_state

    def close(self) -> None:
        if getattr(self, "handle", None):
            self.lib.MPC_Destroy(self.handle)
            self.handle = None

    def rollout_debug(
        self,
        initial_state: PublicState,
        controls: np.ndarray,
        steps_per_control: int,
    ) -> DebugState:
        values = np.asarray(controls, dtype=np.float64).reshape(-1, 4)
        native_controls = (Control * len(values))(*(Control(*row) for row in values))
        final_state = DebugState()
        success = self.lib.MPC_RolloutDebug(
            self.handle,
            ct.byref(initial_state),
            native_controls,
            len(values),
            int(steps_per_control),
            ct.byref(final_state),
        )
        if not success:
            raise RuntimeError(f"MPC_RolloutDebug failed: {self._last_error()}")
        return final_state

    def __enter__(self) -> "NativePredictor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
