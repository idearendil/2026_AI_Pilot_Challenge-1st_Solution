"""Versioned CUDA-bundle observation reconstruction for CPU deployment.

Keep historical CPU bundles on their original provider path. CUDA training stores
applied commands (throttle 0..1), and observes the initial state without advancing
HP/time/angular history. Later observations first consume the last applied command
and advance the current post-step state. Each provider owns its reconstruction.
"""
from __future__ import annotations

import numpy as np

from claude_code import my_observation as MO

CUDA_OBSERVATION_CONTRACT = "cuda_claude164r_command_poststep_v1"


def uses_cuda_observation_contract(metadata):
    contract = metadata.get("observation_contract")
    if contract is not None and contract != CUDA_OBSERVATION_CONTRACT:
        raise ValueError(f"Unsupported observation contract: {contract!r}")
    enabled = (contract == CUDA_OBSERVATION_CONTRACT or
               metadata.get("trainer") == "cuda_fdm.PPOGPUTrainer")
    if enabled:
        if (metadata.get("observation_module") != "claude_code.my_observation" or
                int(metadata.get("observation_size", 0)) != MO.OBSERVATION_SIZE):
            raise ValueError("CUDA observation contract requires claude_code.my_observation / 214D")
        if (float(metadata.get("policy_hz", 10)) != 10. or
                int(metadata.get("action_repeat", 6)) != 6):
            raise ValueError("CUDA observation contract currently requires 10Hz / action_repeat=6")
    return enabled


class CUDAObservationState:
    """One instance per agent; no shared singleton or updates on cached frames."""

    def __init__(self):
        self.rec = MO.StateReconstructor()
        self.reset()

    def reset(self):
        self.rec.reset()
        self._started = False
        self._pending_command = None

    def begin_step(self, own, target):
        for label, value in (("own", own), ("target", target)):
            state = np.asarray(value, dtype=np.float64).reshape(-1)
            if state.size < 9 or not np.isfinite(state[:9]).all():
                raise FloatingPointError(f"Invalid {label} state for CUDA observation contract")
        if self._started:
            if self._pending_command is None:
                raise RuntimeError("Observation advanced without an applied command")
            self.rec.push_action(self._pending_command)
            self.rec.advance(own, target)
        else:
            self._started = True
        self._pending_command = None

    def observation(self, own, target):
        self.begin_step(own, target)
        obs = MO.build_observation(own, target, self.rec._geo, reconstructor=self.rec)
        if not np.isfinite(obs).all():
            raise FloatingPointError("Non-finite reconstructed CUDA observation")
        return obs

    def record_command(self, command):
        command = np.asarray(command, dtype=np.float64).reshape(-1)
        if command.size != 4 or not np.isfinite(command).all():
            raise FloatingPointError("Invalid applied command for CUDA action history")
        if (np.any(np.abs(command[:3]) > 1.000001) or
                not -1e-6 <= command[3] <= 1.000001):
            raise ValueError("Applied command must use throttle [0,1] and other controls [-1,1]")
        # Hybrid controllers may update at 60Hz: retain the most recently applied
        # command, but shift history only at the next 10Hz observation boundary.
        self._pending_command = command.copy()
