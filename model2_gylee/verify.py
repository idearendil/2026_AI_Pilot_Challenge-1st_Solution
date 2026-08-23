"""Offline package integrity and deterministic inference smoke test."""

from __future__ import annotations

import numpy as np
import torch

from . import my_observation
from .loader import load_model
from .model import discrete_indices_to_continuous, policy_action_to_command


def main() -> None:
    model, rms, metadata = load_model()
    assert my_observation.OBSERVATION_MODE == "claude47r"
    assert my_observation.OBSERVATION_SIZE == 47
    assert rms.mean.shape == (47,) and rms.var.shape == (47,)
    raw = np.zeros(47, dtype=np.float64)
    obs = np.clip((raw - rms.mean) / np.sqrt(rms.var + 1e-8), -10, 10)
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        indices = model.act_deterministic(obs_t)[0].cpu().numpy()
    command = policy_action_to_command(
        discrete_indices_to_continuous(indices, model.num_bins)
    )
    assert indices.shape == (4,) and command.shape == (4,)
    assert np.isfinite(command).all()
    assert np.all(command[:3] >= -1) and np.all(command[:3] <= 1)
    assert 0 <= float(command[3]) <= 1
    print("model2_gylee verification: PASS")
    print(f"snapshot SHA-256: {metadata['snapshot_sha256']}")
    print(f"iteration/global step: {metadata['iteration']}/{metadata['global_step']}")
    print(
        "contract: 47D claude47r -> 512x512x512 tanh -> "
        "4 axes x 19 categorical bins"
    )
    print(f"deterministic smoke command: {command.tolist()}")


if __name__ == "__main__":
    main()
