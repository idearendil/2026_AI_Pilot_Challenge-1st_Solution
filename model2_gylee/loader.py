"""Load ver09 iter-1415 and expose it as a DogFightEnv opponent provider."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import my_observation
from .model import make_actor_critic
from .normalizers import RunningMeanStd
from .self_play import SelfPlayProvider


DEFAULT_SNAPSHOT = Path(__file__).resolve().parent / "model" / "iter_1415.pt"
EXPECTED_SNAPSHOT_SHA256 = (
    "6A12ECD587A51627C0A67504579092382CC3FFE00C7F554345D825035C0E16CF"
)
EXPECTED_MODEL_KWARGS = {
    "obs_dim": 47,
    "act_dim": 4,
    "hidden": (512, 512, 512),
    "activation": "tanh",
    "num_bins": 19,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint payload must be a dict, got {type(payload)!r}")
    return payload


def _validate_contract(payload: dict[str, Any]) -> None:
    required = {"state_dict", "model_kwargs", "obs_rms"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"checkpoint is missing keys: {sorted(missing)}")
    kwargs = dict(payload["model_kwargs"])
    for key, expected in EXPECTED_MODEL_KWARGS.items():
        actual = kwargs.get(key)
        if key == "hidden" and actual is not None:
            actual = tuple(actual)
        if actual != expected:
            raise ValueError(
                f"model contract mismatch for {key}: got {actual!r}, expected {expected!r}"
            )
    rms = payload["obs_rms"]
    if not isinstance(rms, dict):
        raise ValueError("iter-1415 requires observation RMS data")
    mean = np.asarray(rms.get("mean"), dtype=np.float64)
    var = np.asarray(rms.get("var"), dtype=np.float64)
    if mean.shape != (47,) or var.shape != (47,):
        raise ValueError(
            f"observation RMS mismatch: mean={mean.shape}, var={var.shape}"
        )
    if not np.isfinite(mean).all() or not np.isfinite(var).all() or np.any(var < 0):
        raise ValueError("observation RMS contains invalid values")


def load_model(
    snapshot_path: str | Path = DEFAULT_SNAPSHOT,
    *,
    device: str = "cpu",
    verify_checksum: bool = True,
):
    """Return ``(model, observation_rms, metadata)`` for ver09 iter1415."""
    path = Path(snapshot_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"model2_gylee checkpoint not found: {path}")
    digest = _sha256(path)
    if verify_checksum and digest != EXPECTED_SNAPSHOT_SHA256:
        raise ValueError(
            f"iter-1415 SHA-256 mismatch: got {digest}, "
            f"expected {EXPECTED_SNAPSHOT_SHA256}"
        )
    payload = _torch_load(path)
    _validate_contract(payload)
    kwargs = dict(payload["model_kwargs"])
    model = make_actor_critic(**kwargs)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device)
    model.eval()
    obs_rms = RunningMeanStd.from_state_dict(payload["obs_rms"])
    metadata = {
        "name": "model2_gylee",
        "lineage": "ver09",
        "snapshot_kind": "latest_durable_training_snapshot_not_selected_best",
        "iteration": 1415,
        "global_step": int(payload.get("training_state", {}).get("global_step", 0)),
        "snapshot_path": str(path),
        "snapshot_sha256": digest,
        "observation_mode": my_observation.OBSERVATION_MODE,
        "observation_size": my_observation.OBSERVATION_SIZE,
        "model_kwargs": kwargs,
    }
    return model, obs_rms, metadata


def make_opponent_provider(
    snapshot_path: str | Path = DEFAULT_SNAPSHOT,
    *,
    step_ratio: int = 6,
    device: str = "cpu",
    explore: bool = False,
    verify_checksum: bool = True,
) -> SelfPlayProvider:
    """Create one independent provider for one environment/worker."""
    model, obs_rms, _ = load_model(
        snapshot_path, device=device, verify_checksum=verify_checksum
    )
    return SelfPlayProvider(
        model=model,
        obs_rms=obs_rms,
        observation_fn=my_observation.build_observation,
        observation_mode=my_observation.OBSERVATION_MODE,
        step_ratio=step_ratio,
        device=device,
        explore=explore,
    )


__all__ = [
    "DEFAULT_SNAPSHOT",
    "EXPECTED_SNAPSHOT_SHA256",
    "load_model",
    "make_opponent_provider",
]
