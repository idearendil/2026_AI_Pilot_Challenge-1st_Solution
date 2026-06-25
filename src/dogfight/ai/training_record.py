from __future__ import annotations

import json
import math
import platform
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from dogfight.envs.observation import describe_observation
from dogfight.envs.reward import describe_reward


def save_training_record(
    output_dir,
    algorithm_name: str,
    cli_args: dict,
    env_config: dict,
    algorithm_config: dict,
    result_history: list[dict],
    workspace_root,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace_root = Path(workspace_root)

    observation_summary = env_config.get("observation_summary")
    if not isinstance(observation_summary, dict):
        observation_summary = describe_observation(env_config["observation_mode"])
    reward_summary = describe_reward(env_config["reward"], env_config["wez"])

    record = _json_safe({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "algorithm": algorithm_name,
        "cli_args": cli_args,
        "env_config": env_config,
        "algorithm_config": algorithm_config,
        "observation_summary": observation_summary,
        "reward_summary": reward_summary,
        "result_history": result_history,
    })

    (output_dir / "training_record.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "training_record.md").write_text(_to_markdown(record), encoding="utf-8")

    for relative_path in (
        Path("src/dogfight/envs/observation.py"),
        Path("src/dogfight/envs/reward.py"),
        Path("src/dogfight/config.py"),
    ):
        source = workspace_root / relative_path
        if source.exists():
            destination = output_dir / relative_path.name
            shutil.copy2(source, destination)

    observation_module = env_config.get("observation_module")
    if observation_module:
        module_path = Path(*str(observation_module).split(".")).with_suffix(".py")
        source = workspace_root / module_path
        if source.exists():
            shutil.copy2(source, output_dir / module_path.name)

    return output_dir


def _json_safe(value: Any) -> Any:
    """Convert RLlib configs/results into JSON-serializable values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if callable(value):
        module = getattr(value, "__module__", value.__class__.__module__)
        name = getattr(value, "__qualname__", getattr(value, "__name__", None))
        return f"{module}.{name}" if name else str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            pass
    return str(value)


def _to_markdown(record: dict) -> str:
    lines = [
        "# Training Record",
        "",
        f"- Created at: `{record['created_at']}`",
        f"- Python: `{record['python_version']}`",
        f"- Platform: `{record['platform']}`",
        f"- Algorithm: `{record['algorithm']}`",
        "",
        "## Observation",
        "",
        f"- Mode: `{record['observation_summary']['mode']}`",
        f"- Size: `{record['observation_summary']['size']}`",
        f"- Description: {record['observation_summary']['description']}",
        "- Features:",
    ]
    lines.extend([f"  - `{feature}`" for feature in record["observation_summary"]["features"]])
    lines.extend(
        [
            "",
            "## Reward",
            "",
            f"- Description: {record['reward_summary']['description']}",
            f"- Step penalty: `{record['reward_summary']['step_penalty']}`",
            f"- Damage scale: `{record['reward_summary']['damage_scale']}`",
            f"- Pursuit scale: `{record['reward_summary'].get('pursuit_scale', 'n/a')}`",
            f"- Pursuit half angle (deg): `{record['reward_summary'].get('pursuit_half_angle_deg', 'n/a')}`",
            f"- Pursuit range (m): `{record['reward_summary'].get('pursuit_range_m', 'n/a')}`",
            f"- Low altitude penalty: `{record['reward_summary']['low_altitude_penalty']}`",
            f"- Win reward: `{record['reward_summary']['win_reward']}`",
            f"- Loss reward: `{record['reward_summary']['loss_reward']}`",
            f"- Draw reward: `{record['reward_summary']['draw_reward']}`",
            "",
            "## CLI Arguments",
            "",
            "```json",
            json.dumps(record["cli_args"], indent=2, ensure_ascii=False),
            "```",
            "",
            "## Environment Config",
            "",
            "```json",
            json.dumps(record["env_config"], indent=2, ensure_ascii=False),
            "```",
            "",
            "## Training History",
            "",
        ]
    )
    for item in record["result_history"]:
        lines.append(
            f"- iter `{item['iteration']}`: reward_mean=`{item.get('reward_mean', 'n/a')}`, "
            f"episode_len_mean=`{item.get('episode_len_mean', 'n/a')}`"
        )
    lines.append("")
    return "\n".join(lines)
