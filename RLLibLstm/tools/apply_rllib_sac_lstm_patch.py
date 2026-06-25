"""Apply DogFightEnv Ray 2.54 SAC LSTM patch files to a conda env.

The patch payload is kept under ``RLLibLstm/ray_2_54_0_patched`` so another PC
can restore the exact RLlib files without brittle text replacements.

Typical use:

    python RLLibLstm/tools/apply_rllib_sac_lstm_patch.py \
        C:/Users/USER/anaconda3/envs/aip

The positional path may be a conda env root, ``python.exe``, ``site-packages``,
``ray`` package root, or ``ray/rllib`` root. Use ``--dry-run`` first on a new PC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable


PATCH_MARKER = "DogFightEnv SAC actor/recurrent-Q LSTM patch"
EXPECTED_RAY_VERSION = "2.54.0"
PATCH_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PATCH_ROOT = PATCH_PACKAGE_ROOT / "ray_2_54_0_patched" / "ray" / "rllib"

TARGET_RELATIVE_PATHS = (
    Path("algorithms/sac/default_sac_rl_module.py"),
    Path("algorithms/sac/sac_catalog.py"),
    Path("algorithms/sac/torch/default_sac_torch_rl_module.py"),
    Path("algorithms/sac/torch/sac_torch_learner.py"),
    Path("utils/replay_buffers/prioritized_episode_buffer.py"),
)


def sha256(path: Path) -> str:
    """Return a file's SHA256 digest."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _existing_paths(paths: Iterable[Path]) -> list[Path]:
    return [path for path in paths if path.exists()]


def _python_site_package_candidates(env_root: Path) -> list[Path]:
    """Return possible site-packages directories for a conda env root."""

    candidates = [env_root / "Lib" / "site-packages"]
    candidates.extend((env_root / "lib").glob("python*/site-packages"))
    return candidates


def infer_env_root(path: Path) -> Path | None:
    """Infer a conda env root from a Python executable path."""

    if not path.is_file():
        return None
    name = path.name.lower()
    if name in {"python.exe", "python"}:
        # Windows: <env>/python.exe. POSIX: <env>/bin/python.
        if path.parent.name.lower() == "bin":
            return path.parent.parent
        return path.parent
    return None


def find_rllib_root(input_path: Path) -> Path:
    """Find ``ray/rllib`` under a conda env, Python executable, or package path."""

    path = input_path.expanduser().resolve()
    env_root = infer_env_root(path) or path

    candidates: list[Path] = []

    if path.name == "rllib" and path.parent.name == "ray":
        candidates.append(path)
    if path.name == "ray":
        candidates.append(path / "rllib")
    if path.name == "site-packages":
        candidates.append(path / "ray" / "rllib")

    for site_packages in _python_site_package_candidates(env_root):
        candidates.append(site_packages / "ray" / "rllib")

    existing = _existing_paths(dict.fromkeys(candidates))
    for candidate in existing:
        if (candidate / "algorithms" / "sac" / "sac.py").exists():
            return candidate

    checked = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Could not find ray/rllib under the supplied path.\n"
        f"input_path={path}\nchecked:\n{checked}"
    )


def infer_python_executable(input_path: Path, rllib_root: Path) -> Path | None:
    """Infer the env Python executable for optional Ray version verification."""

    path = input_path.expanduser().resolve()
    if path.is_file() and path.name.lower() in {"python.exe", "python"}:
        return path

    # ray/rllib -> ray -> site-packages -> Lib -> env root on Windows.
    site_packages = rllib_root.parent.parent
    env_root = None
    if site_packages.name == "site-packages":
        if site_packages.parent.name == "Lib":
            env_root = site_packages.parent.parent
        elif site_packages.parent.parent.name == "lib":
            env_root = site_packages.parent.parent.parent

    if env_root is None:
        return None

    candidates = [
        env_root / "python.exe",
        env_root / "bin" / "python",
        env_root / "bin" / "python3",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def read_ray_version(python_executable: Path | None) -> str | None:
    """Read Ray version from the target env, if a Python executable is known."""

    if python_executable is None:
        return None
    try:
        completed = subprocess.run(
            [str(python_executable), "-c", "import ray; print(ray.__version__)"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:
        return None
    return completed.stdout.strip() or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply DogFightEnv Ray 2.54 SAC LSTM patched RLlib files to a conda "
            "environment."
        )
    )
    parser.add_argument(
        "conda_path",
        nargs="?",
        default=None,
        help=(
            "Conda env root, python.exe, site-packages, ray package root, or "
            "ray/rllib root. Defaults to CONDA_PREFIX when available."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be copied; do not modify site-packages.",
    )
    parser.add_argument(
        "--backup-dir",
        default=None,
        help=(
            "Directory for backups. Defaults to "
            "RLLibLstm/tools/backups/<env-name>_<timestamp>."
        ),
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Overwrite files without creating backups. Not recommended.",
    )
    parser.add_argument(
        "--skip-version-check",
        action="store_true",
        help="Do not warn/fail when target Ray version is not 2.54.0.",
    )
    parser.add_argument(
        "--force-version",
        action="store_true",
        help="Apply even when the detected Ray version is not 2.54.0.",
    )
    parser.add_argument(
        "--record",
        default=None,
        help="Patch record JSON path. Defaults to RLLibLstm/patch_record.json.",
    )
    return parser


def resolve_input_path(raw_path: str | None) -> Path:
    if raw_path:
        return Path(raw_path)

    import os

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        return Path(conda_prefix)
    raise SystemExit(
        "No conda path supplied and CONDA_PREFIX is empty. "
        "Pass the conda env root, for example:\n"
        "  python RLLibLstm/tools/apply_rllib_sac_lstm_patch.py "
        "C:/Users/USER/anaconda3/envs/aip"
    )


def default_backup_dir(rllib_root: Path) -> Path:
    site_packages = rllib_root.parent.parent
    env_name = "unknown_env"
    if site_packages.name == "site-packages":
        if site_packages.parent.name == "Lib":
            env_name = site_packages.parent.parent.name
        elif site_packages.parent.parent.name == "lib":
            env_name = site_packages.parent.parent.parent.name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(__file__).resolve().parent / "backups" / f"{env_name}_{timestamp}"


def apply_patch_files(args: argparse.Namespace) -> dict:
    input_path = resolve_input_path(args.conda_path)
    rllib_root = find_rllib_root(input_path)
    python_executable = infer_python_executable(input_path, rllib_root)
    ray_version = read_ray_version(python_executable)

    if (
        not args.skip_version_check
        and ray_version is not None
        and ray_version != EXPECTED_RAY_VERSION
        and not args.force_version
    ):
        raise SystemExit(
            f"Target Ray version is {ray_version}, expected {EXPECTED_RAY_VERSION}. "
            "Use --force-version to apply anyway."
        )

    backup_dir = (
        Path(args.backup_dir).expanduser().resolve()
        if args.backup_dir
        else default_backup_dir(rllib_root)
    )
    if not args.dry_run and not args.no_backup:
        backup_dir.mkdir(parents=True, exist_ok=True)

    records = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "python": str(python_executable) if python_executable else None,
        "input_path": str(input_path.expanduser().resolve()),
        "rllib_root": str(rllib_root),
        "ray_version": ray_version,
        "expected_ray_version": EXPECTED_RAY_VERSION,
        "patch_marker": PATCH_MARKER,
        "dry_run": args.dry_run,
        "backup_dir": None if args.no_backup else str(backup_dir),
        "files": {},
    }

    for relative_path in TARGET_RELATIVE_PATHS:
        source = PATCH_ROOT / relative_path
        target = rllib_root / relative_path
        label = relative_path.as_posix()

        if not source.exists():
            raise FileNotFoundError(source)
        if not target.exists():
            raise FileNotFoundError(target)

        original_hash = sha256(target)
        source_hash = sha256(source)
        changed = source_hash != original_hash
        backup_path = backup_dir / f"{target.name}.{original_hash[:12]}.bak"

        if not args.dry_run:
            if not args.no_backup and not backup_path.exists():
                backup_path.write_bytes(target.read_bytes())
            shutil.copy2(source, target)

        records["files"][label] = {
            "source": str(source),
            "target": str(target),
            "backup": None if args.no_backup else str(backup_path),
            "original_sha256": original_hash,
            "source_sha256": source_hash,
            "patched_sha256": source_hash if args.dry_run else sha256(target),
            "changed": changed,
        }

    if not args.dry_run:
        record_path = (
            Path(args.record).expanduser().resolve()
            if args.record
            else PATCH_PACKAGE_ROOT / "patch_record.json"
        )
        record_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        records["record_path"] = str(record_path)

    return records


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    records = apply_patch_files(args)
    print(json.dumps(records, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
