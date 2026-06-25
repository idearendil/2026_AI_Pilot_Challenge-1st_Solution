"""Compatibility launcher for the Training tab in the unified dashboard."""

from __future__ import annotations

import sys
from pathlib import Path


ENV_ROOT = Path(__file__).resolve().parents[2]
RELEASE_TOOLS = ENV_ROOT / "tools"
sys.path.insert(0, str(RELEASE_TOOLS))

from dogfight_dashboard.server import main  # noqa: E402


def _has_option(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:])


def _translate_legacy_args() -> None:
    for index, arg in enumerate(sys.argv):
        if arg == "--logdir":
            sys.argv[index] = "--training-logdir"
        elif arg.startswith("--logdir="):
            sys.argv[index] = "--training-logdir=" + arg.split("=", 1)[1]


if __name__ == "__main__":
    _translate_legacy_args()
    if not _has_option("--env-root"):
        sys.argv.extend(["--env-root", str(ENV_ROOT)])
    if not _has_option("--default-tab"):
        sys.argv.extend(["--default-tab", "training"])
    main()
