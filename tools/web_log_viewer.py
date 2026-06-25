"""Compatibility launcher for the Replay tab in the unified dashboard."""

from __future__ import annotations

import sys
from pathlib import Path


ENV_ROOT = Path(__file__).resolve().parents[1]
RELEASE_TOOLS = ENV_ROOT / "tools"
sys.path.insert(0, str(RELEASE_TOOLS))

from dogfight_dashboard.server import main  # noqa: E402


def _has_option(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:])


if __name__ == "__main__":
    if not _has_option("--env-root"):
        sys.argv.extend(["--env-root", str(ENV_ROOT)])
    if not _has_option("--default-tab"):
        sys.argv.extend(["--default-tab", "replay"])
    if not _has_option("--port"):
        sys.argv.extend(["--port", "7870"])
    main()
