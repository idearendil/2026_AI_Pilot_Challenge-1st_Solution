from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dogfight.unreal.client import UnrealAIPilotUDPClient  # noqa: E402
from dogfight.unreal.protocol import AIType  # noqa: E402
from mpc.command_policy import MPCCommandPolicy  # noqa: E402
from mpc.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Release_MPC competition UDP client")
    parser.add_argument("--server-ip", default=os.getenv("DOGFIGHT_SERVER_IP", "221.151.77.208"))
    parser.add_argument("--server-port", type=int, default=int(os.getenv("DOGFIGHT_SERVER_PORT", "9999")))
    parser.add_argument("--team-name", default=os.getenv("DOGFIGHT_TEAM_NAME", "Release_MPC"))
    parser.add_argument("--config", default=str(ROOT / "configs" / "mpc.yaml"))
    parser.add_argument("--monitor", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    policy = MPCCommandPolicy(ROOT, config)
    client = UnrealAIPilotUDPClient(
        command_policy=policy,
        server_ip=args.server_ip,
        server_port=args.server_port,
        team_name=args.team_name,
        ai_type=AIType.etc,
        command_delay_sec=0.0,
        enable_terminal_monitor=args.monitor,
    )
    try:
        client.run()
    finally:
        policy.close()


if __name__ == "__main__":
    main()
