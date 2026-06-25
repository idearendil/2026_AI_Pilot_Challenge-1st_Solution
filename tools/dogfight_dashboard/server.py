"""Unified HTTP server for DogFightEnv training and replay dashboards."""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

try:
    from .training_data import MetricsReader
except ImportError:  # pragma: no cover - direct script execution.
    from training_data import MetricsReader  # type: ignore


PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"
TOOLS_DIR = PACKAGE_DIR.parent
DEFAULT_ENV_ROOT = TOOLS_DIR.parent / "MyTrainEnv"

if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from web_log_viewer.log_data import (  # noqa: E402
    DEFAULT_WEZ_ANGLE_DEG,
    DEFAULT_WEZ_MIN_RANGE_M,
    DEFAULT_WEZ_RANGE_M,
)
from web_log_viewer.server import ViewerRepository  # noqa: E402


class DashboardRepository:
    """Bundle training metrics and replay log repositories."""

    def __init__(
        self,
        env_root: Path,
        training_logdir: Path | None,
        replay_logdir: Path | None,
        mesh_path: Path | None,
        default_tab: str,
    ) -> None:
        self.env_root = env_root.resolve()
        self.training_logdir = (
            training_logdir or self.env_root / "artifacts" / "dashboard"
        ).resolve()
        self.training_logdir.mkdir(parents=True, exist_ok=True)
        self.metrics = MetricsReader(self.training_logdir)
        self.replay = ViewerRepository(
            env_root=self.env_root,
            logdir=replay_logdir,
            mesh_path=mesh_path,
        )
        self.default_tab = default_tab if default_tab in {"training", "replay"} else "training"


def make_handler(repository: DashboardRepository):
    """Create a request handler bound to dashboard repositories."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)

            if path.startswith("/api/"):
                self._handle_api(path, query)
                return

            if path == "/":
                path = "/index.html"
            file_path = (STATIC_DIR / path.lstrip("/")).resolve()
            if not _is_relative_to(file_path, STATIC_DIR) or not file_path.is_file():
                self.send_error(404)
                return
            body = file_path.read_bytes()
            mime, _ = mimetypes.guess_type(str(file_path))
            self.send_response(200)
            self.send_header("Content-Type", mime or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _handle_api(self, path: str, query: dict) -> None:
            try:
                if path == "/api/app/config":
                    self._json(
                        {
                            "envRoot": str(repository.env_root),
                            "trainingLogdir": str(repository.training_logdir),
                            "replayLogdir": str(repository.replay.logdir),
                            "mesh": str(repository.replay.mesh_path),
                            "defaultTab": repository.default_tab,
                        }
                    )
                elif path == "/api/training/runs":
                    self._json({"runs": repository.metrics.list_runs()})
                elif path == "/api/training/metrics":
                    run = _query_value(query, "run")
                    since = int(_query_value(query, "since_step", "0"))
                    smooth = int(_query_value(query, "smooth", "1"))
                    self._json(repository.metrics.read_metrics(run, since, smooth))
                elif path == "/api/training/latest":
                    self._json(repository.metrics.get_latest(_query_value(query, "run")))
                elif path == "/api/training/config":
                    self._json(repository.metrics.get_config(_query_value(query, "run")))
                elif path == "/api/replay/logs":
                    self._json({"logs": repository.replay.list_logs()})
                elif path == "/api/replay/data":
                    self._json(
                        repository.replay.load_replay(
                            ownship=_query_value(query, "ownship"),
                            target=_query_value(query, "target"),
                        )
                    )
                elif path == "/api/replay/mesh/f16":
                    self._json(repository.replay.load_mesh())
                elif path == "/api/replay/config":
                    self._json(
                        {
                            "envRoot": str(repository.replay.env_root),
                            "logdir": str(repository.replay.logdir),
                            "mesh": str(repository.replay.mesh_path),
                            "defaults": {
                                "wezMinRangeM": DEFAULT_WEZ_MIN_RANGE_M,
                                "wezRangeM": DEFAULT_WEZ_RANGE_M,
                                "wezAngleDeg": DEFAULT_WEZ_ANGLE_DEG,
                            },
                        }
                    )
                else:
                    self._json({"error": "not found"}, status=404)
            except Exception as exc:
                self._json({"error": str(exc)}, status=400)

        def _json(self, data: dict, status: int = 200) -> None:
            body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return Handler


def _query_value(query: dict, key: str, default: str | None = None) -> str:
    value = (query.get(key) or [default])[0]
    if value is None or value == "":
        raise ValueError(f"{key} parameter required")
    return str(value)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DogFight unified dashboard")
    parser.add_argument(
        "--env-root",
        default=str(DEFAULT_ENV_ROOT),
        help="DogFightEnv environment root containing artifacts/, logs/, and assets/.",
    )
    parser.add_argument(
        "--training-logdir",
        default=None,
        help="Training metrics directory. Defaults to <env-root>/artifacts/dashboard.",
    )
    parser.add_argument(
        "--logdir",
        "--replay-logdir",
        dest="replay_logdir",
        default=None,
        help="Replay CSV directory. Defaults to <env-root>/logs.",
    )
    parser.add_argument(
        "--mesh",
        default=None,
        help="F-16 OBJ mesh path. Defaults to <env-root>/assets/meshes/f16.",
    )
    parser.add_argument(
        "--default-tab",
        choices=["training", "replay"],
        default="training",
        help="Initial tab opened by the browser UI.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = DashboardRepository(
        env_root=Path(args.env_root).expanduser(),
        training_logdir=(
            Path(args.training_logdir).expanduser() if args.training_logdir else None
        ),
        replay_logdir=Path(args.replay_logdir).expanduser() if args.replay_logdir else None,
        mesh_path=Path(args.mesh).expanduser() if args.mesh else None,
        default_tab=args.default_tab,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(repository))
    first_url = f"http://{args.host}:{args.port}"
    print(f"DogFight dashboard: {first_url}/?tab={repository.default_tab}")
    print(f"Env root:           {repository.env_root}")
    print(f"Training logdir:    {repository.training_logdir}")
    print(f"Replay logdir:      {repository.replay.logdir}")
    print(f"Mesh:               {repository.replay.mesh_path}")
    replay_logs = repository.replay.list_logs()
    if replay_logs:
        first = replay_logs[0]
        query = f"ownship={quote(first['ownship'])}&target={quote(first['target'])}"
        print(f"Latest replay API:  {first_url}/api/replay/data?{query}")
    print("Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
