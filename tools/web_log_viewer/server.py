"""HTTP server for the DogFightEnv web log playback viewer."""

from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

try:
    from .log_data import (
        DEFAULT_WEZ_ANGLE_DEG,
        DEFAULT_WEZ_MIN_RANGE_M,
        DEFAULT_WEZ_RANGE_M,
        build_viewer_data,
        discover_log_pairs,
        infer_summary_path,
        parse_obj_mesh,
        viewer_data_to_json,
    )
except ImportError:  # pragma: no cover - supports direct script execution.
    from log_data import (  # type: ignore
        DEFAULT_WEZ_ANGLE_DEG,
        DEFAULT_WEZ_MIN_RANGE_M,
        DEFAULT_WEZ_RANGE_M,
        build_viewer_data,
        discover_log_pairs,
        infer_summary_path,
        parse_obj_mesh,
        viewer_data_to_json,
    )


PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"
DEFAULT_ENV_ROOT = PACKAGE_DIR.parents[1] / "MyTrainEnv"


class ViewerRepository:
    """Resolve log and mesh files for the web viewer API."""

    def __init__(self, env_root: Path, logdir: Path | None, mesh_path: Path | None):
        self.env_root = env_root.resolve()
        self.logdir = (logdir or self.env_root / "logs").resolve()
        self.mesh_path = (
            mesh_path
            or self.env_root / "assets" / "meshes" / "f16" / "f16_simple_cc_by.obj"
        ).resolve()

    def list_logs(self) -> list[dict]:
        """Return discoverable Blue/Red log pairs."""
        pairs = []
        for ownship, target in discover_log_pairs(self.logdir):
            metadata_path = infer_summary_path(ownship)
            try:
                stat_time = max(ownship.stat().st_mtime, target.stat().st_mtime)
            except OSError:
                stat_time = 0.0
            pairs.append(
                {
                    "label": _pair_label(ownship),
                    "ownship": self._url_path(ownship),
                    "target": self._url_path(target),
                    "ownshipName": ownship.name,
                    "targetName": target.name,
                    "metadataName": metadata_path.name if metadata_path else None,
                    "lastModified": stat_time,
                }
            )
        pairs.sort(key=lambda item: item["lastModified"], reverse=True)
        return pairs

    def load_replay(self, ownship: str, target: str) -> dict:
        """Load and serialize one replay log pair."""
        ownship_path = self._safe_log_path(ownship)
        target_path = self._safe_log_path(target)
        metadata_path = infer_summary_path(ownship_path)
        data = build_viewer_data(
            ownship_log=ownship_path,
            target_log=target_path,
            metadata_path=metadata_path,
            fallback_end_condition="n/a",
        )
        return viewer_data_to_json(data)

    def load_mesh(self) -> dict:
        """Load the configured F-16 mesh as JSON geometry."""
        return parse_obj_mesh(self.mesh_path)

    def _safe_log_path(self, value: str) -> Path:
        raw = unquote(value)
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = self.logdir / candidate
        resolved = candidate.resolve()
        if not _is_relative_to(resolved, self.logdir):
            raise ValueError("log path must stay inside the configured logdir")
        if not resolved.is_file():
            raise FileNotFoundError(f"log not found: {resolved}")
        return resolved

    def _url_path(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.logdir).as_posix()
        except ValueError:
            return path.name


def make_handler(repository: ViewerRepository):
    """Create a request handler bound to a viewer repository."""

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
                if path == "/api/logs":
                    self._json({"logs": repository.list_logs()})
                elif path == "/api/replay":
                    self._json(
                        repository.load_replay(
                            ownship=_query_value(query, "ownship"),
                            target=_query_value(query, "target"),
                        )
                    )
                elif path == "/api/mesh/f16":
                    self._json(repository.load_mesh())
                elif path == "/api/config":
                    self._json(
                        {
                            "envRoot": str(repository.env_root),
                            "logdir": str(repository.logdir),
                            "mesh": str(repository.mesh_path),
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


def _query_value(query: dict, key: str) -> str:
    value = (query.get(key) or [""])[0]
    if value == "":
        raise ValueError(f"{key} parameter required")
    return str(value)


def _pair_label(ownship: Path) -> str:
    name = ownship.name
    for marker in ("_ownship_(F-16)[Blue].csv", "_ownship_[Blue].csv"):
        if marker in name:
            return name.replace(marker, "")
    return name.replace(".csv", "")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DogFightEnv web log viewer")
    parser.add_argument(
        "--env-root",
        default=str(DEFAULT_ENV_ROOT),
        help="DogFightEnv environment root containing logs/ and assets/.",
    )
    parser.add_argument(
        "--logdir",
        default=None,
        help="Log directory. Defaults to <env-root>/logs.",
    )
    parser.add_argument(
        "--mesh",
        default=None,
        help="F-16 OBJ mesh path. Defaults to <env-root>/assets/meshes/f16.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7870)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = ViewerRepository(
        env_root=Path(args.env_root).expanduser(),
        logdir=Path(args.logdir).expanduser() if args.logdir else None,
        mesh_path=Path(args.mesh).expanduser() if args.mesh else None,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(repository))
    first_url = f"http://{args.host}:{args.port}"
    print(f"Web log viewer: {first_url}")
    print(f"Env root:       {repository.env_root}")
    print(f"Logdir:         {repository.logdir}")
    print(f"Mesh:           {repository.mesh_path}")
    if repository.list_logs():
        first = repository.list_logs()[0]
        query = f"ownship={quote(first['ownship'])}&target={quote(first['target'])}"
        print(f"Latest replay:  {first_url}/?{query}")
    print("Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
