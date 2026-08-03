from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]

# (source relative to Release_MPC, path inside the ZIP)
FILES = (
    ("share_docs/README.md", "README.md"),
    ("share_docs/ARCHITECTURE.md", "docs/ARCHITECTURE.md"),
    ("share_docs/BUILD_AND_INTEGRATION.md", "docs/BUILD_AND_INTEGRATION.md"),
    ("share_docs/VALIDATION.md", "docs/VALIDATION.md"),
    ("configs/mpc.yaml", "configs/mpc.yaml"),
    ("student/__init__.py", "student/__init__.py"),
    ("student/my_submission.py", "student/my_submission.py"),
    ("src/mpc/__init__.py", "src/mpc/__init__.py"),
    ("src/mpc/command_policy.py", "src/mpc/command_policy.py"),
    ("src/mpc/config.py", "src/mpc/config.py"),
    ("src/mpc/native.py", "src/mpc/native.py"),
    ("src/mpc/planner.py", "src/mpc/planner.py"),
    ("src/mpc/provider.py", "src/mpc/provider.py"),
    ("src/mpc/target_prediction.py", "src/mpc/target_prediction.py"),
    ("src/mpc/transforms.py", "src/mpc/transforms.py"),
    ("src/dogfight/__init__.py", "src/dogfight/__init__.py"),
    ("src/dogfight/ai/__init__.py", "src/dogfight/ai/__init__.py"),
    ("src/dogfight/ai/action_provider.py", "src/dogfight/ai/action_provider.py"),
    ("src/dogfight/unreal/__init__.py", "src/dogfight/unreal/__init__.py"),
    ("src/dogfight/unreal/client.py", "src/dogfight/unreal/client.py"),
    ("src/dogfight/unreal/protocol.py", "src/dogfight/unreal/protocol.py"),
    ("native/reduced_predictor/CMakeLists.txt", "native/reduced_predictor/CMakeLists.txt"),
    ("native/reduced_predictor/generated_f16_data.h", "native/reduced_predictor/generated_f16_data.h"),
    ("native/reduced_predictor/reduced_predictor.cpp", "native/reduced_predictor/reduced_predictor.cpp"),
    ("native/reduced_predictor/reduced_predictor.h", "native/reduced_predictor/reduced_predictor.h"),
    ("runtime/predictor/Release/MPCJSBSim.dll", "runtime/predictor/Release/MPCJSBSim.dll"),
    ("aircraft/f16/f16.xml", "aircraft/f16/f16.xml"),
    ("engine/F100-PW-229.xml", "engine/F100-PW-229.xml"),
    ("tests/test_mpc_core.py", "tests/test_mpc_core.py"),
    ("tools/build_native.cmd", "tools/build_native.cmd"),
    ("tools/package_team_share.py", "tools/package_team_share.py"),
)

GENERATED = {
    "requirements.txt": "numpy==2.2.6\nPyYAML>=6.0,<7.0\npytest>=8.0,<9.0\n",
    "VERSION.txt": "Release_MPC team share v7\n2026-08-03\n",
}

FORBIDDEN_BASENAMES = {
    "JSBSimAIPLib.dll",
    "AIP_DCS_baseline.dll",
    "Rule_BaselineCore.xml",
}
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".csv", ".log"}


def _validate_output(output: Path) -> None:
    output = output.resolve()
    if output == ROOT or ROOT not in output.parents:
        raise ValueError("output must remain inside Release_MPC")


def _source_path(source: str, destination: str) -> Path:
    """Use authoring paths in Release_MPC and packaged paths after extraction."""
    authoring_path = ROOT / source
    if authoring_path.is_file():
        return authoring_path
    packaged_path = ROOT / destination
    if packaged_path.is_file():
        return packaged_path
    return authoring_path


def _validate_file_list() -> None:
    missing = [
        source for source, destination in FILES
        if not _source_path(source, destination).is_file()
    ]
    if missing:
        raise FileNotFoundError("missing team-share files: " + ", ".join(missing))
    destinations = [destination.replace("\\", "/") for _, destination in FILES]
    destinations.extend(GENERATED)
    if len(destinations) != len(set(destinations)):
        raise RuntimeError("duplicate destination in team-share allow-list")
    forbidden = [
        name for name in destinations
        if Path(name).name in FORBIDDEN_BASENAMES
        or Path(name).suffix.lower() in FORBIDDEN_SUFFIXES
    ]
    if forbidden:
        raise RuntimeError("forbidden artifact entered team-share allow-list: " + ", ".join(forbidden))


def _manifest(entries: dict[str, bytes]) -> bytes:
    lines = [
        f"{hashlib.sha256(content).hexdigest()}  {name}"
        for name, content in sorted(entries.items())
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the minimal Release_MPC teammate source bundle"
    )
    parser.add_argument(
        "--output",
        default="dist/Release_MPC_team_share.zip",
        help="output path relative to Release_MPC",
    )
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    _validate_output(output)
    _validate_file_list()

    entries = {
        destination.replace("\\", "/"): _source_path(source, destination).read_bytes()
        for source, destination in FILES
    }
    entries.update({name: value.encode("utf-8") for name, value in GENERATED.items()})
    entries["MANIFEST_SHA256.txt"] = _manifest(entries)

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in entries.items():
            archive.writestr(name, content)

    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("duplicate entry in generated team-share ZIP")
        if set(names) != set(entries):
            raise RuntimeError("generated ZIP does not match the allow-list")
    print(f"created {output} ({output.stat().st_size} bytes, {len(entries)} entries)")


if __name__ == "__main__":
    main()
