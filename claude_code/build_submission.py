# -*- coding: utf-8 -*-
"""제출용 단일 실행 파일(zip) 빌더.

`claude_code/submission_client.py` 를 PyInstaller(onedir)로 얼려서,
  exe + config.json + 모델 번들 + (altguard 시) Release_MPC_team_share
를 하나의 폴더로 묶고 zip 으로 압축한다. 주최측은 zip 을 풀고 exe 만 실행하면
config.json(상대경로)을 읽어 대회 서버에 접속한다.

사용 예
------
  python claude_code/build_submission.py --mode altguard \
      --team-name team01 --server-ip 221.151.77.208 --server-port 9999

결과: dist/submission_<mode>/DogfightSubmission/  및  dist/submission_<mode>.zip
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MPC_ROOT = ROOT / "Release_MPC_team_share"
BASIC_BUNDLE = ROOT / "artifacts" / "models" / "team01" / "basic"
ENTRY = ROOT / "claude_code" / "submission_client.py"
APP_NAME = "DogfightSubmission"

# 런타임에 sys.path.append 로 동적 로드되거나(mpc/yaml), 함수 내부에서 import 되어
# PyInstaller 정적 분석이 놓칠 수 있는 모듈들을 명시적으로 포함한다.
HIDDEN_IMPORTS = [
    "yaml",
    "GeoMathUtil",
    "mpc", "mpc.config", "mpc.native", "mpc.planner", "mpc.provider",
    "mpc.target_prediction", "mpc.transforms",
    "claude_code.submission_client",
    "claude_code.altguard_provider", "claude_code.altblend_provider",
    "claude_code.action_provider",
    "claude_code.model", "claude_code.my_observation",
    "dogfight.unreal", "dogfight.unreal.client", "dogfight.unreal.protocol",
    "dogfight.ai.action_provider", "dogfight.ai.student_hooks",
    "dogfight.envs.observation",
]

# 제출 UDP 경로는 시뮬레이터가 필요 없다. dogfight.__init__ 의 lazy 처리로 실행 시엔
# 로드되지 않지만, PyInstaller 정적 분석이 lazy 본문의 single_agent_env→JSBSimWrapper
# 를 따라가 JSBSimAIPLib.dll(+디버그 CRT)을 번들하려다 실패하는 것을 원천 차단한다.
EXCLUDES = [
    "FighterSim",
    "JSBSimWrapper",
    "dogfight.envs.single_agent_env",
]


def _ensure_pyinstaller() -> None:
    try:
        import PyInstaller  # noqa: F401
        print(f"PyInstaller 확인됨: {PyInstaller.__version__}")
        return
    except ImportError:
        pass
    print("PyInstaller 미설치 → pip 설치 시작 …")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
    import PyInstaller  # noqa: F401
    print(f"PyInstaller 설치 완료: {PyInstaller.__version__}")


def _make_config(args) -> dict:
    cfg = {
        "server_ip": args.server_ip,
        "server_port": args.server_port,
        "team_name": args.team_name,
        "mode": args.mode,
        "bundle_dir": "model",
        "guard_altitude_ft": 3000.0,
    }
    if args.mode == "altguard":
        cfg["mpc_root"] = "Release_MPC_team_share"
        cfg["action_repeat"] = 1
    elif args.mode == "altblend":
        cfg["mpc_root"] = "Release_MPC_team_share"
        cfg["action_repeat"] = 1
        cfg["blend_hi_ft"] = 4000.0
        cfg["blend_lo_ft"] = 2000.0
    else:
        cfg["action_repeat"] = 6
    return cfg


def _run_pyinstaller(work: Path, dist: Path) -> Path:
    import PyInstaller.__main__ as pyi

    pathex = [str(ROOT), str(SRC), str(MPC_ROOT / "src")]
    cli = [
        str(ENTRY),
        "--name", APP_NAME,
        "--onedir",
        "--noconfirm",
        "--clean",
        "--console",
        "--distpath", str(dist),
        "--workpath", str(work / "build"),
        "--specpath", str(work / "spec"),
        "--copy-metadata", "torch",
        "--copy-metadata", "numpy",
    ]
    for p in pathex:
        cli += ["--paths", p]
    for hi in HIDDEN_IMPORTS:
        cli += ["--hidden-import", hi]
    for ex in EXCLUDES:
        cli += ["--exclude-module", ex]
    print("PyInstaller 실행 …\n  " + " ".join(cli))
    pyi.run(cli)
    app_dir = dist / APP_NAME
    exe = app_dir / (APP_NAME + ".exe")
    if not exe.is_file():
        raise RuntimeError(f"빌드 실패: exe 를 찾을 수 없음 {exe}")
    return app_dir


def _assemble(app_dir: Path, cfg: dict) -> None:
    # config.json
    (app_dir / "config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    # 모델 번들
    dst_model = app_dir / "model"
    if dst_model.exists():
        shutil.rmtree(dst_model)
    shutil.copytree(BASIC_BUNDLE, dst_model)
    print(f"  + model/ ({BASIC_BUNDLE.name})")
    # altguard/altblend MPC 자원(폴더 통째로: predictor DLL + f16 에셋 + configs)
    if cfg["mode"] in ("altguard", "altblend"):
        dst_mpc = app_dir / "Release_MPC_team_share"
        if dst_mpc.exists():
            shutil.rmtree(dst_mpc)
        shutil.copytree(MPC_ROOT, dst_mpc,
                        ignore=shutil.ignore_patterns("dist", "__pycache__", "*.pyc"))
        print("  + Release_MPC_team_share/ (MPCJSBSim.dll + f16 assets + configs)")


def _zip(app_dir: Path, out_zip: Path) -> None:
    if out_zip.exists():
        out_zip.unlink()
    base = out_zip.with_suffix("")
    # app_dir 를 zip 최상위 폴더로 유지
    shutil.make_archive(str(base), "zip", root_dir=str(app_dir.parent),
                        base_dir=app_dir.name)
    print(f"zip 생성: {out_zip} ({out_zip.stat().st_size/1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description="제출 exe/zip 빌더")
    ap.add_argument("--mode", choices=["altguard", "altblend", "basic"], default="altguard")
    ap.add_argument("--team-name", default="team01")
    ap.add_argument("--server-ip", default="221.151.77.208")
    ap.add_argument("--server-port", type=int, default=9999)
    ap.add_argument("--out", default=None, help="출력 루트(기본 dist/submission_<mode>)")
    args = ap.parse_args()

    if not BASIC_BUNDLE.exists():
        raise FileNotFoundError(f"basic 번들 없음: {BASIC_BUNDLE}")
    if args.mode in ("altguard", "altblend") and not MPC_ROOT.exists():
        raise FileNotFoundError(f"MPC 자원 폴더 없음: {MPC_ROOT}")

    _ensure_pyinstaller()

    out_root = Path(args.out).resolve() if args.out else (ROOT / "dist" / f"submission_{args.mode}")
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    work = out_root / "_pyi"
    dist = out_root / "_dist"

    cfg = _make_config(args)
    print(f"=== 제출 빌드: mode={args.mode} team={args.team_name} "
          f"server={args.server_ip}:{args.server_port} ===")

    app_dir = _run_pyinstaller(work, dist)
    _assemble(app_dir, cfg)

    # onedir 를 out_root/DogfightSubmission 로 옮겨 최종 정리
    final_app = out_root / APP_NAME
    if final_app.exists():
        shutil.rmtree(final_app)
    shutil.move(str(app_dir), str(final_app))
    shutil.rmtree(dist, ignore_errors=True)

    out_zip = ROOT / "dist" / f"submission_{args.mode}.zip"
    _zip(final_app, out_zip)

    print("\n완료.")
    print(f"  폴더: {final_app}")
    print(f"  zip : {out_zip}")
    print(f"  config.json: {json.dumps(cfg, ensure_ascii=False)}")
    print("  실행: 압축 해제 후 DogfightSubmission.exe 실행")


if __name__ == "__main__":
    main()
