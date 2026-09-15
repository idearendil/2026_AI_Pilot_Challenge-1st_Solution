# -*- coding: utf-8 -*-
"""제출용 단일 실행 파일(zip) 빌더 (순수 학습모델).

`claude_code/submission_client.py` 를 PyInstaller(onedir)로 얼려서,
  exe + config.json + 학습 번들(model/)
을 하나의 폴더로 묶고 zip 으로 압축한다. 주최측(또는 로컬 BattleServer_V1.2_VeryLow)은
zip 을 풀고 exe 만 실행하면 config.json(상대경로)을 읽어 서버에 접속한다.

CPU 학습 번들(snapshot_to_bundle)·CUDA 학습 번들(gpu_ckpt_to_bundle) 모두 지정 가능하다.
제어 주기(10/60Hz)와 argmax/stochastic 은 config.json(control_hz·deterministic)에서 정한다.

사용 예
------
  # CUDA 학습 번들, 10Hz stochastic, 로컬 BattleServer 확인
  python claude_code/build_submission.py --bundle-dir artifacts/gpu_ppo_final \
      --team-name team01 --server-ip 127.0.0.1 --server-port 9999
  # 60Hz argmax
  python claude_code/build_submission.py --bundle-dir artifacts/cpu_ppo_final \
      --control-hz 60 --deterministic

결과: dist/submission/DogfightSubmission/  및  dist/submission.zip
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
ENTRY = ROOT / "claude_code" / "submission_client.py"
APP_NAME = "DogfightSubmission"

# 함수 내부에서 import 되어 PyInstaller 정적 분석이 놓칠 수 있는 모듈들을 명시적으로 포함한다.
HIDDEN_IMPORTS = [
    "GeoMathUtil",
    "claude_code.submission_client",
    "claude_code.action_provider",
    "claude_code.high_rate",
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
    return {
        "server_ip": args.server_ip,
        "server_port": args.server_port,
        "team_name": args.team_name,
        "bundle_dir": "model",
        "control_hz": int(args.control_hz),        # 10(action_repeat=6) | 60(action_repeat=1)
        "deterministic": bool(args.deterministic),  # true=argmax | false=stochastic
    }


def _run_pyinstaller(work: Path, dist: Path) -> Path:
    import PyInstaller.__main__ as pyi

    pathex = [str(ROOT), str(SRC)]
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


def _assemble(app_dir: Path, cfg: dict, bundle_dir: Path) -> None:
    # config.json
    (app_dir / "config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    # 모델 번들
    dst_model = app_dir / "model"
    if dst_model.exists():
        shutil.rmtree(dst_model)
    shutil.copytree(bundle_dir, dst_model)
    print(f"  + model/ ({bundle_dir.name})")


def _zip(app_dir: Path, out_zip: Path) -> None:
    if out_zip.exists():
        out_zip.unlink()
    base = out_zip.with_suffix("")
    # app_dir 를 zip 최상위 폴더로 유지
    shutil.make_archive(str(base), "zip", root_dir=str(app_dir.parent),
                        base_dir=app_dir.name)
    print(f"zip 생성: {out_zip} ({out_zip.stat().st_size/1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description="제출 exe/zip 빌더 (순수 학습모델)")
    ap.add_argument("--bundle-dir", required=True,
                    help="학습 번들 경로(CPU snapshot_to_bundle 또는 CUDA gpu_ckpt_to_bundle 산출물)")
    ap.add_argument("--control-hz", type=int, choices=[10, 60], default=10,
                    help="제어 주기(10=action_repeat 6, 60=action_repeat 1). 기본 10")
    ap.add_argument("--deterministic", action="store_true",
                    help="action 을 argmax 로(기본은 stochastic 샘플링)")
    ap.add_argument("--team-name", default="team01")
    ap.add_argument("--server-ip", default="127.0.0.1",
                    help="대회 서버 IP(로컬 BattleServer 확인 시 127.0.0.1)")
    ap.add_argument("--server-port", type=int, default=9999)
    ap.add_argument("--out", default=None, help="출력 루트(기본 dist/submission)")
    args = ap.parse_args()

    bundle_dir = Path(args.bundle_dir).resolve()
    if not bundle_dir.exists():
        raise FileNotFoundError(f"번들 없음: {bundle_dir}")

    _ensure_pyinstaller()

    out_root = Path(args.out).resolve() if args.out else (ROOT / "dist" / "submission")
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    work = out_root / "_pyi"
    dist = out_root / "_dist"

    cfg = _make_config(args)
    print(f"=== 제출 빌드(순수 학습모델): team={args.team_name} "
          f"{args.control_hz}Hz {'argmax' if args.deterministic else 'stochastic'} "
          f"server={args.server_ip}:{args.server_port} ===")

    app_dir = _run_pyinstaller(work, dist)
    _assemble(app_dir, cfg, bundle_dir)

    # onedir 를 out_root/DogfightSubmission 로 옮겨 최종 정리
    final_app = out_root / APP_NAME
    if final_app.exists():
        shutil.rmtree(final_app)
    shutil.move(str(app_dir), str(final_app))
    shutil.rmtree(dist, ignore_errors=True)

    out_zip = ROOT / "dist" / "submission.zip"
    _zip(final_app, out_zip)

    print("\n완료.")
    print(f"  폴더: {final_app}")
    print(f"  zip : {out_zip}")
    print(f"  config.json: {json.dumps(cfg, ensure_ascii=False)}")
    print("  실행: 압축 해제 후 DogfightSubmission.exe 실행")


if __name__ == "__main__":
    main()
