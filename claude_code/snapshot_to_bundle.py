"""iteration snapshot(.pt) → 제출용 2-파일 번들(metadata.json + policy_weights.pkl.gz).

train.py 는 평가 승률>0.5 인 'best' iteration 을 번들로 저장한다. 특정 iteration(예:
맨 마지막 iter)의 파라미터를 따로 번들로 받고 싶을 때 이 스크립트로 변환한다. snapshot
은 claude_code/models/<name>/<tag>/iter_NNNN.pt 에 매 iter 저장돼 있다(state_dict +
obs_rms + model_kwargs).

예시 (맨 마지막 iter 를 번들로):
  python claude_code/snapshot_to_bundle.py \
    --snapshot claude_code/models/team01/ppo_mlp_v1/iter_0100.pt \
    --output-dir artifacts/models/team01/ppo_mlp_v1_final

--snapshot 대신 --snapshot-dir 만 주면 그 디렉토리에서 가장 큰 iter 번호를 자동 선택한다.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from claude_code import evaluation
from claude_code.model import make_actor_critic, save_bundle


def _latest_snapshot(snap_dir: Path) -> Path:
    snaps = sorted(snap_dir.glob("iter_*.pt"),
                   key=lambda p: int(re.search(r"iter_(\d+)", p.stem).group(1)))
    if not snaps:
        raise FileNotFoundError(f"snapshot 을 찾을 수 없음: {snap_dir}/iter_*.pt")
    return snaps[-1]


def _iter_num(path: Path) -> int:
    m = re.search(r"iter_(\d+)", path.stem)
    return int(m.group(1)) if m else -1


def parse_args():
    p = argparse.ArgumentParser(description="claude_code snapshot(.pt) → 제출 번들 변환")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--snapshot", help="변환할 iter_NNNN.pt 경로")
    g.add_argument("--snapshot-dir", help="이 디렉토리에서 가장 큰 iter 를 자동 선택")
    p.add_argument("--output-dir", required=True, help="번들 저장 디렉토리")
    p.add_argument("--observation-module", default="claude_code.my_observation",
                   help="학습 때 쓴 관측 모듈 경로(메타에 기록; 추론이 동일 관측 재구성)")
    p.add_argument("--reward-module", default="claude_code.my_reward",
                   help="학습 때 쓴 보상 모듈 경로(메타 기록용)")
    return p.parse_args()


def main():
    args = parse_args()
    snap_path = Path(args.snapshot) if args.snapshot else _latest_snapshot(Path(args.snapshot_dir))
    it = _iter_num(snap_path)

    state_dict, model_kwargs, obs_rms_dict = evaluation.load_snapshot(snap_path)
    model = make_actor_critic(**model_kwargs)
    model.load_state_dict(state_dict)
    model.eval()

    # snapshot 의 obs_rms(np 배열) → JSON 직렬화 가능한 state_dict(list) 로.
    rms = evaluation.rms_from_dict(obs_rms_dict)
    obs_norm = rms.state_dict() if rms is not None else None

    # 관측 모듈에서 mode 문자열을 읽어 메타에 정확히 기록.
    obs_mode = "tactical16"
    if args.observation_module:
        from dogfight.ai.student_hooks import load_observation_hook
        obs_mode = load_observation_hook(args.observation_module)["mode"]

    out = Path(args.output_dir)
    save_bundle(
        model, out, obs_norm=obs_norm,
        extra_metadata={
            "reward_module": args.reward_module,
            "observation_module": args.observation_module,
            "observation_mode": obs_mode,
            "selected_iteration": it,
            "source": f"snapshot_to_bundle:{snap_path.name}",
        },
    )
    print(f"[snapshot_to_bundle] iter {it} → 번들 저장 완료: {out}")
    print(f"  - from: {snap_path}")
    print(f"  - obs_size={model.obs_dim} act={model.act_dim} "
          f"num_bins={getattr(model, 'num_bins', 'n/a')} obs_norm={'yes' if obs_norm else 'no'}")
    print("  - metadata.json / policy_weights.pkl.gz")


if __name__ == "__main__":
    main()
