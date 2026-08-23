# -*- coding: utf-8 -*-
"""collect_exe_dataset 로 모은 데이터로 unreal_bt_client.exe 를 모방하는 actor 를
지도학습(behavioral cloning)한다.

- 입력: exe 관점 claude164r 관측(184D). 출력: 축별(roll/pitch/rudder/throttle) num_bins
  카테고리(= MLPDiscreteActorCritic 정책). exe 의 raw action([-1,1]^4)을 축별 최근접 bin 으로
  이산화해 cross-entropy 로 학습한다.
- **누수 없는 val split**: 수집기는 게임 종료 후에만 샤드로 flush 하므로 한 게임이 두 샤드에
  걸치지 않는다 → **샤드 단위로 train/val 을 나누면 val 에피소드가 train 에 전혀 포함되지
  않는다**(게임 경계 = 샤드 경계). --val-frac 비율의 sample 이 되도록 (셔플된) 샤드를
  통째로 val 로 뺀다.
- 결과는 학습 파이프라인 번들(save_bundle)로 저장 → self-play pool 에 `--target-backend rl`
  (SelfPlayProvider)로 바로 투입 가능. obs_normalization(train 통계)과 claude164r/
  my_observation 메타를 담는다.

예시:
  python claude_code/train_exe_clone.py \
    --dataset-dir claude_code/models/wm/exe_bc_dataset \
    --out-bundle artifacts/models/team01/exe_clone --epochs 8
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    sys.stdout.reconfigure(errors="backslashreplace")
    sys.stderr.reconfigure(errors="backslashreplace")
except Exception:
    pass

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from claude_code.model import make_actor_critic, make_action_grid, save_bundle  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="exe 행동복제(BC) actor 학습")
    p.add_argument("--dataset-dir", default=str(ROOT / "claude_code/models/wm/exe_bc_dataset"))
    p.add_argument("--out-bundle", default=str(ROOT / "artifacts/models/team01/exe_clone"))
    p.add_argument("--num-bins", type=int, default=21, help="축별 이산 bin 수(기본 21)")
    p.add_argument("--hidden", default="512,512,512", help="actor MLP hidden(콤마)")
    p.add_argument("--activation", default="tanh")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--val-frac", type=float, default=0.1, help="val 로 뺄 sample 비율(샤드 단위)")
    p.add_argument("--max-samples", type=int, default=0, help="학습에 쓸 최대 sample(0=전부)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _shard_paths(dataset_dir: Path):
    man = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    paths = [dataset_dir / s.replace("\\", "/") for s in man["shards"]]
    return man, [p for p in paths if p.exists()]


def _load_shards(paths):
    """샤드들을 하나의 (obs, act, src) 로 로드(사전할당으로 메모리 2배 피함)."""
    metas = []
    total = 0
    for p in paths:
        with np.load(p) as z:
            n = int(z["act"].shape[0])
        metas.append((p, n)); total += n
    obs = np.empty((total, 184), dtype=np.float32)
    act = np.empty((total, 4), dtype=np.float32)
    src = np.empty((total,), dtype=np.int8)
    i = 0
    for p, n in metas:
        with np.load(p) as z:
            obs[i:i + n] = z["obs"]; act[i:i + n] = z["act"]; src[i:i + n] = z["src"]
        i += n
    return obs, act, src


def _to_bins(act: np.ndarray, num_bins: int) -> np.ndarray:
    """raw action[-1,1]^4 → 축별 최근접 bin index(0..num_bins-1)."""
    a = np.clip(np.asarray(act, np.float32), -1.0, 1.0)
    idx = np.rint((a + 1.0) * 0.5 * (num_bins - 1)).astype(np.int64)
    return np.clip(idx, 0, num_bins - 1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device(args.device)
    dataset_dir = Path(args.dataset_dir)
    man, paths = _shard_paths(dataset_dir)
    if not paths:
        raise FileNotFoundError(f"샤드를 찾을 수 없습니다: {dataset_dir}")

    # 샤드 sample 수 파악 → 셔플 후 val-frac 만큼(누적 sample) 샤드 통째로 val 로.
    counts = []
    for p in paths:
        with np.load(p) as z:
            counts.append(int(z["act"].shape[0]))
    total = sum(counts)
    rng = np.random.default_rng(args.seed)
    order = list(rng.permutation(len(paths)))
    val_target = int(total * float(np.clip(args.val_frac, 0.01, 0.5)))
    val_idx, acc = [], 0
    for j in order:
        if acc < val_target:
            val_idx.append(j); acc += counts[j]
    val_set = set(val_idx)
    val_paths = [paths[j] for j in range(len(paths)) if j in val_set]
    tr_paths = [paths[j] for j in range(len(paths)) if j not in val_set]
    print(f"[bc] 샤드 {len(paths)}개(총 {total:,} sample) → train {len(tr_paths)}개 / "
          f"val {len(val_paths)}개(약 {acc:,} sample, 게임 경계=샤드 경계라 누수 없음)")

    hidden = tuple(int(x) for x in args.hidden.split(","))
    nb = int(args.num_bins)

    print("[bc] val 로드...", flush=True)
    Xv, Av, _ = _load_shards(val_paths)
    Yv = _to_bins(Av, nb)
    print("[bc] train 로드...", flush=True)
    Xt, At, _ = _load_shards(tr_paths)
    Yt = _to_bins(At, nb)
    if args.max_samples and Xt.shape[0] > args.max_samples:
        sel = rng.permutation(Xt.shape[0])[:args.max_samples]
        Xt, Yt = Xt[sel], Yt[sel]
    print(f"[bc] train {Xt.shape[0]:,} / val {Xv.shape[0]:,}  obs_dim=184  bins={nb}")

    # obs 정규화 통계는 **train 에서만** 산출(누수 방지).
    xm = Xt.mean(0).astype(np.float64)
    xv = Xt.var(0).astype(np.float64) + 1e-8
    xm32 = xm.astype(np.float32)
    xstd32 = np.sqrt(xv).astype(np.float32)

    def norm_inplace(x):
        x -= xm32; x /= xstd32; np.clip(x, -10.0, 10.0, out=x)
        return x

    # in-place 정규화(복사본 없이) 후 CPU 텐서. 배치마다 GPU 로 전송(obs 통째 적재 회피).
    Xt_t = torch.from_numpy(norm_inplace(Xt)); Yt_t = torch.from_numpy(Yt)
    Xv_t = torch.from_numpy(norm_inplace(Xv)).to(dev); Yv_t = torch.from_numpy(Yv).to(dev)
    del At, Av

    model = make_actor_critic(obs_dim=184, act_dim=4, hidden=hidden,
                              activation=args.activation, critic_hidden=hidden,
                              critic_activation=args.activation, num_bins=nb).to(dev)
    # actor(정책) 파라미터만 학습(critic 은 opponent 로 안 쓰임).
    actor_params = [p for n, p in model.named_parameters() if "actor" in n]
    opt = torch.optim.AdamW(actor_params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    lossf = nn.CrossEntropyLoss()
    B = int(args.batch_size)

    def _logits(x):
        return model.actor_logits(x).view(-1, 4, nb)

    @torch.no_grad()
    def evaluate():
        model.eval()
        tot = 0.0; correct = np.zeros(4); n = 0
        for i in range(0, Xv_t.shape[0], B):
            xb = Xv_t[i:i + B]; yb = Yv_t[i:i + B]
            lg = _logits(xb)
            loss = sum(lossf(lg[:, k, :], yb[:, k]) for k in range(4)) / 4.0
            tot += float(loss) * xb.shape[0]
            pred = lg.argmax(-1)
            for k in range(4):
                correct[k] += float((pred[:, k] == yb[:, k]).sum())
            n += xb.shape[0]
        return tot / max(n, 1), correct / max(n, 1)

    ntr = Xt_t.shape[0]
    best = float("inf"); best_state = None; t0 = time.perf_counter()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(ntr)
        run = 0.0
        for i in range(0, ntr, B):
            j = perm[i:i + B]
            xb = Xt_t[j].to(dev, non_blocking=True)
            yb = Yt_t[j].to(dev, non_blocking=True)
            lg = _logits(xb)
            loss = sum(lossf(lg[:, k, :], yb[:, k]) for k in range(4)) / 4.0
            opt.zero_grad(); loss.backward(); opt.step()
            run += float(loss) * xb.shape[0]
        sched.step()
        vl, vacc = evaluate()
        tr_loss = run / ntr
        print(f"  ep{ep+1:2d}/{args.epochs}  train_ce={tr_loss:.4f}  val_ce={vl:.4f}  "
              f"val_acc[r,p,rud,thr]={np.round(vacc,3).tolist()}  "
              f"lr={sched.get_last_lr()[0]:.2e}  ({time.perf_counter()-t0:.0f}s)", flush=True)
        if vl < best:
            best = vl; best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    vl, vacc = evaluate()
    print(f"[bc] best val_ce={best:.4f}  val_acc[r,p,rud,thr]={np.round(vacc,3).tolist()}")

    # 번들로 저장(claude164r/my_observation + train obs_rms).
    out = Path(args.out_bundle)
    # obs_normalization 은 {mean,var,count} 포맷이어야 SelfPlayProvider(RunningMeanStd)가 로드됨.
    obs_norm = {"mean": xm.tolist(), "var": xv.tolist(), "count": float(ntr)}
    save_bundle(model.to("cpu"), out, obs_norm=obs_norm,
                extra_metadata={"observation_mode": "claude164r",
                                "observation_module": "claude_code.my_observation",
                                "source": "behavioral_cloning:unreal_bt_client.exe",
                                "bc_dataset": str(dataset_dir),
                                "bc_val_ce": float(best),
                                "bc_val_acc": [float(x) for x in vacc]})
    print(f"[bc] 번들 저장: {out}  (self-play pool 에 --target-backend rl --target-bundle-dir "
          f"{out} 로 투입 가능)")


if __name__ == "__main__":
    main()
