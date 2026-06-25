# DogFight Unified Dashboard

This compatibility entrypoint now opens the `Training` tab of the unified
DogFight dashboard. It still reads scalar training metrics from:

```text
artifacts/dashboard/<run_name>/metrics.jsonl
```

Start it from `Release`:

```bash
python tools/dashboard.py --training-logdir artifacts/dashboard --port 7860
```

The legacy command remains supported:

```bash
python tools/training_dashboard/server.py --logdir artifacts/dashboard --port 7860
```

Then open:

```text
http://127.0.0.1:7860
```

Use the `Training` tab for charts and the `Replay` tab for Tacview CSV playback.
The dashboard remains separate from JSBSim, XML, DLL, and RLlib configuration
code.

## 표시 지표
- Reward/outcome/episode length/count, distance/WEZ, reward component, learner
  stability, value diagnostic은 기본 표시한다.
- DogFightCallbacks가 제공하는 `initial/final ATA`, `initial/final AA`,
  `initial alpha`, `headon_guard_fail`, `ep_altitude_penalty_steps`, action
  axis mean/std도 dashboard metric으로 노출한다.
- SAC/replay/throughput 계열 값은 RLlib result에 존재할 때만 표시된다.
- sidebar의 Metrics 영역은 현재 로그에 있는 지표, 기대했지만 아직 없는 지표,
  chart group에 없는 추가 지표를 구분해 보여준다.

