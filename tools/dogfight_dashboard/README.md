# DogFight Unified Dashboard

Local tabbed dashboard for DogFightEnv training metrics and Tacview CSV replay.

## Run

From `DogFightEnv/MyTrainEnv` or `DogFightEnv/Release`:

```powershell
C:\Users\USER\anaconda3\envs\aip\python.exe tools\dashboard.py `
  --training-logdir artifacts\dashboard `
  --replay-logdir logs `
  --port 7860
```

Then open:

```text
http://127.0.0.1:7860
```

## Tabs

- `Training`: scalar charts from `metrics.jsonl` and run `config.json`.
- `Replay`: Three.js/WebGL playback for Blue/Red Tacview CSV pairs.

## Compatibility

- `tools/training_dashboard/server.py --logdir artifacts/dashboard` still works
  and opens the `Training` tab.
- `tools/web_log_viewer.py --logdir <csv-dir>` still works and opens the
  `Replay` tab.
- The old PyVista viewer has been removed from active code paths.

## 판단 근거

- Training metrics and replay logs are both read-only local artifacts, so one
  HTTP server can safely serve both.
- API paths are namespaced under `/api/training/*` and `/api/replay/*` to avoid
  endpoint collisions.
- Keeping compatibility wrappers reduces command churn while making
  `tools/dashboard.py` the preferred entrypoint.

