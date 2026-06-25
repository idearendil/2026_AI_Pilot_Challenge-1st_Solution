# DogFight Replay Viewer Compatibility Layer

`tools/web_log_viewer.py` now opens the `Replay` tab of the unified DogFight
dashboard. The parser and replay data API remain in this package so existing
Tacview CSV playback contracts continue to work.

## Run

From `DogFightEnv/MyTrainEnv`:

```powershell
C:\Users\USER\anaconda3\envs\aip\python.exe tools\web_log_viewer.py --port 7870
```

From `DogFightEnv/Release`:

```powershell
C:\Users\USER\anaconda3\envs\aip\python.exe tools\web_log_viewer.py --port 7870
```

Then open:

```text
http://127.0.0.1:7870/?tab=replay
```

The preferred all-in-one entrypoint is:

```powershell
C:\Users\USER\anaconda3\envs\aip\python.exe tools\dashboard.py --port 7860
```

## 판단 근거

- PyVista has been removed from the active viewer path.
- Three.js-based replay runs inside the unified dashboard with the same F-16 OBJ
  asset and Tacview CSV parser.
- Keeping this wrapper avoids breaking existing replay commands while moving
  users toward the tabbed dashboard.

