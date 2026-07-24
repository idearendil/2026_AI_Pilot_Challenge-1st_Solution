# -*- coding: utf-8 -*-
"""크래시 자동 재시작 supervisor (PPO/REDQ 공용).

Windows 에서 CUDA + Ray 를 한 프로세스에서 쓰면 몇 번의 iteration 후 네이티브 access
violation 이 날 수 있다(Ray 백그라운드 스레드 ↔ CUDA context 충돌). 이를 in-process 로는
못 잡으므로, 바깥 supervisor 가 학습을 **자식 프로세스**로 띄우고 크래시하면 마지막
체크포인트에서 자동 재시작한다.

핵심 구성:
  - start_heartbeat: 자식이 시작 즉시 데몬 스레드로 heartbeat 파일에 타임스탬프를 찍는다.
    메인 스레드가 무거운 update 로 ray.get 에 블록돼 있어도(GIL 해제) 계속 tick 하므로,
    supervisor 는 '느리지만 살아있음'과 '죽음/hang'을 구분할 수 있다. 프로세스가 죽으면
    (access violation) heartbeat 도 멈춘다.
  - supervise_loop: 자식을 반복 실행. (a) 자식 비정상 종료(즉시 감지) 또는 (b) heartbeat
    가 timeout 초 동안 정지(죽음/hang) 시 트리를 죽이고 재시작. 재시작 전 ray stop 으로
    orphan Ray 프로세스를 정리한다. 자식이 0 으로 끝나면(총 iteration/step 도달) 종료.

자식은 --auto-resume 로 실행돼, 시작 시 체크포인트가 있으면 스스로 이어서 학습한다
(크래시난 iteration 은 다음 시작 시 처음부터 다시 돈다).
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path


def start_heartbeat(path, interval: float = 5.0):
    """데몬 스레드로 interval 초마다 heartbeat 파일에 타임스탬프를 찍는다.

    반환된 stop 이벤트를 set() 하면 스레드가 종료된다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()

    def _beat():
        while True:
            try:
                path.write_text(str(time.time()))
            except Exception:
                pass
            if stop.wait(interval):
                break

    threading.Thread(target=_beat, daemon=True).start()
    return stop


def kill_tree(proc):
    """자식 프로세스 트리 전체 종료(Ray worker/actor 포함). Windows 는 taskkill /T."""
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True)
        else:
            proc.kill()
    except Exception:
        pass


def ray_stop():
    """죽은 자식이 남긴 Ray 프로세스(raylet/gcs/plasma/worker)를 정리한다.

    현재 파이썬의 ray 콘솔 스크립트를 우선 쓰고, 없으면 PATH 의 ray 로 폴백."""
    py = Path(sys.executable)
    cands = [py.parent / "Scripts" / "ray.exe", py.parent / "ray.exe",
             py.parent / "bin" / "ray", py.with_name("ray")]
    ray_cmd = next((str(c) for c in cands if c.exists()), "ray")
    try:
        subprocess.run([ray_cmd, "stop", "--force"], capture_output=True, timeout=60)
    except Exception:
        pass


def supervise_loop(cmd, heartbeat_path, timeout_s: float = 120.0,
                   max_restarts: int = 200, label: str = "train"):
    """cmd(자식, --auto-resume 포함)를 반복 실행하며 크래시/hang 시 재시작한다.

    heartbeat_path 가 timeout_s 초 동안 안 바뀌면 죽음/hang 으로 보고 트리를 죽여 재시작.
    자식이 스스로 0 으로 종료하면 완료로 보고 반환한다."""
    hb = Path(heartbeat_path)
    for attempt in range(1, int(max_restarts) + 1):
        if attempt > 1:
            ray_stop()   # 이전(죽은) 자식의 Ray orphan 정리
        print(f"[{label}/supervise] 실행 #{attempt}/{max_restarts}", flush=True)
        proc = subprocess.Popen(cmd)
        last_mtime = hb.stat().st_mtime if hb.exists() else None
        last_progress = time.time()
        rc = None
        while True:
            try:
                rc = proc.wait(timeout=5.0)
                break                       # 자식이 스스로 종료(정상 or 크래시)
            except subprocess.TimeoutExpired:
                pass
            m = hb.stat().st_mtime if hb.exists() else None
            if m != last_mtime:
                last_mtime, last_progress = m, time.time()
            elif time.time() - last_progress > timeout_s:
                print(f"[{label}/supervise] heartbeat {timeout_s:.0f}s 정지(죽음/hang) "
                      f"→ 트리 종료 후 재시작", flush=True)
                kill_tree(proc)
                proc.wait()
                rc = -1
                break
        if rc == 0:
            print(f"[{label}/supervise] 정상 종료. 실행 #{attempt}.", flush=True)
            return
        print(f"[{label}/supervise] 자식 종료 code={rc} → 체크포인트에서 재시작.", flush=True)
    print(f"[{label}/supervise] 최대 재시작({max_restarts}) 초과. 중단.", flush=True)


def build_child_cmd(entry_file):
    """현재 argv 에서 --supervise 를 빼고 --auto-resume 를 넣은 자식 실행 커맨드."""
    child = [a for a in sys.argv[1:] if a != "--supervise"]
    if "--auto-resume" not in child:
        child.append("--auto-resume")
    return [sys.executable, str(Path(entry_file).resolve())] + child


__all__ = ["start_heartbeat", "kill_tree", "ray_stop", "supervise_loop", "build_child_cmd"]
