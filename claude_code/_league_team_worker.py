# -*- coding: utf-8 -*-
"""리그 평가용 팀 모델 서브프로세스 워커(stdio 브리지).

각 팀원 패키지(final_team_models/<name>)는 자체 `claude_code`(번들 my_observation·
observation_contract)·`frozen_actor`·`inference` 를 포함한다. 서로 다른 패키지를 한
프로세스에 import 하면 `claude_code.my_observation` 이름이 충돌하므로(먼저 로드된 것이
캐싱됨), **각 팀 모델을 독립 프로세스로 격리**해 각자 번들 코드로 충실하게 돌린다.

프로토콜(stdin/stdout, 한 줄당 JSON):
  부모→자식: {"t":"reset"}                      → {"ok":1}
             {"t":"act","own":[9],"tgt":[9]}    → {"a":[roll,pitch,rudder,throttle]}
             {"t":"quit"}                       → (종료)
own/tgt 는 env 가 provider 에 주는 자기/상대 raw state[:9]=[n,e,d,roll,pitch,yaw,u,v,w].
command 은 inference.FlightPolicy.command 이 내는 [roll,pitch,rudder]∈[-1,1],throttle∈[0,1].
FlightPolicy 는 argmax(sample=False)·10Hz·stateful(에피소드마다 reset)이다.

stdout 은 프로토콜 전용이라, 패키지 로딩/추론 중 라이브러리 print 는 stderr 로 돌린다.
"""
import json
import os
import sys
from pathlib import Path


def main():
    pkg = Path(sys.argv[1]).resolve()
    # 프로토콜 전용 stdout 채널을 확보하고, 이후 모든 print 는 stderr 로 보낸다.
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)                       # fd1 → stderr (라이브러리 print 격리)
    sys.stdout = sys.stderr

    # 패키지 번들 코드를 최우선 경로로(메인 claude_code 보다 먼저) + cwd 이동.
    sys.path[:0] = [str(pkg), str(pkg / "src")]
    os.chdir(str(pkg))
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from inference import FlightPolicy          # 패키지 번들 어댑터(argmax·10Hz·stateful)

    fp = FlightPolicy(device="cpu")
    proto.write(json.dumps({"ready": 1}) + "\n")
    proto.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        t = msg.get("t")
        if t == "quit":
            break
        if t == "reset":
            fp.reset()
            proto.write(json.dumps({"ok": 1}) + "\n")
            proto.flush()
            continue
        if t == "act":
            own = np.asarray(msg["own"], dtype=np.float64)
            tgt = np.asarray(msg["tgt"], dtype=np.float64)
            cmd = np.asarray(fp.command(own, tgt), dtype=np.float64).reshape(-1)[:4]
            proto.write(json.dumps({"a": cmd.tolist()}) + "\n")
            proto.flush()
            continue

    try:
        proto.flush()
    except Exception:
        pass


if __name__ == "__main__":
    main()
