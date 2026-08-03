# 빌드, 테스트, 대회 연동

## 필수 환경

Prebuilt DLL을 사용할 때:

- Windows x64
- Python 3.10 이상
- NumPy 2.x
- PyYAML 6.x

C++ predictor를 다시 빌드할 때 추가로 필요합니다.

- Visual Studio 2022 C++ toolchain
- CMake 3.20 이상

`tools/build_native.cmd`의 Visual Studio와 CMake 경로는 현재 PC 기준으로 적혀
있습니다. 설치 edition 또는 경로가 다르면 두 경로만 수정하십시오.

## Python 설치와 smoke test

```powershell
python -m pip install -r requirements.txt
python -m pytest tests -q
python student\my_submission.py --help
```

테스트는 다음을 확인합니다.

- body/NED 좌표 변환과 Euler wrap
- YAML/default config 일치
- ATA가 0°에 가까울수록 높은 attack score
- opponent-independent nose advantage
- CEM antithetic sampling과 reset 재현성
- 좌우 대칭 structured candidates
- native rollout 결정성
- FCS actuator/engine 초기 응답

## Native predictor 재빌드

```powershell
cmd.exe /d /c tools\build_native.cmd
python -m pytest tests -q
```

결과 DLL은 다음 위치에 생성됩니다.

```text
runtime/predictor/Release/MPCJSBSim.dll
```

C++ ABI 구조체를 변경했다면 반드시 아래 두 파일을 동시에 맞추십시오.

```text
native/reduced_predictor/reduced_predictor.h
src/mpc/native.py
```

필드 순서, `double`/`int32`, `_pack_=8`이 다르면 계산값이 조용히 손상될 수
있습니다.

## 대회 서버 실행

```powershell
python student\my_submission.py `
  --server-ip <SERVER_IP> `
  --server-port <SERVER_PORT> `
  --team-name <TEAM_NAME> `
  --config configs\mpc.yaml `
  --monitor
```

`student/my_submission.py`는 다음 역할만 합니다.

1. UDP client 생성
2. 공개 packet을 `MPCCommandPolicy`에 전달
3. roll/pitch/yaw_cmd/throttle packet 전송
4. 종료 시 native handle 해제

여기서 protocol의 `yaw_cmd`는 rudder 명령입니다.

## 다른 코드에 MPC만 붙이기

대회 UDP adapter 없이 사용할 때는 `MPCActionProvider`에 `ActionContext`를
전달하면 됩니다.

```python
from pathlib import Path

from dogfight.ai.action_provider import ActionContext
from mpc.config import load_config
from mpc.provider import MPCActionProvider

root = Path(__file__).resolve().parent
config = load_config(root / "configs" / "mpc.yaml")
provider = MPCActionProvider(root, config)

result = provider.compute_action(
    ActionContext(
        sim=None,
        opponent_sim=None,
        ownship_state=own_state,
        target_state=target_state,
    )
)
action = result.action
```

`ownship_state`와 `target_state`의 최소 0:9 항목은 README의 공개 상태 계약과
같아야 합니다. 각속도를 제공할 수 있다면 9:12를 degree/s로 넣고, 시간은
index 41에 second로 넣을 수 있습니다. 대회 adapter는 이 확장 상태를 causal하게
만듭니다.

Episode 시작 시 `provider.reset()`을 호출하고 종료 시 `provider.close()`를
반드시 호출하십시오.

## 설정 변경

가장 자주 조정하는 값은 `configs/mpc.yaml`에 있습니다.

| 항목 | 현재 값 | 의미 |
|---|---:|---|
| `horizon_seconds` | 2.0 | 예측 길이 |
| `knot_seconds` | 0.5 | action 유지 구간 |
| `cem.candidates` | 48 | 반복당 후보 수 |
| `cem.iterations` | 2 | CEM 갱신 횟수 |
| `compute_budget_ms` | 80 | 계획시간 감시 기준 |
| `policy_hz` | 10 | 새 계획 주기 |
| `simulation_hz` | 60 | predictor/서버 주기 |

Horizon이나 knot 수를 늘리면 탐색 차원이 함께 커집니다. 후보 수를 그대로 둔
채 horizon만 늘리는 것이 반드시 더 정확한 것은 아닙니다.

## 공유 ZIP 다시 만들기

```powershell
python tools\package_team_share.py
```

생성물은 `dist/Release_MPC_team_share.zip`입니다. Packager는 allow-list만
포함하고 Baseline/공식 DLL, 로그, checkpoint, build cache가 들어가면 실패하게
되어 있습니다.
