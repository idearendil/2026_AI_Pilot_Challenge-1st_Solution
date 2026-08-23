# iter1415 학습 reward

`gyLee_reward.py`는 다음 원본을 수정하지 않고 복사한 파일입니다.

```text
ver09_branch_ver08_i0815_outcome100_safelead_atapotential/
└── source/claude_code/gyLee_reward.py
```

SHA-256:

```text
BF99FAA904B0ABBC2634E837C3A9BDFEA00722A6E2B528220A5B5C4763D8A603
```

## 중요한 실행 조건

이 source에는 역사적 `phase1`, `phase2`, 실제 ver09의 `phase3` 프로필이 모두 들어
있습니다. 하위 호환성을 위해 module-level `MY_REWARD_CONFIG` 기본값은 phase1입니다.
iter1415가 사용한 것은 기본값이 아니라 다음 설정입니다.

```text
reward module  = claude_code.gyLee_reward
reward profile = phase3
```

정확한 resolved 설정은 `reward_config_ver09_iter1415.json`에 보존했습니다. 팀원의
launcher가 profile과 override를 따로 받는다면 `phase3`를 선택하고 JSON 값을 전달해야
합니다.

## Import 의존성

원본 파일은 다음을 import합니다.

```python
from dogfight.sim.state_schema import StateIndex
from claude_code.my_observation import ...
```

학습 당시 파일을 그대로 보존하기 위해 이를 `model2_gylee` 상대 import로 바꾸지
않았습니다. 실행하려면 팀원의 `Release` 환경에서 `dogfight`와
`claude_code.my_observation`이 import 가능해야 합니다. ZIP 루트의
`model2_gylee/my_observation.py`는 학습 당시와 동일한 source이므로 필요하면 팀원의
source 배치 규칙에 맞춰 함께 사용하십시오.

고정 opponent로 model2를 구동할 때 reward는 호출되지 않으므로 이 폴더는 실행 필수
의존성이 아니라 학습 재현 자료입니다.
