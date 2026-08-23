# model2_gylee

최근 활성 계보 `ver09`에서 가장 최근에 완전히 저장되어 있던 durable checkpoint
`iter_1415.pt`를 다른 팀원의 DogFightEnv opponent pool에서 사용하기 위한 최소 패키지입니다.

> 이 모델은 최신 학습 snapshot이지, 별도 리그를 거쳐 확정한 final best는 아닙니다.
> 선발된 안정 champion은 별도 `model1_gylee`의 ver05 iter1220입니다.

상세 학습 이력, reward, opponent 분포, observation/action 계약과 통합 주의사항은
[`MODEL_CARD_AND_USAGE.md`](MODEL_CARD_AND_USAGE.md)에 정리되어 있습니다.

실제 학습 당시의 reward source도 `training_reward/gyLee_reward.py`에 **내용을 바꾸지
않은 원본 그대로** 포함했습니다. `phase1/2/3`이 모두 들어 있는 모듈이므로 iter1415를
재현할 때는 반드시 `training_reward/reward_config_ver09_iter1415.json`의
`reward_profile: phase3` 설정을 사용해야 합니다. 고정 opponent 추론에는 reward가
필요하지 않습니다.

## 빠른 설치

압축을 팀원의 DogFightEnv `Release` 루트에 풀어 `Release/model2_gylee/`가 되게 합니다.
필요한 외부 항목은 팀원이 이미 가진 동일 `src/dogfight`, `GeoMathUtil`, JSBSim과
Python `numpy`, `torch`입니다.

```powershell
Set-Location '<팀원 DogFightEnv Release 경로>'
python -m model2_gylee.verify
```

## Opponent 연결

```python
from model2_gylee import make_opponent_provider

opponent = make_opponent_provider(step_ratio=6, explore=False)
env._target_action_provider = opponent
```

- `explore=False`: deterministic argmax, 비교평가와 고정 opponent 권장
- `explore=True`: 각 action 축 categorical 분포에서 stochastic sampling
- 병렬 학습에서는 worker/environment마다 provider를 새로 만들어야 합니다.
- 매 episode마다 환경이 `provider.reset()`을 호출해야 합니다.

원래 프로젝트와 동일한 snapshot loader가 있으면 `model/iter_1415.pt`만 등록할 수도
있지만, 47D observation과 RMS, 상대 관점 재구성, action 변환 계약이 완전히 같은지
확신할 수 없다면 이 패키지 provider를 그대로 사용하는 것이 안전합니다.
