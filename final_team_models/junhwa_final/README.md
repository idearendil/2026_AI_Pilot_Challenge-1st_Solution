# 모델 실행 안내 — 2026-09-11

## 파일 선택

- `3-9/gylee_20k.zip`: 기존 20,000 모델. 원래 3-9·Head-on 혼합 학습, 184→214 입력 호환 변환본.
- `3-9/gylee_69k.zip`: 3-9 학습 69,000 모델.
- `Head-on/gylee_20k.zip`: 위 20k와 동일한 가중치·정규화.
- `Head-on/gylee_46k_headon.zip`: Head-on 학습의 마지막 저장점 46,000. 이전 공유의 44,000 리그 선정본과 다릅니다.
- `3-9/junhwa_*.zip`: final / grid_3L749_gru_last / wide1_v2 / exploiter_survive_v1 네 모델. 원본은 두 시나리오 공용으로 전달받았습니다.

각 ZIP을 별도 폴더에 풉니다. `model.pt`는 **actor 가중치 + cfg + norm(mean/var/count)**를 모두 포함합니다. `metadata.json`, `inference.py`, `frozen_actor.py`, 관측 재구성 코드와 함께 사용하세요. 학습 optimizer·pool은 제외했습니다. 체크포인트의 actor는 손실 없이 추출했고, 키 이름만 공통 어댑터에 맞췄습니다. 검증 내역은 `verification.json`, 원본·배포 파일 해시는 `metadata.json`에 있습니다.

## 실행과 인터페이스

Python 3.11, PyTorch 2.7.0, NumPy 2.2.6에서 검증합니다. 압축 해제 폴더에서:

```sh
python -m pip install -r requirements.txt
python inference.py
```

이미 동일한 214D 관측을 만드는 평가기에 연결할 때:

```python
from inference import Policy
p = Policy(device='cpu')  # CUDA 가능: 'cuda'
state = p.initial_state(batch_size=1)
command, state = p.act(raw_obs_214, state, episode_start=[True])
# 다음 step부터 episode_start=[False], 반환된 state를 계속 전달
# command[0] = [roll, pitch, rudder, throttle]
```

## 반드시 일치시킬 것

1. **관측**: 입력은 214D이며 순서·단위·history가 중요합니다. 동봉 `claude_code/my_observation.py`가 기준입니다. 이미 feature 단위로 스케일된 **추가 running 정규화 전**의 관측을 `Policy.act()`에 넣습니다. 차원만 같은 임의 관측은 호환되지 않습니다.
2. **정규화**: 어댑터가 `clip((obs-mean)/sqrt(var+1e-8), -10, 10)`을 한 번 적용합니다. 모델별 저장 통계를 그대로 고정하고, 평가 중 갱신하거나 다른 모델의 통계로 대체하지 마세요. 외부에서 다시 정규화하면 이중 적용됩니다.
3. **행동**: Tanh 네트워크, 4개 채널 × 21개 bin을 각각 argmax로 선택합니다. roll/pitch/rudder는 [-1,1], throttle은 [0,1]로 변환해 반환합니다. 반환값을 다시 bin으로 해석하거나 throttle을 재변환하지 마세요.
4. **시간/상태**: 정책은 10Hz(0.1초). 60Hz 시뮬레이터라면 6프레임 유지하며, 이미 10Hz인 평가기에는 추가 repeat가 없습니다. 에피소드마다 관측 history·HP/fuel/time 복원 상태와 GRU hidden state를 초기화합니다. GRU 모델은 각 비행기·배치 lane별 hidden state를 유지해야 합니다.
5. **상태에서 관측 만들기**: gylee는 동봉 `CUDAObservationState`의 첫 관측/이후 갱신 순서를 사용합니다. `FlightPolicy.command(own_state,target_state)`가 이를 감싸며 10Hz에 한 번 호출합니다. state는 `[north,east,down,roll,pitch,yaw,u,v,w,...]`(m, degree, m/s). 위치는 NED, **속도 u/v/w는 기체 Body 좌표**입니다. 서버 좌표를 이 계약으로 변환하세요. 새 경기에는 `FlightPolicy.reset()`을 호출합니다. 모든 고도에서 신경망을 사용하며 자동 BT 전환은 없습니다.
6. **시나리오**: Head-on 평가 시작 거리는 5,539m입니다. 평가 기준은 최대 200초, hard deck 304.8m입니다. 모델 파일이 초기조건을 정하지 않으므로 서버/환경에서 3-9 또는 Head-on을 지정합니다.

Junhwa는 제공 cfg·norm을 보존하고 현재 평가 어댑터와 동일하게 실행됩니다. 공유 214D 관측 계약은 제공자의 설명에 근거하며, 제공자 원본 서버의 전체 재구성과 직접 대조한 것은 아닙니다. 동봉 smoke test는 실행·상태관리 검사이고 경기 성능 보장은 아닙니다. 20k의 입력 호환 변환 내역은 해당 패키지 메타데이터에 기록됩니다.
