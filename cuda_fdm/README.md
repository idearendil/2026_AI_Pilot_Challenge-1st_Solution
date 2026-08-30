# cuda_fdm — F16 JSBSim v1.0.0 CUDA 포팅

목표: 대회환경에 맞춘 CPU JSBSim v1.0.0 비행물리를 **평탄화(trace & flatten)**하여
GPU에서 수천 env 병렬 스텝 → PPO 데이터수집 가속.

## 현재 상태 (P2 완료)
JSBSim v1.0.0 의 f16 단일기 데이터플로 전체를 **검증된 double Python 레퍼런스**로 재현.
`ref/jsb_fdm.py` 가 6초 4채널 기동에서 golden(=CPU v1.0.0)과 일치:

| 지표 | 오차 (6초 전체) |
|---|---|
| 위치 | 0.036 ft (11 mm) |
| 자세 | 0.0030 deg |
| 속도 | 0.017 fps (5 mm/s) |
| 받음각 | 0.0005 deg |

모든 최대오차가 마지막 스텝(row360)에 몰려 있음 = 과도 없이 순수 double 누적드리프트만 남음.
이 Python 로직이 곧 CUDA 커널 스펙. bit-exact는 CPU/GPU간 FMA·libm 차이로 불가,
tolerance 기반(≈mm/µdeg)이 현실 목표이며 RL 전이엔 무의미한 차이.

## 구조
- `ref/` — 검증된 서브시스템 (그대로 CUDA-C 번역 대상)
  - `jsb_const.py` 물리/단위 상수
  - `jsb_tables.py` FGTable 1D/2D 보간 (외삽 없음, 끝값 클램프)
  - `jsb_atmos.py` 표준대기(지오포텐셜 층상ISA) + density-altitude
  - `jsb_fcs.py` **전체 F16 FCS** (kinematic/PID/aerosurface/summer/switch, 10채널) — ★DT=1/120 gotcha
  - `jsb_massbalance.py` 질량/CG/관성텐서(v1.0.0 ixz규약)/Jinv
  - `jsb_propulsion.py` F100 터빈(스풀+afterburner+연료) + direct thruster
  - `jsb_auxiliary.py` alpha/beta/qbar/mach/n-pilot/pqr-aero/vground
  - `jsb_aero.py` 40 계수함수 wind→body 힘 + r×F 모멘트
  - `jsb_frames.py` ECI/ECEF/Local/Body 변환 + WGS84 측지(Fukushima) + J2중력 + 사원수
  - `jsb_accel.py` F=ma + 원심/코리올리 + 중력
  - `jsb_fdm.py` **통합 스텝 루프** (Propagate 적분 → 힘경로 → Accelerations)
  - `f16_aero_data.py` (자동생성) 공력 테이블/함수
- `gen/extract_f16.py` f16.xml aerodynamics → f16_aero_data.py 추출기
- `tests/val_*.py` 레이어별 golden 대조 (atmos_aux/fcs/aero/accel/fulltraj)

오라클: `<scratchpad>/golden_trace.py` → `golden_trace.csv` (v1.0.0 매스텝 106컬럼 덤프).

## 반드시 지킬 gotcha (포팅 함정)
1. **FCS dt = 1/120** (적분 dt 1/60의 절반). FGFCSComponent.dt가 load_model 시점 기본 dt를 고정 캡처. 모든 kinematic/PID slew에 적용.
2. **FGMatrix33 data[] = column-major**. GetEuler/GetQuaternion 포팅시 data[k]=m[k%3][k//3].
3. **중력 = WGS84 J2** (지구자전 ECI 적분). altitudeASL = |eci| − a(20925646.32546), 측지고도 아님.
4. 적분기: 사원수·vPQRi = RectEuler, eci_pos = AB3, eci_vel = AB2 (다단 히스토리 deque 5-deep, IC 5×복제).
5. throttle_pos = 2×throttle_cmd → 0.8 이면 military 1.0 + afterburner 0.6.
6. 관성 baseJ 에 ixz 원부호(-982) 사용 (v1.0.0 규약).
7. 모델순서: Propagate→Inertial→Atmosphere→FCS→MassBalance→Auxiliary→Propulsion→Aero→Accelerations. FCS는 alpha/mach/n-pilot/pqr-aero 를 전프레임, attitude만 현재프레임.

## P3 완료 — CUDA-C 포팅 + GPU 검증 (2026-08-30)
검증된 `jsb_*.py` 를 **host/CUDA 공용 C**(`gen/fdm.cuh` + 자동생성 `gen/f16_gen.cuh`)로 이식.
같은 소스를 (1) g++ 호스트 컴파일 (2) NVRTC 디바이스 컴파일 → 둘 다 golden 대조.

| 대조 | pos | att | vel | alpha |
|---|---|---|---|---|
| host-C(g++) vs golden | 0.0356 ft | 0.0030° | 0.017 fps | 0.00055° |
| GPU(NVRTC) vs golden | 0.0356 ft | 0.0030° | 0.017 fps | 0.00055° |
| **GPU vs host-C** | **0 ft** | 2.9e-15 rad | 1.1e-12 fps | 5.5e-16 rad |

`--fmad=false` 로 CPU/GPU 사실상 bit-동일. Python 레퍼런스와도 동일 수준.
**성능(RTX 3070 Ti, FP64)**: 262k env → **48 M env-steps/s** (8k=32M, 65k=44M).

### 파일 (gen/, tests/)
- `gen/gen_c_fdm.py` — ref 데이터 → `gen/f16_gen.cuh` 자동생성(상수·41테이블·공력40함수 평탄화). **CUDA 툴킷 불필요**.
- `gen/fdm.cuh` — 핵심 로직(frames/quat/atmos/mass/fcs/prop/aux/accel + 통합 `fdm_step`). host/device 매크로 스위치.
- `gen/fdm_kernel.cu` — `__global__ fdm_run`(스레드=env, N step 내부루프, env0 궤적기록).
- `tests/export_seed.py` — `FDM.seed` 재사용 → `_bin/{seed.bin(101d),actions.bin,golden_ref.bin}`.
- `tests/host_val.cpp` — seed→FdmState(memcpy 101 double)→N step→`out_c.bin`. g++.
- `tests/cuda_rt.py` — NVRTC(torch 번들 `nvrtc64_130_0.dll`) + CUDA Driver API(nvcuda.dll) ctypes 런타임. 메모리는 torch cuda 텐서.
- `tests/gpu_val.py [nenv]` — NVRTC 컴파일→nenv env 병렬→`out_gpu.bin`+성능.
- `tests/compare.py <out.bin>` — golden 대조.

### 실행
```
# aip: 검증 레퍼런스        aip_gpu: torch+CUDA
D:\...\envs\aip\python.exe   cuda_fdm/gen/gen_c_fdm.py         # 헤더 재생성
D:\...\envs\aip\python.exe   cuda_fdm/tests/export_seed.py     # 시드/골든 bin
PATH=+ucrt64/bin; g++ -O2 -Igen tests/host_val.cpp -o host_val.exe; ./host_val.exe
D:\...\envs\aip\python.exe   cuda_fdm/tests/compare.py out_c.bin
D:\...\envs\aip_gpu\python.exe cuda_fdm/tests/gpu_val.py 4     # GPU 검증
D:\...\envs\aip\python.exe   cuda_fdm/tests/compare.py out_gpu.bin
```
- 환경: **aip_gpu**(torch 2.13+cu130, CUDA13, RTX 3070 Ti). **nvcc/CUDA툴킷 없이 NVRTC로 컴파일**.
- g++ 는 msys2 ucrt64 — 실행시 `D:/other_programs/msys64/ucrt64/bin` 을 PATH 에 넣어야 런타임 DLL 로드됨.

## P4 완료 — PyTorch 배치 env + IC reset (2026-08-30)
`gpu_env.py` `GpuDogfight`: 1v1=2기/env(스레드=aircraft, nac=2·nenv). NVRTC `fdm_step_batch`
커널(action_repeat=substeps 내부루프) + torch cuda 텐서 상태.
- API: `reset_ic(ic_list)` / `load_seed(seed)` / `step(actions,substeps)` → obs(nenv,ppe,17) / `get_state` / `get_obs`.
- obs 17: eci_pos3,eci_vel3,euler3,vUVW3,alpha,beta,mach,Vt,alt_asl.
- **검증**: 배치 step() 경로 env0/plane0 = golden 0.0356ft/0.003°/0.017fps(P3와 동일). 2기 독립·env 결정성 OK.
- **성능 step()**: nenv4096(8192기) 0.34ms/step=24M ac-steps/s (ss4=28.6M); 16k env 30~37M. 런치오버헤드 대규모서 무시.

`ic.py` — 대회 IC(f16_init.xml: lat/lon/alt/vt/gamma/phi/psi/theta/alpha/beta) → 초기 ECI seed.
- **지구모델 판명**: 위치 = **geodetic 방향 + 반경 |eci|=SEMI_MAJOR+alt**(altitudeASL 규약). `<latitude>`=geodetic.
  법선 위 반경조건 t 를 2차식으로 풀어 golden eci_pos 를 **9e-9 ft**, eci_vel **6e-13 fps** 재현.
- body vel: u=vt·cosα·cosβ, v=vt·sinβ, w=vt·sinα·cosβ; eci_vel=Tb2i·vUVW+ω×eci_pos. 자세는 ref seed 체인.
- `build_seed_vector(...)` → 101-double(CPU ref FDM.seed, 에피소드당 1회라 성능 무관).
- reset_ic 후 golden 스케줄 전체오차 **1.1ft/6s**(row1 0.42fps 과도=정확한 IC미분 히스토리 없이 시드한 본질적 과도, golden 의 IC-eval 아티팩트힘 미재현; RL 무의미).

## P5 완료 — tolerance 회귀 CI + 성능(occupancy/FP32) (2026-08-30)
### tolerance 회귀 CI
`tests/regression_test.py` — golden(6초 4채널) 대조를 tolerance PASS/FAIL 로 고정, 실패시 exit(1).
네 경로 검증: A)GPU 궤적커널 fdm_run, B)배치 env FP64, C)host-C(있으면), D)배치 env FP32.
TOL=관측오차 ~3x 헤드룸(pos 0.1ft/att 0.01°/vel 0.05fps/alpha 0.005°), FP32 는 pos 60ft 별도.

### occupancy 최적화 (레지스터 캡)
프로파일(`tests/profile_kernel.py`)로 병목 실측: 커널이 **255 레지스터 하드캡**에 걸려
**occupancy 17%**(SM당 8/48 warp), 스필 352B. 순수 FP64 연산이라 레지스터 폭증.
→ 드라이버 JIT `CU_JIT_MAX_REGISTERS`(cuModuleLoadDataEx)로 레지스터 상한 → occupancy↑.
`--maxrregcount`(NVRTC)는 **무효**(PTX만 생성, 레지스터 할당은 드라이버 JIT). 스윕(`profile_regcap.py`):

| regcap | occ | ss1 | ss4 |
|---|---|---|---|
| 255(기본) | 17% | 28.6 | 39.4 |
| **96(FP64 채택)** | 42% | **35.5 (+24%)** | 44.2 (+12%) |

수치 불변(레지스터 배치만 변경, golden 대조 동일).

### FP32 옵션 (소비자 GPU FP64 1/64 rate 회피)
`cuda_rt.to_fp32()` — 검증된 FP64 소스를 기계변환(float 리터럴 f접미사→승격방지, double→float),
**FP64 원본은 불변**(단일 진실원본, assemble 시점만 변환). `GpuDogfight(precision='fp32')`.
- **정확도**: 자세/속도/받음각은 **FP64급**(0.003°/0.015fps/0.0006°). **위치만 43ft/6s** 드리프트
  — ECI 절대좌표(~20.9M ft)의 FP32 해상한계(값당 ~2.5ft)가 360스텝 적분 누적. RL 상대기하
  수백~수천 ft 스케일이면 무의미. (완전해결=항법FP64/동역학FP32 혼합정밀, 후속 옵션.)
- **처리량**(65536 aircraft, end-to-end GpuDogfight.step()):

| | FP64(regcap96) | FP32(regcap128) | 배속 |
|---|---|---|---|
| ss=1 | 37.0 M ac-steps/s | **123.6 M/s** | 3.3x |
| ss=4(action_repeat) | 42.7 M sim-steps/s | **312.7 M/s** | **7.3x** |

### 실행
```
D:\...\envs\aip_gpu\python.exe cuda_fdm/tests/regression_test.py    # tolerance CI (exit 0/1)
D:\...\envs\aip_gpu\python.exe cuda_fdm/tests/profile_kernel.py     # regs/occupancy/스윕
D:\...\envs\aip_gpu\python.exe cuda_fdm/tests/profile_regcap.py     # regcap 스윕
```

## P6 진행 — RL 통합 (벡터화 env 코어, FP64) (2026-08-30)
`cuda_fdm/rl_env.py` `GpuDogfightVecEnv` — GpuDogfight(fp64) 위에 대회 학습 규약.
- **상태 브릿지 ECI→대회 9-DOF** (`state9()` → (nenv,2,9): N/E/D[m], roll/pitch/yaw[deg], body u/v/w[m/s]).
  `states` 에서 자체계산(obs 비의존, reset 직후 유효). **커널 kinematics 를 torch 로 bit-동일 포팅**
  (euler 0, vUVW 1e-13). N/E/D 는 FighterSim 규약 재현: origin lat37.9146/lon128.1819/alt0,
  `pm.geodetic2ned(mGeodLat, mLon, alt_asl, origin)` — **DLL 이 alt_asl(=|eci|-a)을 측지고도로
  취급하는 관행** 그대로(euler 는 geocentric-NED, 위치는 geodetic-NED; DLL 내부 불일치까지 미러).
  pymap3d 대조 **0.056mm**. euler·위치 모두 golden/DLL 규약 정합.
- **대회 IC 분포**(env_utils.STANDARD_ENV_CONFIG): 시나리오 A(수직·반대 4)/B(head-on 1),
  고도 2000~30000ft·속도 200~300m/s·거리 A{2000,2500,3000}·B 10000ft, 두 기체 공유. NED→geodetic→
  `ic.build_seed_vector`. **IC 왕복**(NED→seed→ECI→bridge→NED) sub-mm/heading 0°/speed 1e-13.
- **★staggered 랜덤 리셋**(사용자 요청): 첫 iteration/학습재개 시 각 env 를 랜덤 목표위상
  p_e∈[0,max_steps) '유효' 스텝 진행 후 캡처 → 수많은 env 가 t≈0 에 몰리는 위상편향 제거.
  **종료-인지**: 워밍업 중 이탈(지면관통/발산) env 는 즉시 새 IC 재시드+age리셋 → 캡처상태는 항상
  유효 IC 로부터 p_e 스텝. 검증: stagger 후 고도범위 [291,8818]m(전부 유효), 위상 N std 877m(무stagger 688m).
- **★claude164r 관측 + my_reward 보상 GPU 벡터화** (`cuda_fdm/obs_reward.py` `BatchObsReward`):
  - **관측 184dim**(=50 스칼라+114 벡터+20 action-history; "claude164r"=action-hist 이전 164). 기하
    (distance/ATA/aspect/LOS az·el)·6프레임 방향벡터·damage_rate 3-tier·연료·pqr(SO3 log)·뱅크각·
    pursuit 를 **전부 torch 배치**로. 레이아웃/상수는 `my_observation` 에서 직접 import → 정합 강제.
  - **재구성 상태**(hp/연료/시간/자세history/pqr/action-history/shaping prev_x)를 **기체단위(nac) 배치**로
    유지(`advance()` = RL step 당 1회, sim.step 후 적분). partner=a^1, 거리·시간은 env 내 공유.
  - **보상**: HP차분 damage + 거리·조준·고도 포텐셜 shaping(telescoping) + 종료 alt항. self-play 로 두 기체 모두.
  - **CPU 참조 대조**(`tests/obs_reward_val.py`): 기하 vs GeometryInfo **1e-13**, 관측 vs `build_observation`
    (40스텝×16기) **9e-19**, 보상 vs `compute_reward` **6e-15**, terminal 합성 **0** → 사실상 비트일치.
- **step() 전체 종료+autoreset**: 고도<300m / HP<=0 / 비유한 → terminated, SIM_TIME>200s → truncated.
  done env 는 새 IC 재시드+재구성 초기화 후 새 관측 반환(`info['terminal_obs']` 에 종료관측). 반환
  `(obs (nenv,2,184), reward (nenv,2), done (nenv,), info)`. reward 는 reset 전 상태로 계산(정합).
- **stagger 에 재구성 통합**: 워밍업이 hp/시간/pqr/action-history 까지 함께 진행·캡처 → 위상 물리·재구성 일치.

### ★커널 융합 (성능 최적화) `cuda_fdm/gen/obs_kernel.cu`
- torch build_obs/advance/reward(작은 연산 수백개 → launch-bound, 45ms)를 **융합 NVRTC 커널 2개**로:
  - `advance_kernel`(1 thread/env): action push + hp/연료/pqr/시간 적분 + 종료 판정 + 보상. env 하나가
    두 기체 소유 → rate 커플링 race 없음. NED 브릿지+kinematics 도 커널 내 계산(states 직접).
  - `build_obs_kernel`(1 thread/기체): states+재구성 → obs 184 (순수·비파괴). autoreset 후 관측 재빌드에 재사용.
- **sync-free autoreset**: CPU IC 빌드/`.item()`/`.any()` 제거 — GPU IC 풀(ic_pool_size쌍, 1회 생성)에서
  `randint` gather 후 `torch.where` 로 done env 만 덮어쓰고 재구성은 `masked_fill_` 로 초기화(전부 in-place).
  step() 이 CPU-GPU 동기화를 일으키지 않는다(done 은 GPU 텐서 반환, 소비자가 필요시 동기화).
- **커널 정합**(`tests/obs_reward_val.py` (E)): 커널 vs torch(=CPU참조 비트일치) obs **7e-9**, reward **8e-11**,
  hp/pqr **정확일치**.
- **성능**(RTX 3070 Ti):

  | env | 물리단독 | +advance커널 | +build_obs커널 | full step(autoreset포함) | 이전 torch |
  |---|---|---|---|---|---|
  | 4096(8192기) | 3.16ms | 3.74 | 4.80 | **3.68ms** | ~50ms (**13.6×**) |
  | 16384(32768기) | — | — | — | **10.3ms**(19.1M sim-frame/s) | ~61ms (6×) |

  → obs+reward+종료+autoreset 이 **물리 비용에 근접**(사실상 공짜). 검증: `tests/rl_env_val.py`+`tests/obs_reward_val.py`(A~E) 전부 OK.

### ★GPU 네이티브 PPO + gated self-play `cuda_fdm/ppo_gpu.py` + `cuda_fdm/train_gpu.py`
- CPU 단일-env `claude_code/ppo.py` 와 **동일 학습 규약**을 GPU 벡터 env 위에서 **전 과정 GPU 텐서**로 재현
  (롤아웃·GAE·업데이트에 numpy 왕복이나 per-step `.item()` 동기화 없음; 로깅 sync 는 iteration 당 1회).
- **원본과 동일**: 관측=claude164r(my_observation, 커널 비트일치) · 보상=my_reward(MY_REWARD_CONFIG) ·
  **행동공간=discrete**(4채널×`num_bins`=21 균등격자 linspace(-1,1,21), 채널별 독립 Categorical=model.MLPDiscreteActorCritic; 원본 train.py --action-bins 기본=21) ·
  초기분포=STANDARD_ENV_CONFIG · action_repeat=6(substeps). 행동→env: throttle 만 [-1,1]→[0,1](0.5z+0.5),
  roll/pitch/rudder 그대로(=`action_provider.policy_action_to_command`; CUDA FCS thr_pos=2·c_thr).
- **opponent pool gated self-play**: env 기체0=**main actor**(학습), 기체1=**opponent**(pool 샘플 frozen).
  **main 기체 전이만** 버퍼(T,nenv)에 담겨 opponent 데이터 자동 배제.
  - **EMA 승률 게이팅**(원본): 각 엔트리 EMA(main 승률, α=0.1); **evictable 최소 EMA ≥ gate(0.6)** 면 현재 main 을
    새 evictable snapshot(EMA 0.5) 추가(cap 4 초과 시 oldest evictable FIFO). 고정주기 추가는 없음.
  - **softmax 가중 샘플링**(원본): p_i = f/m + (1-f)·softmax(-ema_i/τ) (f=0.5, τ=0.3) — 승률 낮은(어려운) 후보 우대.
  - **milestone**(`milestone_period`=500): 그때 main 을 permanent(never-evict) 추가 + capacity +1, 이어 **exploiter**
    를 그때 main 유일 상대(frozen)로 scratch 학습(승률 target 0.7/max_iters) 후 permanent 추가(exploiter 학습 중
    main 가중치/optimizer/obs_rms/env 위상 저장·복원, global_step 불변).
- **autoreset**: truncation 은 γ·V(terminal_obs_main) fold 부트스트랩, terminated 는 경계. 승패는 env 가 노출한
  per-기체 terminal hp/alt(autoreset 전 캡처)로 GPU 판정(main 관점)→EMA·exploiter 게이팅·로깅.
- **검증**(`tests/ppo_gpu_val.py`): (A) discrete 격자==linspace(-1,1,21)·throttle 리맵 (B) GAE vs 참조 **2.4e-7**
  (C) 40iter NaN無·EV 0.12→0.90·**mean_return -4.6→+1.8**·게이팅 성장 (D) 게이팅 FIFO·softmax 참조식 1e-8
  (E) exploiter/milestone permanent 2·main-only 배치 T·nenv.
- **처리량**(RTX 3070 Ti): nenv4096·rollout32 **~0.29M RL-step/s**(opponent forward P개 가감). entropy≈4·ln21(discrete 초기),
  EV·KL·clipfrac 건강. checkpoint=model+opt+norm+**pool(EMA 포함)**+iteration(resume 복원).
- **실행**: `python -m cuda_fdm.train_gpu --nenv 4096 --iters 2000 --milestone-period 500 --save runs/gpu.pt --log runs/gpu.csv`
  (resume `--resume`; self-play 끄기 `--milestone-period 0` + gate 매우 높게; exploiter 끄기 `--exploiter-iters 0`).

### 다음 (P6 잔여)
- reward/curriculum 스케줄(own_damage_weight 0.5→1.0, shaping 배율) — 현재 kernel_advance cfg 고정.
- pool 무한성장(permanent 누적) 상한.
- (P6 별도) 혼합정밀 FP32: 위치 43ft 드리프트 제거하며 FP32 속도 유지 → `memory/cuda-mixed-precision-p6`.
- 잔여: 기어/LEF/스피드브레이크 부수채널; 연료==0·two-circle guard 종료(현재 미적용).
- ✅완료: **P3** CUDA-C+GPU검증 · **P4** 배치env+IC · **P5** tolerance CI+occupancy+FP32 ·
  **P6** 벡터화 RL env(브릿지+stagger+obs/reward/종료/autoreset, CPU참조 비트일치, 커널융합 13.6× 가속) ·
  **GPU 네이티브 PPO**(opponent pool 자기대전, main-only 학습, exploiter, sync-free 롤아웃/GAE/업데이트).
