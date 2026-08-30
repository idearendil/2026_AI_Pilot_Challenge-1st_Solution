# -*- coding: utf-8 -*-
"""대회 DLL(JSBSimAIPLib) vs 순정 pip jsbsim 궤적 일치 검증.

같은 초기조건(IC)과 같은 입력 시퀀스를
  (a) 현재 대회 DLL      : FighterSim.JSBSim  (Windows 전용, 채점과 동일한 물리)
  (b) 순정 pip jsbsim    : jsbsim.FGFDMExec + 같은 aircraft/f16/f16.xml
에 각각 먹여 상태 궤적(N,E,D, roll,pitch,yaw, u,v,w)을 겹쳐 비교한다.

목적: "리눅스에서 pip jsbsim + 같은 XML 로 동일 FDM 을 재현할 수 있는가?" 판정.
  - 궤적이 tight 하게 일치  -> 리눅스 학습환경 이식 사실상 확정(전이비용 ~0).
  - 어긋남               -> 채널별 어디서 얼마나 벌어지는지 CSV + 요약으로 진단
                           (보통 원인은: FCS 입력 부호(SGN), trim 여부(TRIM),
                            혹은 DLL 빌드 JSBSim 버전 차이).

실행(Windows, aip 파이썬):
  D:/other_programs/anaconda3/envs/aip/python.exe claude_code/validate_jsbsim_port.py

DLL 은 Windows 전용이므로 이 검증은 Windows 에서 돈다(양쪽을 나란히 돌릴 수 있는 유일한
곳). 여기서 통과하면, 리눅스에는 (b) 경로만 그대로 배포하면 된다.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\AIP_LIB_claude\DogFightEnv\Release")
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

FEET_TO_METER = 0.30480
METER_TO_FEET = 3.28084

# ---------------------------------------------------------------------------
# 설정 (실험으로 조정하는 부분)
# ---------------------------------------------------------------------------
DT_HZ = 60                                   # config.py sim_hz
# config.py DEFAULT_ENV_CONFIG["ownship"] = [N, E, D, roll, pitch, yaw, speed]
IC = dict(N=1000.0, E=0.0, D=-7000.0, roll=0.0, pitch=0.0, yaw=0.0, speed=300.0)
ORIGIN = (37.91455691666666, 128.18188127777776, 0.0)   # FighterSim datum(lat,lon,alt)

# FighterSim.step 액션 순서: [roll(x_stick), pitch(y_stick), rudder, throttle(0~1)]
# 순정 jsbsim 쪽 FCS 입력 프로퍼티 매핑 + 부호(부호는 미지 -> 실험으로 확정)
FCS = dict(
    roll="fcs/aileron-cmd-norm",
    pitch="fcs/elevator-cmd-norm",
    rudder="fcs/rudder-cmd-norm",
    throttle="fcs/throttle-cmd-norm",
)
SGN = dict(roll=+1.0, pitch=+1.0, rudder=+1.0)   # 필요시 -1.0 로 뒤집어 재실행

# 항력 보정: DLL(=순정 공식 JSBSim v1.0.0. "JSBSim-ML"은 ML포크가 아니라 XML config
# markup v2.0) 항력이 jsbsim 기어-up 보다 큼. 이는 버전차가 아님(정확히 같은 v1.0.0으로
# 빌드해도 잔존) → gear상태/config 차이로 추정. 부분 기어전개(gear-pos)로 근사 상쇄한다.
# 0.0=기어up 정직물리(잔차 ~4m/s), 0.30=경험 보정(<1m/s). 순수 물리비교를 원하면 0.0.
GEAR_POS = 0.30

# jsbsim run_ic 후 simple trim 적용 여부. DLL 이 내부 trim 하는지 미지이므로
# 스크립트는 trim ON/OFF 두 버전을 모두 DLL 과 비교해 더 맞는 쪽을 보고한다.
CSV_OUT = ROOT / "artifacts" / "jsbsim_port_validation.csv"

# 입력 스케줄: (구간 끝 시각[s], [roll,pitch,rudder,throttle01]) — 채널별로 하나씩 자극
SCHEDULE = [
    (1.5, [0.0, 0.0, 0.0, 0.80]),   # A 순항: 무입력, throttle 0.8 -> FDM/엔진/적분 기본기
    (3.0, [0.0, 0.3, 0.0, 0.80]),   # B pitch pull
    (4.5, [0.3, 0.0, 0.0, 0.80]),   # C roll
    (6.0, [0.0, 0.0, 0.3, 0.80]),   # D rudder
]
TOTAL_T = SCHEDULE[-1][0]
N_STEPS = int(round(TOTAL_T * DT_HZ))


def action_at(t: float):
    for t_end, act in SCHEDULE:
        if t < t_end:
            return act
    return SCHEDULE[-1][1]


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------
def wrap180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def ang_err(a: float, b: float) -> float:
    return abs(wrap180(a - b))


STATE_LABELS = ["N", "E", "D", "roll", "pitch", "yaw", "u", "v", "w"]
ANG_IDX = {3, 4, 5}


def diff_vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """상태 9성분 절대오차 (각도 성분은 wrap 처리)."""
    out = np.empty(9)
    for i in range(9):
        out[i] = ang_err(a[i], b[i]) if i in ANG_IDX else abs(a[i] - b[i])
    return out


# ---------------------------------------------------------------------------
# (a) DLL 사이드
# ---------------------------------------------------------------------------
def make_dll():
    import JSBSimWrapper
    import FighterSim
    space = JSBSimWrapper.create_battleSpace()
    cfg = [1, 1, IC["N"], IC["E"], IC["D"], IC["roll"], IC["pitch"], IC["yaw"], IC["speed"]]
    sim = FighterSim.JSBSim(cfg, None, DT_HZ, space)
    return sim


def dll_state(sim) -> np.ndarray:
    s = sim.get_state()
    return np.array([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8]], dtype=float)


# ---------------------------------------------------------------------------
# (b) 순정 jsbsim 사이드
# ---------------------------------------------------------------------------
def make_jsbsim(init_lat, init_lon, init_alt_m, init_yaw, init_pitch, init_roll,
                init_speed, trim: bool):
    import jsbsim
    fdm = jsbsim.FGFDMExec(str(ROOT))     # ROOT/aircraft, ROOT/engine 사용
    fdm.set_debug_level(0)
    if not fdm.load_model("f16"):
        raise RuntimeError("jsbsim load_model('f16') 실패")
    fdm.set_dt(1.0 / DT_HZ)
    # IC: FighterSim.reset 이 DLL 에 넘긴 값과 동일하게. 래퍼는 pitch 를 gamma(경로각)로 넣는다.
    fdm["ic/lat-geod-deg"] = init_lat
    fdm["ic/long-gc-deg"] = init_lon
    fdm["ic/h-sl-ft"] = init_alt_m * METER_TO_FEET
    fdm["ic/psi-true-deg"] = init_yaw
    fdm["ic/phi-deg"] = init_roll
    fdm["ic/gamma-deg"] = init_pitch
    fdm["ic/beta-deg"] = 0.0
    fdm["ic/vt-fps"] = init_speed * METER_TO_FEET
    fdm["gear/gear-cmd-norm"] = 0.0        # airstart: 기어 올림(안 올리면 CDgear 항력이 붙어 감속)
    fdm.run_ic()
    # 엔진 airstart: 안 하면 추력 0(글라이더)으로 감속해 버린다. (-1 = 전 엔진 running)
    try:
        fdm.get_propulsion().init_running(-1)
    except Exception as exc:
        print(f"  [warn] engine init_running 실패: {exc}")
    fdm["gear/gear-cmd-norm"] = 0.0
    fdm.run_ic()
    fdm["gear/gear-pos-norm"] = GEAR_POS   # 기어 위치 즉시 세팅(GEAR_POS=포크 항력 보정)
    # FBW override: 주최측은 "ON"이라 했으나, 실측(probe_fbw/probe_pitch)상 DLL 은 pitch-scheduler/
    # roll-rate-command 경로(=override OFF)의 과도응답 모양까지 정확히 재현됨. override ON(직결)은
    # 피치를 크게 오버슈트(t1.0 -20도 vs DLL -6.6도). => 충실한 재현은 override OFF. 세팅하지 않음(기본 0).
    if trim:
        try:
            fdm["simulation/do_simple_trim"] = 1
        except Exception as exc:   # trim 실패해도 계속(결과에 드러남)
            print(f"  [warn] simple trim 실패: {exc}")
    return fdm


def jsbsim_step(fdm, act):
    roll, pitch, rudder, throttle = act
    fdm[FCS["roll"]] = SGN["roll"] * roll
    fdm[FCS["pitch"]] = SGN["pitch"] * pitch
    fdm[FCS["rudder"]] = SGN["rudder"] * rudder
    fdm[FCS["throttle"]] = throttle
    fdm["gear/gear-cmd-norm"] = 0.0       # 기어 계속 up 유지(kinematic 재전개 방지)
    fdm["gear/gear-pos-norm"] = GEAR_POS  # 보정 위치 고정
    fdm.run()


def jsbsim_state(fdm) -> np.ndarray:
    import pymap3d as pm
    lat = fdm["position/lat-geod-deg"]
    lon = fdm["position/long-gc-deg"]
    alt_m = fdm["position/h-sl-ft"] * FEET_TO_METER
    n, e, d = pm.geodetic2ned(lat, lon, alt_m, *ORIGIN)
    phi = np.degrees(fdm["attitude/phi-rad"])
    theta = np.degrees(fdm["attitude/theta-rad"])
    psi = wrap180(np.degrees(fdm["attitude/psi-rad"]))
    u = fdm["velocities/u-fps"] * FEET_TO_METER
    v = fdm["velocities/v-fps"] * FEET_TO_METER
    w = fdm["velocities/w-fps"] * FEET_TO_METER
    return np.array([n, e, d, phi, theta, psi, u, v, w], dtype=float)


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def run_pair(trim: bool):
    """DLL 과 (trim 지정) jsbsim 을 동일 입력으로 N_STEPS 돌려 궤적/오차 반환."""
    sim = make_dll()
    # DLL 초기화가 래퍼에 저장한 실제 IC(geodetic/heading/pitch/roll/speed)를 그대로 jsbsim 에 사용
    fdm = make_jsbsim(sim._init_pos_lat, sim._init_pos_lon, sim._init_pos_alt,
                      sim._init_heading, sim._init_pitch, sim._init_roll,
                      sim._init_speed, trim=trim)

    rows = []
    max_err = np.zeros(9)
    for k in range(N_STEPS):
        t = k / DT_HZ
        act = action_at(t)
        # DLL: FighterSim.step 액션순서 [roll,pitch,rudder,throttle01]
        sim.step(np.array(act, dtype=np.float32))
        jsbsim_step(fdm, act)

        a = dll_state(sim)     # 진실값(대회 DLL)
        b = jsbsim_state(fdm)  # 순정 jsbsim
        d = diff_vec(a, b)
        max_err = np.maximum(max_err, d)
        if k % 30 == 0 or k == N_STEPS - 1:
            rows.append((round(t, 3), a.copy(), b.copy(), d.copy()))

    # 정리
    import JSBSimWrapper
    try:
        JSBSimWrapper.RemoveSpace(sim._space_id)
    except Exception:
        pass
    return rows, max_err


def print_report(label, rows, max_err):
    print(f"\n================= jsbsim ({label}) vs DLL =================")
    hdr = "  t[s] |" + "".join(f"{n:>8}" for n in STATE_LABELS)
    for tag, rows_key in (("DLL   ", 1), ("jsbsim", 2), ("|err| ", 3)):
        pass
    # 마지막 스냅샷만 상세 표로
    t, a, b, d = rows[-1]
    print(f"[최종 t={t}s]")
    print(hdr)
    print("  DLL   |" + "".join(f"{x:8.2f}" for x in a))
    print("  jsb   |" + "".join(f"{x:8.2f}" for x in b))
    print("  |err| |" + "".join(f"{x:8.3f}" for x in d))
    print("  ---- 전체 구간 채널별 max|err| ----")
    print("        |" + "".join(f"{x:8.3f}" for x in max_err))
    # 종합 판정 지표: 위치(m), 각도(deg), 속도(m/s)
    pos = max_err[[0, 1, 2]].max()
    ang = max_err[[3, 4, 5]].max()
    vel = max_err[[6, 7, 8]].max()
    print(f"  요약: 위치 max {pos:.2f} m | 자세 max {ang:.2f} deg | 속도 max {vel:.2f} m/s")
    return pos, ang, vel


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # Windows cp949 콘솔 한글 깨짐 방지
    except Exception:
        pass
    print(f"검증: DT={DT_HZ}Hz, {N_STEPS} steps ({TOTAL_T}s), IC={IC}")
    print(f"입력 스케줄(구간끝s, [roll,pitch,rud,thr01]): {SCHEDULE}")
    print(f"FCS 매핑={FCS}\nSGN(부호)={SGN}")

    results = {}
    for trim in (False, True):
        label = "trim ON" if trim else "trim OFF"
        try:
            rows, max_err = run_pair(trim)
        except Exception as exc:
            import traceback
            print(f"\n[{label}] 실패: {exc}")
            traceback.print_exc()
            continue
        results[label] = (rows, max_err, print_report(label, rows, max_err))

    # CSV: 더 잘 맞는(위치오차 작은) 쪽을 저장
    if results:
        best = min(results.items(), key=lambda kv: kv[1][2][0])
        label, (rows, max_err, _) = best
        CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
        with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t"] + [f"dll_{n}" for n in STATE_LABELS]
                       + [f"jsb_{n}" for n in STATE_LABELS]
                       + [f"err_{n}" for n in STATE_LABELS])
            for t, a, b, d in rows:
                w.writerow([t] + list(a) + list(b) + list(d))
        print(f"\nCSV 저장({label}): {CSV_OUT}")

        print("\n=================== 판정 가이드 ===================")
        print(" - 위치<~수 m, 자세<~1deg, 속도<~1m/s 수준으로 6s 유지 -> 사실상 동일 FDM.")
        print("   => 리눅스 pip jsbsim 이식 확정. (b) 경로만 리눅스에 배포하면 됨.")
        print(" - 특정 채널만 부호 반대로 크게 벌어짐 -> SGN[해당채널] 뒤집어 재실행.")
        print(" - 순항(A구간, 0~1.5s)부터 서서히 벌어짐 -> trim/IC 불일치(둘 중 맞는 라벨 참고).")
        print(" - 전 채널 무작위로 크게 어긋남 -> DLL 빌드 JSBSim 버전 차이 가능성, 버전 핀 검토.")


if __name__ == "__main__":
    main()
