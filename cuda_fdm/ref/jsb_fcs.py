# -*- coding: utf-8 -*-
"""F-16 Flight Control System (flight_control name="F-16 FC") double 정밀 복제.
원본: aircraft/f16/f16.xml L319-994 + jsbsim FGKinemat/FGPID/FGGain/FGSummer/FGSwitch.

실행순서(채널): Flaps, Roll, Pitch, Yaw, LandingGear, LEF, Throttle, Speedbrake, Hook, Canopy.
채널간 property 참조는 "이번 프레임 먼저 쓴 값이 있으면 그것, 없으면 전프레임 값".
alpha/mach/n-pilot/p,q,r-aero/vc/vg = 전프레임 Auxiliary/Accelerations 값(FCS가 그들보다 먼저 실행).

상태: kinematic 출력들, PID 3개(Input_prev, Input_prev2, I_out_total).
DT=1/60.
"""
from .jsb_tables import Table1D

# FGFCSComponent.dt 는 생성자(load_model 시점)에서 GetChannelDeltaT()=GetDeltaT()*rate 로
# 한 번만 캡처된다. 대회/현재 환경은 load_model 후 set_dt(1/60) 순서라, 로드시점 기본
# dt=1/120 이 FCS 전 컴포넌트(kinematic/PID)에 고정된다(=적분 dt 1/60의 절반). 이 gotcha를
# 그대로 재현해야 golden 과 bit-정합. (simulation/channel-dt property 는 1/60 을 보이지만
# 그것은 라이브 게터일 뿐, 컴포넌트 내부 dt 는 1/120.)
DT = 1.0 / 120.0
DEG2RAD = 0.017453292519943295769236907684886


def _clip(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def _equal_roundoff(a, b):
    # jsbsim EqualToRoundoff (float eps 기반). double 계산이므로 넉넉히.
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


class Kinemat:
    """FGKinemat. detents/times, DoScale. output_state = 이전 출력값."""
    def __init__(self, detents, times, doscale=True):
        self.detents = list(map(float, detents))
        self.times = list(map(float, times))
        self.doscale = doscale
        self.output = 0.0

    def run(self, inp, out_seed=None):
        """out_seed: 출력 property가 외부/앞단에서 이미 쓰였으면 그 값에서 시작
        (러더=PID출력, gear=외부강제). None이면 자신의 이전 출력에서 시작."""
        dt0 = DT
        Input = inp
        if self.doscale:
            Input *= self.detents[-1]
        Output = self.output if out_seed is None else out_seed
        Input = _clip(Input, self.detents[0], self.detents[-1])
        n = len(self.detents)
        while dt0 > 0.0 and not _equal_roundoff(Input, Output):
            ind = 1
            while True:
                cond = (self.detents[ind] < Output) if (Input < Output) else (self.detents[ind] <= Output)
                if not cond:
                    break
                ind += 1
                if ind >= n:
                    break
            if ind >= n:
                ind = n - 1
            if self.times[ind] <= 0.0:
                Output = Input
                break
            Rate = (self.detents[ind] - self.detents[ind - 1]) / self.times[ind]
            ThisInput = _clip(Input, self.detents[ind - 1], self.detents[ind])
            ThisDt = abs((ThisInput - Output) / Rate)
            if dt0 < ThisDt:
                ThisDt = dt0
                if Output < Input:
                    Output += ThisDt * Rate
                else:
                    Output -= ThisDt * Rate
            else:
                Output = ThisInput
            dt0 -= ThisDt
        self.output = Output
        return Output


class PID:
    """FGPID non-standard, ki type미지정=AB2. trigger!=0 이면 적분 스킵."""
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.input_prev = 0.0
        self.input_prev2 = 0.0
        self.i_out_total = 0.0

    def run(self, Input, trigger, clip=None):
        Dval = (Input - self.input_prev) / DT
        test = trigger
        I_out_delta = 0.0
        if abs(test) < 0.000001:
            I_out_delta = 1.5 * Input - 0.5 * self.input_prev   # AB2
        if test < 0.0:
            self.i_out_total = 0.0
        self.i_out_total += self.ki * DT * I_out_delta
        Output = self.kp * Input + self.i_out_total + self.kd * Dval
        self.input_prev2 = 0.0 if test < 0.0 else self.input_prev
        self.input_prev = Input
        if clip is not None:
            Output = _clip(Output, clip[0], clip[1])
        return Output


def aerosurface_zc(Input, InMin, InMax, OutMin, OutMax):
    """AEROSURFACE_SCALE, ZeroCentered=true (f16 기본), Gain=1."""
    if Input == 0.0:
        return 0.0
    elif Input > 0.0:
        return (Input / InMax) * OutMax
    else:
        return (Input / InMin) * OutMin


class F16FCS:
    def __init__(self):
        # kinematics
        self.k_tef = Kinemat([-1.0, 0.0, 1.0], [3.0, 0.0, 3.0])       # tef-control
        self.k_aileron = Kinemat([-1.0, 1.0], [0.3, 0.3])            # aileron-position -> left-aileron-pos-norm
        self.k_elevator = Kinemat([-1.0, 1.0], [0.3, 0.3])          # elevator-position-normalized
        self.k_rudder = Kinemat([-1.0, 1.0], [0.4, 0.4])            # rudder-position
        self.k_gear = Kinemat([0.0, 1.0], [0.0, 5.0])               # gear-control
        self.k_lef = Kinemat([-1.0, 1.0], [3.0, 3.0])              # lef-control
        # PIDs
        self.pid_roll = PID(3.00000, 0.00050, -0.00125)
        self.pid_gload = PID(0.3000, 0.0250, 0.0000)
        self.pid_yaw = PID(0.105500, 0.000010, 0.00005)
        # scheduled gain tables
        self.t_ail_speed = Table1D([(0.0, 1.0), (1.0, 0.15)])       # aileron-speed-compensated
        self.t_elev_sched = Table1D([(-0.5236, 0.0), (-0.5, 0.11), (0.0, 1.0),
                                     (0.5, 0.11), (0.5236, 0.0)])    # elevator-scheduler
        self.t_yaw_rate = Table1D([(80.0, 0.0), (100.0, 15.0), (150.0, 100.0)])  # yaw-rate-norm
        # persistent property values (전프레임 참조용, 이번프레임 갱신)
        self.left_aileron_pos_norm = 0.0
        self.elevator_pos_norm = 0.0
        self.rudder_pos_norm = 0.0
        self.tef_control = 0.0
        self.lef_control = 0.0
        self.gear_pos_norm = 0.0

    def step(self, cmd, aux, gear_pos_force):
        """cmd: dict aileron,elevator,rudder,throttle,pitch_trim,yaw_trim,gear (norm)
        aux: 전프레임 dict alpha_rad,mach,vc_kts,vg_fps,n_pilot_y,n_pilot_z,
             p_aero,q_aero,r_aero,pitch_rad,roll_rad
        gear_pos_force: 외부에서 매프레임 강제하는 gear/gear-pos-norm (우리 테스트 0.30)
        반환: dict 조종면 (aero 입력)."""
        alpha = aux["alpha_rad"]
        mach = aux["mach"]
        vc = aux["vc_kts"]
        vg = aux["vg_fps"]

        # ===== Flaps channel =====
        # tef-pos-rad switch (default 0; vc<250 ->0.349; mach>0.9 -> -0.0349) AND logic
        if vc < 250:
            tef_pos_rad = 0.349
        elif mach > 0.9:
            tef_pos_rad = -0.0349
        else:
            tef_pos_rad = 0.0
        tef_pos_norm = 2.864789 * tef_pos_rad
        tef_control = self.k_tef.run(tef_pos_norm)
        self.tef_control = tef_control

        # ===== Roll channel =====
        roll_rate_norm = 0.31821 * aux["p_aero"]
        roll_trim_error = cmd["aileron"] - roll_rate_norm
        ail_pid_trigger = 0.0 if vc < 20.0 else 1.0
        roll_rate_pid = self.pid_roll.run(roll_trim_error, ail_pid_trigger)
        roll_rate_command = _clip(roll_rate_pid + cmd["aileron"], -1.0, 1.0)
        aileron_pos_rad = aerosurface_zc(roll_rate_command, -1.0, 1.0, -0.375, 0.375)
        # aileron-position kinematic (fbw-override off -> input=roll_rate_command)
        left_aileron_pos_norm = self.k_aileron.run(roll_rate_command)
        self.left_aileron_pos_norm = left_aileron_pos_norm
        aileron_speed_comp = self.t_ail_speed.value(mach) * left_aileron_pos_norm
        left_flaperon_norm = _clip(-tef_control - aileron_speed_comp, -1.0, 1.0)
        right_flaperon_norm = _clip(tef_control - aileron_speed_comp, -1.0, 1.0)
        flaperon_summer = left_flaperon_norm + right_flaperon_norm
        flaperon_mix_rad = 1.4324 * flaperon_summer
        left_aileron_pos_rad = aerosurface_zc(aileron_speed_comp, -1.0, 1.0, -0.375, 0.375)
        right_aileron_pos_rad = aerosurface_zc(-aileron_speed_comp, -1.0, 1.0, -0.375, 0.375)

        # ===== Pitch channel =====
        import math
        n_pilot_z_corr = math.cos(aux["pitch_rad"]) * math.cos(aux["roll_rad"])
        g_load_corrected = aux["n_pilot_z"] - n_pilot_z_corr
        elevator_cmd_limiter = _clip(cmd["elevator"] + cmd["pitch_trim"], -1.0, 0.44)
        elevator_scheduler = self.t_elev_sched.value(alpha) * elevator_cmd_limiter
        alpha_limiter_norm = 1.0472 * alpha
        pitch_rate_norm = 6.2 * aux["q_aero"]
        g_load_norm = 0.020 * g_load_corrected
        pitch_trim_error = elevator_scheduler + pitch_rate_norm - g_load_norm
        elev_pid_trigger = 0.0 if vc < 5.0 else 1.0
        g_load_pid = self.pid_gload.run(pitch_trim_error, elev_pid_trigger, clip=(-1.0, 1.0))
        pitch_scheduler = _clip(elevator_scheduler + alpha_limiter_norm + g_load_pid, -1.0, 1.0)
        # fbw-override off -> switch = pitch_scheduler
        elevator_pos_norm = self.k_elevator.run(pitch_scheduler)
        self.elevator_pos_norm = elevator_pos_norm
        elevator_pos_rad = aerosurface_zc(elevator_pos_norm, -1.0, 1.0, -0.436, 0.436)
        dht_left_pos_rad = _clip(-elevator_pos_rad - left_aileron_pos_rad, -0.436, 0.436)
        dht_right_pos_rad = _clip(elevator_pos_rad + right_aileron_pos_rad, -0.436, 0.436)

        # ===== Yaw channel =====
        yaw_rate_norm = self.t_yaw_rate.value(vg) * aux["r_aero"]
        yaw_load_norm = 0.25 * aux["n_pilot_y"]
        yaw_trim_error = cmd["rudder"] + yaw_rate_norm + yaw_load_norm
        rud_pid_trigger = 0.0 if vc < 10.0 else 1.0
        # yaw-load-pid writes rudder-pos-norm
        yaw_load_pid = self.pid_yaw.run(yaw_trim_error, rud_pid_trigger, clip=(-1.0, 1.0))
        rudder_pos_norm_pid = yaw_load_pid   # PID output written to fcs/rudder-pos-norm
        yaw_scheduler = _clip(cmd["rudder"] + cmd["yaw_trim"] + yaw_load_pid, -1.0, 1.0)
        # rudder-position kinematic: output property = fcs/rudder-pos-norm, seeded by PID output
        rudder_pos_norm = self.k_rudder.run(yaw_scheduler, out_seed=rudder_pos_norm_pid)
        self.rudder_pos_norm = rudder_pos_norm
        rudder_pos_rad = aerosurface_zc(rudder_pos_norm, -1.0, 1.0, -0.524, 0.524)

        # ===== Landing Gear channel =====
        # gear-wow: needs WOW (airborne=0). gear-pos externally forced.
        gear_wow = 0.0
        # gear-control kinematic: input gear-cmd(0), output=gear/gear-pos-norm seeded by force
        gear_pos_norm = self.k_gear.run(cmd["gear"], out_seed=gear_pos_force)
        self.gear_pos_norm = gear_pos_norm

        # ===== LEF channel =====
        # switch priority: (wow&pos>0)->-0.0349; (pos==0 & a>0.2618)->0.436;
        #                  (wow==0 & a>0.0873)->0.262; (mach>0.9)->-0.0349; else 0
        if gear_wow == 1 and gear_pos_norm > 0:
            lef_pos_rad = -0.0349
        elif gear_pos_norm == 0 and alpha > 0.2618:
            lef_pos_rad = 0.436
        elif gear_wow == 0 and alpha > 0.0873:
            lef_pos_rad = 0.262
        elif mach > 0.9:
            lef_pos_rad = -0.0349
        else:
            lef_pos_rad = 0.0
        lef_pos_norm = 2.293578 * lef_pos_rad
        lef_control = self.k_lef.run(lef_pos_norm)
        self.lef_control = lef_control

        # ===== Speedbrake ===== (cmd 0, alpha<53 -> 0). speedbrake-pos-rad 미tie=0
        speedbrake_pos_rad = 0.0

        return dict(
            aileron_pos_rad=aileron_pos_rad,
            elevator_pos_rad=elevator_pos_rad,
            rudder_pos_rad=rudder_pos_rad,
            lef_pos_rad=lef_pos_rad,
            flaperon_mix_rad=flaperon_mix_rad,
            speedbrake_pos_rad=speedbrake_pos_rad,
            gear_pos_norm=gear_pos_norm,
            left_aileron_pos_norm=left_aileron_pos_norm,
            elevator_pos_norm=elevator_pos_norm,
            rudder_pos_norm=rudder_pos_norm,
            # 진단
            roll_rate_command=roll_rate_command,
            pitch_scheduler=pitch_scheduler,
            yaw_scheduler=yaw_scheduler,
            elevator_scheduler=elevator_scheduler,
            g_load_pid=g_load_pid,
            roll_rate_pid=roll_rate_pid,
            yaw_load_pid=yaw_load_pid,
            aileron_speed_compensated=aileron_speed_comp,
            tef_control=tef_control,
            left_aileron_pos_rad=left_aileron_pos_rad,
            right_aileron_pos_rad=right_aileron_pos_rad,
            dht_left_pos_rad=dht_left_pos_rad,
            dht_right_pos_rad=dht_right_pos_rad,
        )
