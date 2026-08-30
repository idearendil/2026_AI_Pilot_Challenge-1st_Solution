# -*- coding: utf-8 -*-
"""F100-PW-229 터빈 + direct thruster (FGTurbine/FGThruster) double 복제.
throttle-pos-norm = 2*throttle-cmd (FCS throttle1 gain). >1 이면 afterburner(AugMethod=2).
연료소모: FuelFlow_pph → 내부탱크 등분 감소.
"""
import math
from .jsb_tables import Table2D
from .jsb_const import INCHTOFT

# 상수 (F100-PW-229.xml + FGTurbine 기본)
MILTHRUST = 17800.0
MAXTHRUST = 29000.0
BYPASSRATIO = 0.4
TSFC = 0.74
ATSFC = 2.05
IDLEN1, IDLEN2 = 40.0, 53.0
MAXN1, MAXN2 = 100.0, 100.0
AUGMENTED = 1
AUGMETHOD = 2
BLEEDDEMAND = 0.0
IdleFF = MILTHRUST ** 0.2 * 107.0

_COLS = [-10000, 0, 10000, 20000, 30000, 40000, 50000, 60000]
_IDLE_MACH = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
_IDLE = [
    [0.0430, 0.0488, 0.0528, 0.0694, 0.0899, 0.1183, 0.1467, 0.0],
    [0.0500, 0.0501, 0.0335, 0.0544, 0.0797, 0.1049, 0.1342, 0.0],
    [0.0040, 0.0047, 0.0020, 0.0272, 0.0595, 0.0891, 0.1203, 0.0],
    [-0.0804, -0.0804, -0.0560, -0.0237, 0.0276, 0.0718, 0.1073, 0.0],
    [-0.2129, -0.2129, -0.1498, -0.1025, 0.0474, 0.0868, 0.0900, 0.0],
    [-0.2839, -0.2839, -0.1104, -0.0469, -0.0270, 0.0552, 0.0800, 0.0],
]
_MIL_MACH = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
_MIL = [
    [1.2600, 1.0000, 0.7400, 0.5340, 0.3720, 0.2410, 0.1490, 0.0],
    [1.1710, 0.9340, 0.6970, 0.5060, 0.3550, 0.2310, 0.1430, 0.0],
    [1.1500, 0.9210, 0.6920, 0.5060, 0.3570, 0.2330, 0.1450, 0.0],
    [1.1810, 0.9510, 0.7210, 0.5320, 0.3780, 0.2480, 0.1540, 0.0],
    [1.2580, 1.0200, 0.7820, 0.5820, 0.4170, 0.2750, 0.1700, 0.0],
    [1.3690, 1.1200, 0.8710, 0.6510, 0.4750, 0.3150, 0.1950, 0.0],
    [1.4850, 1.2300, 0.9750, 0.7440, 0.5450, 0.3640, 0.2250, 0.0],
    [1.5941, 1.3400, 1.0860, 0.8450, 0.6280, 0.4240, 0.2630, 0.0],
]
_AUG_MACH = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6]
_AUG = [
    [1.1816, 1.0000, 0.8184, 0.6627, 0.5280, 0.3756, 0.2327, 0.0],
    [1.1308, 0.9599, 0.7890, 0.6406, 0.5116, 0.3645, 0.2258, 0.0],
    [1.1150, 0.9474, 0.7798, 0.6340, 0.5070, 0.3615, 0.2240, 0.0],
    [1.1284, 0.9589, 0.7894, 0.6420, 0.5134, 0.3661, 0.2268, 0.0],
    [1.1707, 0.9942, 0.8177, 0.6647, 0.5309, 0.3784, 0.2345, 0.0],
    [1.2411, 1.0529, 0.8648, 0.7017, 0.5596, 0.3983, 0.2467, 0.0],
    [1.3287, 1.1254, 0.9221, 0.7462, 0.5936, 0.4219, 0.2614, 0.0],
    [1.4365, 1.2149, 0.9933, 0.8021, 0.6360, 0.4509, 0.2794, 0.0],
    [1.5711, 1.3260, 1.0809, 0.8700, 0.6874, 0.4860, 0.3011, 0.0],
    [1.7301, 1.4579, 1.1857, 0.9512, 0.7495, 0.5289, 0.3277, 0.0],
    [1.8314, 1.5700, 1.3086, 1.0474, 0.8216, 0.5786, 0.3585, 0.0],
    [1.9700, 1.6900, 1.4100, 1.2400, 0.9100, 0.6359, 0.3940, 0.0],
    [2.0700, 1.8000, 1.5300, 1.3400, 1.0000, 0.7200, 0.4600, 0.0],
    [2.2000, 1.9200, 1.6400, 1.4400, 1.1000, 0.8000, 0.5200, 0.0],
]


def _seek(v, target, accel, decel, dt):
    if v > target:
        v -= dt * decel
        if v < target:
            v = target
    elif v < target:
        v += dt * accel
        if v > target:
            v = target
    return v


class F16Turbine:
    def __init__(self, n1=100.0, n2=100.0):
        self.N1 = n1
        self.N2 = n2
        self.N2norm = (n2 - IDLEN2) / (MAXN2 - IDLEN2)
        self.FuelFlow_pph = IdleFF
        self.idle_tab = Table2D(_IDLE_MACH, _COLS, _IDLE)
        self.mil_tab = Table2D(_MIL_MACH, _COLS, _MIL)
        self.aug_tab = Table2D(_AUG_MACH, _COLS, _AUG)

    def _spool_val(self, delay, densratio):
        n = min(1.0, self.N2norm + 0.1)
        return delay / (1 + 3 * (1 - n) ** 3 + (1 - densratio))

    def step(self, throttle_pos_norm, mach, densalt, T, densratio, dt, cg):
        ThrottlePos = throttle_pos_norm
        if ThrottlePos > 1.0:
            AugmentCmd = ThrottlePos - 1.0
            ThrottlePos -= AugmentCmd
        else:
            AugmentCmd = 0.0
        N1_factor = MAXN1 - IDLEN1
        N2_factor = MAXN2 - IDLEN2
        d = BYPASSRATIO
        n2up = self._spool_val(1.0 * 90.0 / (d + 3.0), densratio)
        n2dn = self._spool_val(3.0 * 90.0 / (d + 3.0), densratio)
        n1up = self._spool_val(1.0 * 90.0 / (d + 3.0), densratio)
        n1dn = self._spool_val(2.4 * 90.0 / (d + 3.0), densratio)
        self.N2 = _seek(self.N2, IDLEN2 + ThrottlePos * N2_factor, n2up, n2dn, dt)
        self.N1 = _seek(self.N1, IDLEN1 + ThrottlePos * N1_factor, n1up, n1dn, dt)
        self.N2norm = (self.N2 - IDLEN2) / N2_factor
        idlethrust = MILTHRUST * self.idle_tab.value(mach, densalt)
        milthrust = (MILTHRUST - idlethrust) * self.mil_tab.value(mach, densalt)
        thrust = idlethrust + milthrust * self.N2norm * self.N2norm
        # (not augmentation branch) fuel/nozzle/bleed
        correctedTSFC = TSFC * math.sqrt(T / 389.7) * (0.84 + (1 - self.N2norm) ** 2)
        self.FuelFlow_pph = _seek(self.FuelFlow_pph, thrust * correctedTSFC, 1000.0, 10000.0, dt)
        if self.FuelFlow_pph < IdleFF:
            self.FuelFlow_pph = IdleFF
        thrust = thrust * (1.0 - BLEEDDEMAND)
        # AugMethod 2
        if AUGMETHOD == 2 and AugmentCmd > 0.0:
            tdiff = (MAXTHRUST * self.aug_tab.value(mach, densalt)) - thrust
            thrust += tdiff * AugmentCmd
            self.FuelFlow_pph = _seek(self.FuelFlow_pph, thrust * ATSFC, 5000.0, 10000.0, dt)
        # 힘/모멘트 (direct thruster at structural (0,0,0))
        rz = INCHTOFT * (cg[2] - 0.0)
        ry = INCHTOFT * (0.0 - cg[1])
        forces = (thrust, 0.0, 0.0)
        moments = (0.0, rz * thrust, -ry * thrust)
        fuel_burn_lbs = self.FuelFlow_pph / 3600.0 * dt
        return dict(thrust=thrust, N1=self.N1, N2=self.N2,
                    fuelflow_pph=self.FuelFlow_pph, fuel_burn=fuel_burn_lbs,
                    forces=forces, moments=moments)
