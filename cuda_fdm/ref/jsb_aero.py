# -*- coding: utf-8 -*-
"""F-16 공력 (FGAerodynamics + f16.xml aerodynamics) double 복제.
- 함수/테이블은 f16_aero_data.py(자동추출)에서.
- 힘: DRAG/SIDE/LIFT = wind축(atWind) → body. 모멘트: ROLL/PITCH/YAW = body(atBodyXYZ).
- 모멘트 CG 전이: vMoments = MRC + RPBody × vForces.
"""
import math
from .jsb_tables import Table1D, Table2D
from . import f16_aero_data as D

SW = 300.0
BW = 30.0
CBAR = 11.32

# 상수 property
CONST = {"metrics/Sw-sqft": SW, "metrics/bw-ft": BW, "metrics/cbarw-ft": CBAR}

FORCE_AXES = ["DRAG", "SIDE", "LIFT"]
MOMENT_AXES = ["ROLL", "PITCH", "YAW"]


def _build_table(t):
    if t is None:
        return None
    if t["type"] == "1d":
        return ("1d", t["indep"], Table1D(t["rows"]))
    else:
        return ("2d", t["row_indep"], t["col_indep"],
                Table2D(t["rowvals"], t["colvals"], t["data"]))


class F16Aero:
    def __init__(self):
        self.funcs = {}   # axis -> list of (name, factors, value, tableobj)
        for ax, fs in D.AXES.items():
            lst = []
            for f in fs:
                lst.append((f["name"], f["factors"], f["value"], _build_table(f["table"])))
            self.funcs[ax] = lst
        kt = D.TOP_FUNCTIONS["aero/function/kCLge"]["table"]
        self.kclge_tab = Table1D(kt["rows"])
        self.kclge_indep = kt["indep"]

    def _eval_func(self, factors, value, tab, ctx):
        v = 1.0
        for p in factors:
            v *= ctx[p]
        if value is not None:
            v *= value
        if tab is not None:
            if tab[0] == "1d":
                v *= tab[2].value(ctx[tab[1]])
            else:
                v *= tab[3].value(ctx[tab[1]], ctx[tab[2]])
        return v

    def compute(self, st, RPBody):
        """st: dict property->value (qbar,alpha,beta,Vt,pqr-aero,surfaces,gear,h_b_mac...).
        RPBody: (x,y,z) ft. 반환 forces(body 3), moments(body 3)."""
        Vt = st["Vt"]
        twovel = 2.0 * Vt
        bi2vel = BW / twovel if twovel != 0 else 0.0
        ci2vel = CBAR / twovel if twovel != 0 else 0.0
        kclge = self.kclge_tab.value(st["aero/h_b-mac-ft"])
        ctx = dict(CONST)
        ctx.update({
            "aero/qbar-psf": st["aero/qbar-psf"],
            "aero/alpha-rad": st["aero/alpha-rad"],
            "aero/beta-rad": st["aero/beta-rad"],
            "aero/bi2vel": bi2vel, "aero/ci2vel": ci2vel,
            "velocities/mach": st["velocities/mach"],
            "aero/function/kCLge": kclge,
            "velocities/p-aero-rad_sec": st["velocities/p-aero-rad_sec"],
            "velocities/q-aero-rad_sec": st["velocities/q-aero-rad_sec"],
            "velocities/r-aero-rad_sec": st["velocities/r-aero-rad_sec"],
            "fcs/aileron-pos-rad": st["fcs/aileron-pos-rad"],
            "fcs/elevator-pos-rad": st["fcs/elevator-pos-rad"],
            "fcs/rudder-pos-rad": st["fcs/rudder-pos-rad"],
            "fcs/lef-pos-rad": st["fcs/lef-pos-rad"],
            "fcs/flaperon-mix-rad": st["fcs/flaperon-mix-rad"],
            "fcs/speedbrake-pos-rad": st["fcs/speedbrake-pos-rad"],
            "gear/gear-pos-norm": st["gear/gear-pos-norm"],
        })
        axis_sum = {}
        for ax in FORCE_AXES + MOMENT_AXES:
            s = 0.0
            for (name, factors, value, tab) in self.funcs[ax]:
                s += self._eval_func(factors, value, tab, ctx)
            axis_sum[ax] = s
        # 힘: atWind. vFnative=(DRAG,SIDE,LIFT); negate drag,lift; Tw2b*.
        drag = -axis_sum["DRAG"]
        side = axis_sum["SIDE"]
        lift = -axis_sum["LIFT"]
        a = st["aero/alpha-rad"]
        b = st["aero/beta-rad"]
        ca, sa, cb, sb = math.cos(a), math.sin(a), math.cos(b), math.sin(b)
        # Tw2b (wind->body)
        Fx = ca * cb * drag + (-ca * sb) * side + (-sa) * lift
        Fy = sb * drag + cb * side + 0.0 * lift
        Fz = sa * cb * drag + (-sa * sb) * side + ca * lift
        forces = (Fx, Fy, Fz)
        # 모멘트: atBodyXYZ. MRC + RPBody x forces
        mrc = (axis_sum["ROLL"], axis_sum["PITCH"], axis_sum["YAW"])
        rx, ry, rz = RPBody
        cross = (ry * Fz - rz * Fy, rz * Fx - rx * Fz, rx * Fy - ry * Fx)
        moments = (mrc[0] + cross[0], mrc[1] + cross[1], mrc[2] + cross[2])
        return forces, moments, dict(bi2vel=bi2vel, ci2vel=ci2vel, kclge=kclge)
