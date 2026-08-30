# -*- coding: utf-8 -*-
"""JSBSim v1.0.0 물리상수 (원본: FGJSBBase.cpp, FGAtmosphere.cpp).
전부 double. CUDA 포팅시 그대로 __constant__ 로.
"""
# 단위 (FGJSBBase.cpp)
RADTODEG = 57.295779513082320876798154814105
DEGTORAD = 0.017453292519943295769236907684886
KTSTOFPS = 1.68781
FPSTOKTS = 1.0 / KTSTOFPS
INCHTOFT = 1.0 / 12
FTTOM = 0.3048
SLUGTOLB = 32.174049
LBTOSLUG = 1.0 / SLUGTOLB
KGTOSLUG = 0.06852168

# 대기 (FGAtmosphere.cpp)
KtoDegR = 1.8
Rstar = 8.31432 * (KGTOSLUG / (KtoDegR * FTTOM * FTTOM))   # ft*lbf/R/mol
Mair = 28.9645 * KGTOSLUG / 1000.0                          # slug/mol
g0 = 9.80665 / FTTOM                                        # ft/s^2
Reng = Rstar / Mair                                         # ft*lbf/slug/R
SHRatio = 1.40
StdDaySLtemperature = 518.67
StdDaySLpressure = 2116.228
SutherlandConstant = 198.72
Beta_visc = 2.269690E-08

# 표준대기 온도테이블 (FGStandardAtmosphere.cpp L99-107): geopot alt(ft) -> temp(degR)
STD_ATMOS_TEMP = [
    (0.0000,      518.67),
    (36089.2388,  389.97),
    (65616.7979,  389.97),
    (104986.8766, 411.57),
    (154199.4751, 487.17),
    (167322.8346, 487.17),
    (232939.6325, 386.37),
    (278385.8268, 336.5028),
    (298556.4304, 336.5028),
]
EarthRadius = 6356766.0 / FTTOM   # ft  (FGStandardAtmosphere.cpp L63)
