# -*- coding: utf-8 -*-
"""표준대기 (FGStandardAtmosphere) double 정밀 복제.
바이어스/그래디언트 0 (기본), 습도무시. 지오포텐셜 고도 기반 층상 ISA.
"""
import math
from . import jsb_const as C
from .jsb_tables import Table1D


class StandardAtmosphere:
    def __init__(self):
        self.temp_tab = Table1D(C.STD_ATMOS_TEMP)          # geopot ft -> degR
        self.alts = [r[0] for r in C.STD_ATMOS_TEMP]
        self.temps = [r[1] for r in C.STD_ATMOS_TEMP]
        self.nrows = len(self.alts)
        # LapseRates (FGStandardAtmosphere.cpp CalculateLapseRates, bias/grad=0)
        self.lapse = []
        for b in range(self.nrows - 1):
            t0, t1 = self.temps[b], self.temps[b + 1]
            h0, h1 = self.alts[b], self.alts[b + 1]
            self.lapse.append((t1 - t0) / (h1 - h0))
        # PressureBreakpoints (CalculatePressureBreakpoints, SLpress=StdDaySLpressure)
        self.pbrk = [0.0] * self.nrows
        self.pbrk[0] = C.StdDaySLpressure
        for b in range(self.nrows - 1):
            BaseTemp = self.temps[b]
            deltaH = self.alts[b + 1] - self.alts[b]
            Tmb = BaseTemp   # bias/grad = 0
            Lmb = self.lapse[b]
            if Lmb != 0.0:
                Exp = C.g0 * C.Mair / (C.Rstar * Lmb)
                factor = Tmb / (Tmb + Lmb * deltaH)
                self.pbrk[b + 1] = self.pbrk[b] * factor ** Exp
            else:
                self.pbrk[b + 1] = self.pbrk[b] * math.exp(-C.g0 * C.Mair * deltaH / (C.Rstar * Tmb))

    def geopot(self, geometalt):
        ER = C.EarthRadius
        return (geometalt * ER) / (ER + geometalt)

    def geometric(self, geopotalt):
        ER = C.EarthRadius
        return (geopotalt * ER) / (ER - geopotalt)

    def temperature(self, altitude):
        """degR at geometric altitude (bias/grad=0)."""
        GeoPotAlt = self.geopot(altitude)
        if GeoPotAlt >= 0.0:
            return self.temp_tab.value(GeoPotAlt)
        else:
            return self.temp_tab.value(0.0) + GeoPotAlt * self.lapse[0]

    def pressure(self, altitude):
        GeoPotAlt = self.geopot(altitude)
        BaseAlt = self.alts[0]
        b = 0
        for bb in range(self.nrows - 2):
            testAlt = self.alts[bb + 1]
            if GeoPotAlt < testAlt:
                b = bb
                break
            BaseAlt = testAlt
            b = bb + 1
        Tmb = self.temperature(self.geometric(BaseAlt))   # == BaseTemp
        deltaH = GeoPotAlt - BaseAlt
        Lmb = self.lapse[b]
        if Lmb != 0.0:
            Exp = C.g0 * C.Mair / (C.Rstar * Lmb)
            factor = Tmb / (Tmb + Lmb * deltaH)
            return self.pbrk[b] * factor ** Exp
        else:
            return self.pbrk[b] * math.exp(-C.g0 * C.Mair * deltaH / (C.Rstar * Tmb))

    def __init_density_bp(self):
        # StdDensityBreakpoints[i] = pbrk[i]/(Reng*temp[i])  (FGStandardAtmosphere)
        self.dbrk = [self.pbrk[i] / (C.Reng * self.temps[i]) for i in range(self.nrows)]

    def density_altitude(self, density, geometric_alt):
        if not hasattr(self, "dbrk"):
            self.__init_density_bp()
        b = 0
        for bb in range(len(self.dbrk) - 2):
            if density >= self.dbrk[bb + 1]:
                b = bb
                break
            b = bb + 1
        Tmb = self.temps[b]
        Hb = self.alts[b]
        Lmb = self.lapse[b]
        pb = self.dbrk[b]
        if Lmb != 0.0:
            Exp = -1.0 / (1.0 + (C.g0 * C.Mair) / (C.Rstar * Lmb))
            da = Hb + (Tmb / Lmb) * ((density / pb) ** Exp - 1)
        else:
            Factor = -(C.Rstar * Tmb) / (C.g0 * C.Mair)
            da = Hb + Factor * math.log(density / pb)
        return self.geometric(da)

    def calculate(self, altitude):
        """returns dict(T,P,rho,a,mu,nu,sigma,densalt)."""
        T = self.temperature(altitude)
        P = self.pressure(altitude)
        rho = P / (C.Reng * T)
        a = math.sqrt(C.SHRatio * C.Reng * T)
        mu = C.Beta_visc * T ** 1.5 / (C.SutherlandConstant + T)
        nu = mu / rho
        # SL density (std): StdDaySLpressure/(Reng*StdDaySLtemperature)
        rho_sl = C.StdDaySLpressure / (C.Reng * C.StdDaySLtemperature)
        sigma = rho / rho_sl
        densalt = self.density_altitude(rho, altitude)
        return dict(T=T, P=P, rho=rho, a=a, mu=mu, nu=nu, sigma=sigma, densalt=densalt)
