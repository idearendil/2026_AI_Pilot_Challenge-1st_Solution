# -*- coding: utf-8 -*-
# AIAICE JSBSimWrapper.py V.0.7.3((2022.04.18))
# JSBSim FDM and Rule based AI pilot model are separated

# AIAICE JSBSimWrapper.py V.0.7.2
# #theta = root.find("alpha")#2022.01.28
# #theta = root.find("theta")#2022.01.28
# gamma = root.find("gamma")#2022.02.08
# gamma.text = str(Init_Pitch)#2022.02.08

# AIAICE JSBSimWrapper.py V.0.7.1
# #theta = root.find("alpha")#2022.01.28
# theta = root.find("theta")#2022.01.28

# AIAICE JSBSimWrapper.py V.0.7
# 1. Add intial speed paramater in def __init__()  
# 2. Add PurePursuitRun()
# Next

# AIAICE JSBSimWrapper.py V.0.6
# 1. Add locking access xml files for jsbsim

# AIPILOT v1.1.2 Added 2023.02.03
# 1. class AIPilot_Adv added.

import ctypes as ct
import os
import math
import struct
from xml.etree.ElementTree import parse
from filelock import Timeout, FileLock

class J_NavigationData(ct.Structure):
    _pack_ = 1
    _fields_ = [
        ("Header", ct.c_uint8 * 2), ("Counter", ct.c_uint8), ("SimTime", ct.c_double), ("AircraftModel", ct.c_uint8),("AircraftID", ct.c_uint8),
        ("Lat", ct.c_int32),("Lon", ct.c_int32),("Alt", ct.c_uint32),("phi", ct.c_int32),("theta", ct.c_int32),("psi", ct.c_int32),
        ("u", ct.c_int32),("v", ct.c_int32),("w", ct.c_int32),("p", ct.c_int32),("q", ct.c_int32),("r", ct.c_int32),
        ("Ax", ct.c_int32),("Ay", ct.c_int32),("Az", ct.c_int32),
        ("AOA", ct.c_float),("AOS", ct.c_float),("KCAS", ct.c_uint16),("KTAS", ct.c_uint16),("GNDS", ct.c_uint16),("MachNum", ct.c_uint16),
        ("VV", ct.c_float), ("Nz", ct.c_int16),("Ny", ct.c_int16),
        ("LonMode", ct.c_uint8), ("LonCtrlCmd", ct.c_float),("ElevatorPosition", ct.c_int16),("FlapCtrlCmd", ct.c_int8),
        ("FlapPosition", ct.c_int16),("LatMode", ct.c_uint8),("LatCtrlCmd", ct.c_float),("AileronPosition", ct.c_int16),
        ("DirCtrlCmd", ct.c_float),("RudderPosition", ct.c_int16),("SpeedMode", ct.c_uint8),("SpeedCtrlCmd1", ct.c_float),
        ("Engine1_N1RPM", ct.c_uint32),("Engine1_N2RPM", ct.c_uint32),("Engine1_FuelFlow", ct.c_double),("SpeedCtrlCmd2", ct.c_float),
        ("Engine2_N1RPM", ct.c_uint32),("Engine2_N2RPM", ct.c_uint32),("Engine2_FuelFlow", ct.c_double),("SpeedBrakeCtrlCmd", ct.c_int8),
        ("SpeedBrakePosition", ct.c_uint16),("Fuel", ct.c_double),("checksum", ct.c_uint8)
    ]

class J_MetaData(ct.Structure):
    _pack_ = 1
    _fields_ = [
        ("NatoName", ct.c_char_p),  ("forceSide", ct.c_char_p), ("CallSign", ct.c_char_p)
    ]

#Add for rule based AI pilot
class ControlValue(ct.Structure):
    _pack = 1
    _fields_ = [
        ("RollCMD", ct.c_float), ("PitchCMD", ct.c_float), ("rudderCMD", ct.c_float), ("Throttle", ct.c_float)
    ]

#Add for AIPilot V1.1.2 
class oPlaneData(ct.Structure):
    _pack = 1
    _fields_ = [
        ("LocationX", ct.c_float),("LocationY", ct.c_float),("LocationZ", ct.c_float),
        ("Roll", ct.c_float), ("Pitch", ct.c_float), ("Yaw", ct.c_float),
        ("Speed", ct.c_float),("Team", ct.c_int),("Resv0", ct.c_float),("Resv1", ct.c_float),("Resv2", ct.c_float)
    ]
    
cur_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
#print("cur_path", cur_path)
path_to_so_file = os.path.join(cur_path, "JSBSimAIPLib.dll")
#print("path_to_so_file", path_to_so_file)
JSBSim = ct.cdll.LoadLibrary(path_to_so_file)

JSBSim.CreateBattleSpace.argtypes = None
JSBSim.CreateBattleSpace.resttype = ct.c_int
JSBSim.RemoveSpace.argtypes = [ct.c_int]
JSBSim.RemoveSpace.resttype = None
JSBSim.Reset.argtypes = [ct.c_int]
JSBSim.Reset.resttype = None
JSBSim.Init.argtypes = [ct.c_int, ct.c_int, ct.c_int, ct.c_double]
JSBSim.Init.resttype = ct.c_int
JSBSim.RunFighter.argtypes = [ct.c_int, ct.c_int, ct.c_int, ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_void_p]
JSBSim.RunFighter.restypes = ct.c_void_p
JSBSim.GetMetaData.argtypes = [ct.c_int, ct.c_int, ct.c_void_p]
JSBSim.GetMetaData.restypes = ct.c_void_p
JSBSim.AutoFighter.argtypes = [ct.c_int, ct.c_int, ct.c_int, ct.c_ubyte, ct.c_ubyte, ct.c_ubyte, ct.c_float, ct.c_float, ct.c_float, ct.c_float, ct.c_float, ct.c_float, ct.c_float, ct.c_void_p] #V0_5
JSBSim.AutoFighter.restypes = ct.c_void_p #V0_5

#JSBSIM step 결과 데이터 가져오기 (21.07.06 추가)
JSBSim.GetData.argtypes = [ct.c_int, ct.c_int, ct.c_void_p]
JSBSim.GetData.restypes = ct.c_void_p

class Fighter(object):
    def __init__(self, spaceID:int, fighterType:int, forceSide:int, delta:float, Init_Lat, Init_Lon, Init_Alt, Init_Yaw, Init_Pitch, Init_Roll, Init_Speed):
        # fighterType -> F16:0x01, F15:0x02, FA50:0x03
        # forceSide   -> 0x01:Red, 0x02 : Blue, 0x04:Unknown

        self._fighterType = fighterType
        self._delta = delta       
        self._forceSide = forceSide
        self._spaceID = spaceID

        if self._fighterType == 0x01 :
            self._fighterTypeName = "f16"

        if forceSide == 0x01 :
            self._forceSideName = "Red"
        elif forceSide == 0x02:
            self._forceSideName = "Blue"
        else:
            self._forceSideName = "UnKnown"
        
        self._initPath = cur_path + "/aircraft/{}/{}_init.xml".format(self._fighterTypeName, self._fighterTypeName) #V04
        
        #Lock file access for multi environment 
        #권한 문제 주의
        with FileLock(self._initPath + ".lock"): #V0_6
            doc = parse(self._initPath)
            root  = doc.getroot()
            latitude = root.find("latitude")
            longitude = root.find("longitude")
            altitude = root.find("altitude")
            psi = root.find("psi")
            phi = root.find("phi")
            #theta = root.find("alpha")#2022.01.28
            gamma = root.find("gamma")#2022.02.08
            vt = root.find("vt") #v.0.7
            latitude.text   = str(Init_Lat)
            longitude.text  = str(Init_Lon)
            altitude.text   = str(Init_Alt)
            psi.text = str(Init_Yaw)
            phi.text = str(Init_Roll)
            gamma.text = str(Init_Pitch)#2022.02.08
            vt.text = str(Init_Speed) #v.0.7
            doc.write(self._initPath)
            self.fighterID = JSBSim.Init(self._spaceID, self._fighterType, self._forceSide, self._delta)
            # print("[JSBSimWrapper] figherID = ", self.fighterID)

    def get_initPath(self):
        return self._initPath

    def get_spaceID(self):
        return self._spaceID

    def get_fighterID(self)->int:
        return self.fighterID

    def get_fighterTypeName(self):
        return self._fighterTypeName
        
    # Manual Control Run
    def Run(self, stick_X, stick_Y, throttle1, throttle2, rudder):
        if throttle1 < 0:
            throttle1 = 0
        if throttle2 < 0:
            throttle2 = 0

        o = J_NavigationData()
        JSBSim.RunFighter(self._spaceID, self.fighterID, self._fighterType, 
                          stick_X, stick_Y, 
                          throttle1, throttle2, rudder, 
                          ct.byref(o))
        OutState = o

        return o
    
    def AutoRun(self, latCtrlMode, lonCtrlMode, spdCtrlMode, theta_cmd, phi_cmd, gamma_cmd, psi_cmd, alt_cmd, beta_cmd, speed_cmd ):
        """Auto Pilot Control Run V0_5        
            Auto Pilot Operation Range( Alt 10,000ft ~ 28,000ft  / Speed  600~1,000ft/sec -> 183 ~ 305m/sec)
            
            -- Args -- 
            - latCtrlMode   :1:Manual Ctrl, 2:SCAS Ctrl, 3:Heading Ctrl
            - Longitudinal  :1:Manual Ctrl, 2:SCAS Ctrl, 3:FlightPathAngle Ctrl, 4:Alt Ctrl
            - Speed         :0:NoCtrl, 2:NoCtrl, 3:Speed Ctrl
            - theta_cmd     :Pitch cmd (deg) [-90 ~ 90]
            - phi_cmd       :Roll cmd (deg) [-90 ~ 90]
            - gamma_cmd     :FlightPathAngle cmd (deg) [-90 ~ 90]
            - psi_cmd       :Heading cmd (deg) [0 ~ 360]
            - alt_cmd       :altitude cmd (feet)
            - beta_cmd      :AOS cmd (deg)
            - speed_cmd     :Speed cmd(ft/s)
        """
        o = J_NavigationData()
        
        JSBSim.AutoFighter(self._spaceID, self.fighterID, self._fighterType, latCtrlMode, lonCtrlMode, spdCtrlMode,
                            theta_cmd, phi_cmd, gamma_cmd, psi_cmd, alt_cmd, beta_cmd, speed_cmd, ct.byref(o)) #OutState)
        OutState = o

        return o

    def get_spaceID(self)->int:
        return self._spaceID
    
    def get_fighterID(self)->int:
        return self.fighterID
    
    def get_forceSide(self)->int:
        return self._forceSide
        
    def get_fdm_data(self)->J_NavigationData:
        o = J_NavigationData()
        JSBSim.GetData(self._spaceID, self.fighterID, ct.byref(o))
        return o

def create_battleSpace()->int:
    return JSBSim.CreateBattleSpace()

def RemoveSpace(spaceID):
    JSBSim.RemoveSpace(spaceID)

def Reset(spaceID:int):
    JSBSim.Reset(spaceID)