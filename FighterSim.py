from typing import List, Dict
import numpy as np

import copy
import math
import ctypes as ct
from JSBSimWrapper import Reset
from JSBSimWrapper import Fighter
from JSBSimWrapper import J_NavigationData

import sys
import os
import threading

import pymap3d as pm

FEET_TO_METER = 0.30480
METER_TO_FEET = 3.28084
KNOT_TO_METER_SEC = 0.51444

def print_holdon(x, y):
    sys.stdout.write("\033[2F")
    sys.stdout.write("\033[K]")
    print("ownship:", x)
    sys.stdout.write("\033[K]")
    print("target:", y)
    
class Sim(object):
    def __init__(self, sim_cfg:list, simHz:int): 
        # "fighter1":[1,1,1,37,127,6000,90,30,30]  
        self._model = None # flight dynamics model        
        self._fighter_type  = sim_cfg[0] #FighterType -> F16:0x01
        self._force_side    = sim_cfg[1] #forceSide   -> 0x01:Red, 0x02:Blue, 0x03:Netural, 0x04:Unknown

        self._init_pos_n    = sim_cfg[2] # #North initial position (deg) #add V0_3
        self._init_pos_e    = sim_cfg[3] # #East initial position (deg) #add V0_3
        self._init_pos_d    = sim_cfg[4] # #Down (meter) #add V0_3

        
        if abs(sim_cfg[5]) == 90.0 or abs(sim_cfg[5]) == 270.0: #V0_4_2 if set initial roll = 90 -> jsbsim fdm not working
            sim_cfg[5] += 0.001

        if abs(sim_cfg[6]) == 90.0 or abs(sim_cfg[6]) == 270.0: #V0_4_2 if set initial pitch = 90 -> jsbsim fdm not working
            sim_cfg[6] += 0.001
        
        self._init_roll     = sim_cfg[5] # initial roll(deg)
        self._init_pitch    = sim_cfg[6] # initial pitch(deg)
        self._init_heading  = sim_cfg[7] # initial yaw(deg)
        self._init_speed    = sim_cfg[8] # initial speed(m/s)
        
        self._simHz         = simHz # flight dynamics model operation cycle

        #Set datum LLA, for NED conversion
        self._origin_lat = 37.91455691666666   #datum lat 36  -> 37  #22.05.06
        self._origin_lon = 128.18188127777776  #datum lon 127 -> 126 #22.05.06
        self._origin_alt = 0    #datum alt
        
        #Set fdm output
        self._out_fdm = J_NavigationData()

        #Set FDM update state
        self.fdm_update_success = True

        self._update_pause = False
        
    def step(self):
        pass

    def reset(self):
        pass        

class JSBSim(Sim):
    def __init__(self, sim_cfg:dict, AIP, simHz:int, space_id:int): 
        super().__init__(sim_cfg, simHz)       
        
        #Set jsbsim state list index
        self._jsbsim_state_index_num = 51
        #Set state np array
        self._state = np.zeros(self._jsbsim_state_index_num)
        self._target_state = np.zeros(self._jsbsim_state_index_num)
        #Set action np array
        self.action = np.zeros(4)
        #Set battle sapce Id
        self._space_id = space_id
        #Set initial Health (0~1)
        self._init_health = 1
        self._AIP = AIP
        self.VP = np.zeros(3)
                                
        #Convert intial positin NED (datum lla is lat:37.0 lon:127, alt:0 meter) -> LLA
        lla = pm.ned2geodetic(self._init_pos_n, self._init_pos_e, self._init_pos_d, self._origin_lat, self._origin_lon, self._origin_alt ) 
        self._init_pos_lat = lla[0] 
        self._init_pos_lon = lla[1] 
        self._init_pos_alt = lla[2]  #meter
    
        print("init ID:{} Lat{} Lon{} Alt(meter){} Speed(m/sec){}".format(self._space_id, self._init_pos_lat, self._init_pos_lon, self._init_pos_alt, self._init_speed))
        self.reset()         
        
    #Set step by action
    def step(self, action):
          
        #Manual controll        
        self.action = action                              
        x_stick = action[0] #roll command range: -1~1 -> -1:left 1:right
        y_stick = action[1] #pitch command range: -1~1 -> -1:aft  1:fwd
        rud     = action[2] #directional command range: -1~1 -> -1:left 1:right
        th      = action[3] #throttle range:  0~1 ->  0:0%   1:100%        
                
        #update JSBsim model
        self._out_fdm = self._model.Run(x_stick, y_stick, th, th, rud)
        self._update_state(self._out_fdm)

    def step_fix(self):
       
        #Manual controll        
        x_stick = 0 #roll command range: -1~1 -> -1:left 1:right
        y_stick = 0 #pitch command range: -1~1 -> -1:aft  1:fwd
        rud     = 0 #directional command range: -1~1 -> -1:left 1:right
        th      = 0 #throttle range:  0~1 ->  0:0%   1:100%

        #update JSBsim model
        self._out_fdm = self._model.Run(x_stick, y_stick, th, th, rud)
        self._update_state(self._out_fdm)  

    #Set step by loiter parameter    
    def step_loiter(self, isLoitering=True, bank=60, pitch=-15): # 
           
        #update JSBsim model
        bank_angle = bank
        pitch_angle = pitch
        if isLoitering:
            self._out_fdm = self._model.AutoRun(0x02, 0x04, 0x03, 
                                                0, bank_angle, 0, 0, 
                                                -self._init_pos_d * METER_TO_FEET, 0, self._init_speed * METER_TO_FEET)
        else:
            self._out_fdm = self._model.AutoRun(0x02, 0x02, 0x03, 
                                                pitch_angle , bank_angle, 0, 0, 
                                                -self._init_pos_d * METER_TO_FEET, 0, self._init_speed * METER_TO_FEET)
        self._update_state(self._out_fdm)

    def step_autopilot(self, heading_cmd:float, altitude_cmd:float, speed_cmd:float): # 
        """_summary_

        Args:
            heading_cmd (float): degree
            altitude_cmd (float): meter (NED, Down direction +)
            speed_cmd (float): meter/sec
        """
        
        #update JSBsim model        
        self._out_fdm = self._model.AutoRun(0x03, 0x04, 0x03, 
                                            0, 0, 0, heading_cmd, 
                                            -altitude_cmd * METER_TO_FEET, 0, speed_cmd * METER_TO_FEET) #
        self._update_state(self._out_fdm)
        
    def _update_state(self, out_fdm):
        #convert LLA -> NED
        #If JSBSim fdm retuns over range, print message.
        lat = out_fdm.Lat/1000000.0
        lon = out_fdm.Lon/1000000.0
        alt = out_fdm.Alt/1000
        if abs( lat ) > 90.0 or abs( lon) > 180.0 or alt < 0.0 :
            print("fdm ouput over range!!!! LAT{}  LON{}  ALT{}".format(lat, lon, alt) )            
            self.fdm_update_success = False
            return
      
        ned = pm.geodetic2ned(lat, lon, alt * FEET_TO_METER, self._origin_lat, self._origin_lon, self._origin_alt)#V0.4
        
        ########################### Position ############################
        self._state[0] = ned[0] #N :My Aircraft Position NED North /meter
        self._state[1] = ned[1] #E :My Aircraft Position NED East  /meter
        self._state[2] = ned[2] #D :My Aircraft Position NED Down  /meter
        ########################### Attitude ############################
        self._state[3] = out_fdm.phi/1000   #Roll  :Euler Angle - Phi(Nose direction +, right-hand rule)         /deg(-180 ~ 180)
        self._state[4] = out_fdm.theta/1000 #Pitch :Euler Angle - Theta(Right-wing direction +, right-hand rule) /deg(-180 ~ 180) 
        self._state[5] = out_fdm.psi/1000   #Yaw   :Euler Angel - Psi(Down direction +, right-hand rule)         /deg(-180 ~ 180)        
        ########################### Velocity ############################
        self._state[6] = out_fdm.u/1000 * FEET_TO_METER  #u :body x axis Velocity (Nose direction +, right-hand rule)       / meter/sec
        self._state[7] = out_fdm.v/1000 * FEET_TO_METER  #v :body y axis Velocity (Right-wing direction +, right-hand rule) / meter/sec
        self._state[8] = out_fdm.w/1000 * FEET_TO_METER  #w :body z axis Velocity (Down direction +, right-hand rule)       / meter/sec        
        ########################### Angular Rate ############################
        self._state[9]  = out_fdm.p/1000   #p :Aircraft Angular Rate - Roll (Nose direction +, right-hand rule)           / deg/sec
        self._state[10] = out_fdm.q/1000   #q :Aircraft Angular Rate - Pitch (Right-wing direction +, right-hand rule)    / deg/sec
        self._state[11] = out_fdm.r/1000   #r :Aircraft Angular Rate - Yaw (Down direction +, right-hand rule)/ deg/sec   / deg/sec        
        ########################### Flight Info added ############################
        self._state[12] = out_fdm.KCAS/10 * KNOT_TO_METER_SEC   #Calibated Air Speed / meter/sec(0~1028 m/s)  (cf. 1Knots->0.51444 m/s)          
        self._state[13] = out_fdm.AOA #Angle of Attack( -90 ~ 90 ) /deg
        self._state[14] = out_fdm.AOS #Angle of Sideslip( -90 ~ 90 ) /deg
        ########################### Aileron Control Info ##################################
        self._state[15] = out_fdm.LatCtrlCmd #Roll Stick Command (Generate Positive X-axis Moment +, RightTurn +)(-1 ~ 1)
        self._state[16] = out_fdm.AileronPosition /500 #Aileron Deflection Angle(-60~60) /deg
        ########################### Elevato Control Info ##################################
        self._state[17] = out_fdm.LonCtrlCmd #Pitch Stick Command (Generate Positive Y-axis Moment +, PitchUp +) (-1 ~ 1)
        self._state[18] = out_fdm.ElevatorPosition /500 #Elevator Deflection Angle(-60~60) /deg
        ########################### Rudder Control Info. ##################################
        self._state[19] = out_fdm.DirCtrlCmd #Rudder Pedal Command (Generate Positive Z-axis Moment +, RightTurn +)(-1 ~ 1) 
        self._state[20] = out_fdm.RudderPosition /500 #Rudder Deflection Angle -60~60 /deg
        ########################### Engine1 ##################################
        self._state[21] = out_fdm.SpeedCtrlCmd1 #Throttle Slider 1 Command( -1 ~ 1 )
        self._state[22] = out_fdm.Engine1_N1RPM #N1 RPM of Engine 1( 0 ~ 100 ) /RPM        
        self._state[23] = out_fdm.Fuel #  /Fuel /LBS 
        ########################### Accleration ##################################
        self._state[24] = out_fdm.Ax/1000 * FEET_TO_METER #X body accel, meter/sec 
        self._state[25] = out_fdm.Ay/1000 * FEET_TO_METER #Y body accel, meter/sec 
        self._state[26] = out_fdm.Az/1000 * FEET_TO_METER #Z body accel, meter/sec         
        ########################### Flight Info added ############################
        self._state[27] = out_fdm.KTAS/10 * KNOT_TO_METER_SEC   #True AirSpeed( 0 ~ 1028) / meter/sec  (cf. 1Knots->0.51444 m/s) 
        self._state[28] = out_fdm.GNDS/10 * KNOT_TO_METER_SEC   #Ground Speed( 0 ~ 1028)  / meter/sec  (cf. 1Knots->0.51444 m/s) 
        self._state[29] = out_fdm.MachNum /1000   #Mach Number of Aircraft (0 ~ 4) 
        self._state[30] = - out_fdm.VV * FEET_TO_METER  #Vertical Velocity(NED Down direction) meter/min 
        self._state[31] = out_fdm.Nz/1000 #Normal Acceleration (-20 ~ 20 ) / G 
        self._state[32] = out_fdm.Ny/1000 #Lateral Acceleration (-20 ~ 20 ) / G 
        ########################### Engine1 added ############################
        self._state[33] = out_fdm.Engine1_N2RPM #N2 RPM of Engine 1( 0 ~ 100 ) /RPM
        self._state[34] = out_fdm.Engine1_FuelFlow/1000 #Fuel Flow of Engine 1(0~ ) / m^3/sec 
        ########################### Engine2 ##################################
        self._state[35] = out_fdm.SpeedCtrlCmd2 #Throttle Slider 2 Command( -1 ~ 1 )
        self._state[36] = out_fdm.Engine2_N1RPM #N1 RPM of Engine 2( 0 ~ 100 ) /RPM
        self._state[37] = out_fdm.Engine2_N2RPM #N2 RPM of Engine 2( 0 ~ 100 ) /RPM
        self._state[38] = out_fdm.Engine2_FuelFlow/1000 #Fuel Flow of Engine 2(0~ ) / m^3/sec 
        self._state[39] = out_fdm.SpeedBrakeCtrlCmd #SpeedBrake Retraction/Neutral/Extension Control Command( -1 ~ 1 ) cf.flight computer controls a speedbrake automatically
        self._state[40] = out_fdm.SpeedBrakePosition/1000 #SpeedBrake Deflection Angle( 0 ~ 60 ) /deg 
        ########################### Etc ##################################
        self._state[41] = out_fdm.SimTime #Simulation Time(sec)
        self._state[42] = out_fdm.Lat/1000000.0 # Lat(deg)
        self._state[43] = out_fdm.Lon/1000000.0 # Lon(deg)
        self._state[44] = out_fdm.Alt/1000.0 * FEET_TO_METER #Alt(meter)
        self._state[45] = self._health #health
        self._state[46] = 0 #reserved
        self._state[47] = 0 #reserved
        self._state[48] = 0 #reserved
        self._state[49] = 0 #reserved
        self._state[50] = 0 #reserved    
           
        #If FDM updating is success, return True
        return True
    
    def get_state(self):
        """ return state """
        self._state
        
        state = copy.deepcopy(self._state)
        
        return state

    def reset(self, logging:bool = False, logname:str = None):
        
        """ make a new JSBSim model instance """
        
        #Create JSBSim FDM(Flight Dynamics Model)
        #_fighter_type:F16:0x01
        #_force_side: :Red:0x01, Blue:0x02,  Unknown:0x04
        self._model = Fighter(self._space_id, self._fighter_type, self._force_side, 1/self._simHz, 
                                self._init_pos_lat, self._init_pos_lon, self._init_pos_alt* METER_TO_FEET, 
                                self._init_heading, self._init_pitch, self._init_roll, self._init_speed* METER_TO_FEET)

        if self._AIP is not None:
            self._AIP.CreateBehaviorTree(self._model.fighterID, self._model._forceSide)
        """ Initialize """
        self._state =np.zeros(self._jsbsim_state_index_num)

        self._state[0] = self._init_pos_n # north /meter
        self._state[1] = self._init_pos_e # east  /meter
        self._state[2] = self._init_pos_d # down  /meter
        
        self._state[3] = self._init_roll     #roll  /deg phi
        self._state[4] = self._init_pitch    #pitch /deg theta 
        self._state[5] = self._init_heading  #yaw   /deg psi
        
        self._state[45] = self._init_health  #health 0~1
        
        self._health = self._init_health
       
        reset_state = copy.deepcopy(self._state)

        return reset_state

    #Get fdm (jsbsim flight dynamics model)    
    def get_fdm(self):        
        return copy.deepcopy(self._out_fdm)
    
    #return JSBSim fdm model        
    def get_model(self):
        return self._model
        
    def deduct_health(self, damage):
        self._health = self._health - damage
        
    #for network model
    def set_fdm_data(self, fdm_data_array):

        self._out_fdm.Lat               = fdm_data_array[42]
        self._out_fdm.Lon               = fdm_data_array[43]
        self._out_fdm.Alt               = fdm_data_array[44]
      
        self._out_fdm.phi               = fdm_data_array[3] 
        self._out_fdm.theta             = fdm_data_array[4] 
        self._out_fdm.psi               = fdm_data_array[5] 
      
        self._out_fdm.u                 = fdm_data_array[6]
        self._out_fdm.v                 = fdm_data_array[7]
        self._out_fdm.w                 = fdm_data_array[8]
      
        self._out_fdm.p                 = fdm_data_array[9]
        self._out_fdm.q                 = fdm_data_array[10]
        self._out_fdm.r                 = fdm_data_array[11]
      
        self._out_fdm.KCAS              = fdm_data_array[12]
        self._out_fdm.AOA               = fdm_data_array[13]
        self._out_fdm.AOS               = fdm_data_array[14]
      
        self._out_fdm.LatCtrlCmd        = fdm_data_array[15]
        self._out_fdm.AileronPosition   = fdm_data_array[16]
      
        self._out_fdm.LonCtrlCmd        = fdm_data_array[17]
        self._out_fdm.ElevatorPosition  = fdm_data_array[18]
      
        self._out_fdm.DirCtrlCmd        = fdm_data_array[19]
        self._out_fdm.RudderPosition    = fdm_data_array[20]
      
        self._out_fdm.SpeedCtrlCmd1     = fdm_data_array[21]
        self._out_fdm.Engine1_N1RPM     = fdm_data_array[22]
        self._out_fdm.Fuel              = fdm_data_array[23]
      
        self._out_fdm.Ax                = fdm_data_array[24]
        self._out_fdm.Ay                = fdm_data_array[25]
        self._out_fdm.Az                = fdm_data_array[26]
      
        self._out_fdm.KTAS              = fdm_data_array[27]
        self._out_fdm.GNDS              = fdm_data_array[28]
        self._out_fdm.MachNum           = fdm_data_array[29]
        self._out_fdm.VV                = fdm_data_array[30]
        self._out_fdm.Nz                = fdm_data_array[31]
        self._out_fdm.Ny                = fdm_data_array[32]
      
        self._out_fdm.Engine1_N2RPM     = fdm_data_array[33]
        self._out_fdm.Engine1_FuelFlow  = fdm_data_array[34]
      
        self._out_fdm.SpeedCtrlCmd2     = fdm_data_array[35]
        self._out_fdm.Engine2_N1RPM     = fdm_data_array[36]
        self._out_fdm.Engine2_N2RPM     = fdm_data_array[37]
        self._out_fdm.Engine2_FuelFlow  = fdm_data_array[38]
        self._out_fdm.SpeedBrakeCtrlCmd = fdm_data_array[39]
        self._out_fdm.SpeedBrakePosition= fdm_data_array[40]
      
        self._out_fdm.SimTime           = fdm_data_array[41]
        
        
    def step_behavior(self, target_sim_model:Fighter):        
        #sim_time_s = self.get_propery('sim_time_s')
        #print("target_fdm : {} {} {} {}".format(target_fdm.Lat, target_fdm.Lon, target_fdm.Alt, target_fdm.Fuel))                
        #self._out_fdm = self._model.BehaviorRun(ct.byref(target_fdm), sim_time_s)
        
        #control actions returned by RAIP
        if self._AIP is None:
            print("Rule Based AIP is none")
        else:
            
            _control_action = self._AIP.Step(
                self._model.fighterID, 
                self._model._forceSide,
                target_sim_model.fighterID, 
                target_sim_model._forceSide,
                # ct.pointer( self._model.get_fdm_data() ), 
                # ct.pointer( target_sim_model.get_fdm_data() ) 
                self._model.get_fdm_data(), 
                target_sim_model.get_fdm_data() 
                )
            _vp = self._AIP.GetVP(
                self._model.fighterID, 
                self._model._forceSide,
                self._model.get_fdm_data()
            )
            
            self.VP[0] = _vp.X
            self.VP[1] = _vp.Y
            self.VP[2] = _vp.Z
            
            #set action
            self.action[0] = _control_action.RollCMD
            self.action[1] = _control_action.PitchCMD
            self.action[2] = _control_action.RudderCMD
            self.action[3] = _control_action.Throttle           
            
            #update JSBSim model
            self._out_fdm = self._model.Run(_control_action.RollCMD,
                                            _control_action.PitchCMD,
                                            _control_action.Throttle,
                                            0,
                                            _control_action.RudderCMD)
            self._update_state(self._out_fdm)
            

    #get rule based model's actions
    def get_action(self, target_sim_model:Fighter):
        return self.action