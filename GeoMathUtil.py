# AIACE GeoMathUtil.py V.0.71a
import numpy as np
import numpy.linalg as la

D2R = np.pi/180.0
R2D = 180.0/np.pi


class GeometryInfo():
    def __init__(self):
        pass
    
    def get_geometry_info_ned(self, ownship_ned:np.array, target_ned:np.array):
        dis = self._get_distance(ownship_ned, target_ned)
        aa  = self._get_aspect_angle(ownship_ned, target_ned, proj=False)
        hca = self._get_heading_cross_angle(ownship_ned, target_ned, proj=False)
        ata = self._get_antenna_train_angle(ownship_ned, target_ned, proj=False)

        return dis, aa, hca, ata 

    def _get_distance(self, ownship_ned, target_ned):
        pos_vec = target_ned[0:3] - ownship_ned[0:3]
        _distance = la.norm(pos_vec)

        return _distance

    def _get_down_angle(self, ownship_rpy):
        _angle = 0
        roll, pitch, heading = ownship_rpy[0]*D2R, ownship_rpy[1]*D2R, ownship_rpy[2]*D2R
        Tx = np.array([[1, 0, 0], [0, np.cos(roll), np.sin(roll)], [0, -np.sin(roll), np.cos(roll)]])
        Ty = np.array([[np.cos(pitch), 0, -np.sin(pitch)], [0, 1, 0], [np.sin(pitch), 0, np.cos(pitch)]])
        Tz = np.array([[np.cos(heading), np.sin(heading), 0], [-np.sin(heading), np.cos(heading), 0], [0, 0, 1]])

        T = np.matmul(np.matmul(Tx, np.matmul(Ty, Tz)), [0, 0, -1])
        _angle = np.arccos(np.clip(T[0], -1.0, 1.0))*R2D

        return _angle


    def _get_aspect_angle(self, ownship_ned, target_ned, proj=False):
        _angle = 0
        roll, pitch, heading = target_ned[3]*D2R, target_ned[4]*D2R, target_ned[5]*D2R
        Tx = np.array([[1, 0, 0], [0, np.cos(roll), np.sin(roll)], [0, -np.sin(roll), np.cos(roll)]])
        Ty = np.array([[np.cos(pitch), 0, -np.sin(pitch)], [0, 1, 0], [np.sin(pitch), 0, np.cos(pitch)]])
        Tz = np.array([[np.cos(heading), np.sin(heading), 0], [-np.sin(heading), np.cos(heading), 0], [0, 0, 1]])
        Tz_pi = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        
        p_ned = ownship_ned[0:3] - target_ned[0:3]
        p_norm_ned = la.norm(p_ned)
        if p_norm_ned != 0:
            p_unit_ned = p_ned/p_norm_ned
        
        else:
            p_unit_ned = p_ned
        
        # 2D ATA
        if (proj == True):
            T = np.matmul(Tz_pi, Tz)
            p_unit_t = np.matmul(T, p_unit_ned)
            _angle = np.arctan2(p_unit_t[1], p_unit_t[0])*R2D
        # 3D ATA
        else:
            T = np.matmul(Tz_pi, np.matmul(Tx, np.matmul(Ty, Tz)))
            p_unit_t = np.matmul(T, p_unit_ned)
            _angle = np.arccos(np.clip(p_unit_t[0],-1.0,1.0))*R2D
            # 3D에서는 부호가 정의가 안되서... 아래는 수정이 필요할 수 있다.
            sign = 1
            if p_unit_t[1] < -0.10:
                sign = -1
            elif -0.01 < p_unit_t[1] < 0.01:
                sign = np.sign(p_unit_t[2])
            _angle = sign*np.arccos(np.clip(p_unit_t[0],-1.0,1.0))*R2D

        return _angle

    def _get_heading_cross_angle(self, ownship_ned, target_ned, proj=False):
        _angle = 0
        # target
        roll, pitch, heading = target_ned[3]*D2R, target_ned[4]*D2R, target_ned[5]*D2R
        Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
        Rz = np.array([[np.cos(heading), -np.sin(heading), 0], [np.sin(heading), np.cos(heading), 0], [0, 0, 1]])

        roll, pitch, heading = ownship_ned[3]*D2R, ownship_ned[4]*D2R, ownship_ned[5]*D2R
        Tx = np.array([[1, 0, 0], [0, np.cos(roll), np.sin(roll)], [0, -np.sin(roll), np.cos(roll)]])
        Ty = np.array([[np.cos(pitch), 0, -np.sin(pitch)], [0, 1, 0], [np.sin(pitch), 0, np.cos(pitch)]])
        Tz = np.array([[np.cos(heading), np.sin(heading), 0], [-np.sin(heading), np.cos(heading), 0], [0, 0, 1]])
        
        if proj == True:
            V_t = np.matmul(Rz, np.array([1, 0, 0]))
            V_t_body = np.matmul(Tz, V_t)
            _angle = np.arctan2(V_t_body[1], V_t_body[0])*R2D
        else:
            V_t = np.matmul(Rz, np.matmul(Ry, np.matmul(Rx, np.array([1, 0, 0]))))
            V_t_body = np.matmul(Tx, np.matmul(Ty, np.matmul(Tz, V_t)))
            # 3D에서는 부호가 정의가 안되서... 아래는 수정이 필요할 수 있다.
            sign = 1
            if V_t_body[1] < -0.10:
                sign = -1
            elif -0.01 < V_t_body[1] < 0.01:
                sign = np.sign(V_t_body[2])
            _angle = sign*np.arccos(np.clip(V_t_body[0],-1.0,1.0))*R2D

        return _angle


    def _get_antenna_train_angle(self, ownship_ned, target_ned, proj = False):
        _angle = 0
        roll, pitch, heading = ownship_ned[3]*D2R, ownship_ned[4]*D2R, ownship_ned[5]*D2R
        Tx = np.array([[1, 0, 0], [0, np.cos(roll), np.sin(roll)], [0, -np.sin(roll), np.cos(roll)]])
        Ty = np.array([[np.cos(pitch), 0, -np.sin(pitch)], [0, 1, 0], [np.sin(pitch), 0, np.cos(pitch)]])
        Tz = np.array([[np.cos(heading), np.sin(heading), 0], [-np.sin(heading), np.cos(heading), 0], [0, 0, 1]])
        
        p_ned = target_ned[0:3] - ownship_ned[0:3] 
        p_norm_ned = la.norm(p_ned)
        if p_norm_ned != 0:
            p_unit_ned = p_ned/p_norm_ned
        
        else:
            p_unit_ned = p_ned
        
        # 2D ATA
        if (proj == True):
            p_unit_t = np.matmul(Tz, p_unit_ned)
            _angle = np.arctan2(p_unit_t[1], p_unit_t[0])*R2D
        # 3D ATA
        else:
            T = np.matmul(Tx, np.matmul(Ty, Tz))
            p_unit_t = np.matmul(T, p_unit_ned)
            _angle = np.arccos(np.clip(p_unit_t[0], -1.0, 1.0))*R2D

        return _angle


    def _get_los_angle(self, ownship_ned, target_ned):
        p_own = ownship_ned[:3]
        p_target = target_ned[:3]
        dis_ned = p_target - p_own
        
        if la.norm(dis_ned) != 0:
            dis_unit_ned = dis_ned / la.norm(dis_ned)
        else:
            dis_unit_ned = dis_ned
        
        phi, theta, psi = D2R*ownship_ned[3], D2R*ownship_ned[4], D2R*ownship_ned[5]
        tx = np.array([[1, 0, 0], [0, np.cos(phi), np.sin(phi)], [0, -np.sin(phi), np.cos(phi)]])
        ty = np.array([[np.cos(theta), 0, -np.sin(theta)], [0, 1, 0], [np.sin(theta), 0, np.cos(theta)]])
        tz = np.array([[np.cos(psi), np.sin(psi), 0], [-np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
        T_nb = np.matmul(tx, np.matmul(ty, tz))
        dis_body_norm = np.matmul(T_nb, dis_unit_ned)
        
        _az = np.arctan2(dis_body_norm[1], dis_body_norm[0])*R2D  # -180 ~ 180
        _el = -np.arcsin(np.clip(dis_body_norm[2], -1.0, 1.0))*R2D  # -90 ~ 90
        
        return _az, _el
