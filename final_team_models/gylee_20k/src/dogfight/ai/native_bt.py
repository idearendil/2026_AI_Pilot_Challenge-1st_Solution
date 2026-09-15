from __future__ import annotations

import ctypes as ct
import os
import struct
import sys


class J_NavigationData(ct.Structure):
    _pack_ = 1
    _fields_ = [
        ("Header", ct.c_uint8 * 2),
        ("Counter", ct.c_uint8),
        ("SimTime", ct.c_double),
        ("AircraftModel", ct.c_uint8),
        ("AircraftID", ct.c_uint8),
        ("Lat", ct.c_int32),
        ("Lon", ct.c_int32),
        ("Alt", ct.c_uint32),
        ("phi", ct.c_int32),
        ("theta", ct.c_int32),
        ("psi", ct.c_int32),
        ("u", ct.c_int32),
        ("v", ct.c_int32),
        ("w", ct.c_int32),
        ("p", ct.c_int32),
        ("q", ct.c_int32),
        ("r", ct.c_int32),
        ("Ax", ct.c_int32),
        ("Ay", ct.c_int32),
        ("Az", ct.c_int32),
        ("AOA", ct.c_float),
        ("AOS", ct.c_float),
        ("KCAS", ct.c_uint16),
        ("KTAS", ct.c_uint16),
        ("GNDS", ct.c_uint16),
        ("MachNum", ct.c_uint16),
        ("VV", ct.c_float),
        ("Nz", ct.c_int16),
        ("Ny", ct.c_int16),
        ("LonMode", ct.c_uint8),
        ("LonCtrlCmd", ct.c_float),
        ("ElevatorPosition", ct.c_int16),
        ("FlapCtrlCmd", ct.c_int8),
        ("FlapPosition", ct.c_int16),
        ("LatMode", ct.c_uint8),
        ("LatCtrlCmd", ct.c_float),
        ("AileronPosition", ct.c_int16),
        ("DirCtrlCmd", ct.c_float),
        ("RudderPosition", ct.c_int16),
        ("SpeedMode", ct.c_uint8),
        ("SpeedCtrlCmd1", ct.c_float),
        ("Engine1_N1RPM", ct.c_uint32),
        ("Engine1_N2RPM", ct.c_uint32),
        ("Engine1_FuelFlow", ct.c_double),
        ("SpeedCtrlCmd2", ct.c_float),
        ("Engine2_N1RPM", ct.c_uint32),
        ("Engine2_N2RPM", ct.c_uint32),
        ("Engine2_FuelFlow", ct.c_double),
        ("SpeedBrakeCtrlCmd", ct.c_int8),
        ("SpeedBrakePosition", ct.c_uint16),
        ("Fuel", ct.c_double),
        ("checksum", ct.c_uint8),
    ]


class ControlValue(ct.Structure):
    _pack_ = 1
    _fields_ = [
        ("RollCMD", ct.c_float),
        ("PitchCMD", ct.c_float),
        ("RudderCMD", ct.c_float),
        ("Throttle", ct.c_float),
    ]


class OPlaneData(ct.Structure):
    _pack_ = 1
    _fields_ = [
        ("LocationX", ct.c_float),
        ("LocationY", ct.c_float),
        ("LocationZ", ct.c_float),
        ("Roll", ct.c_float),
        ("Pitch", ct.c_float),
        ("Yaw", ct.c_float),
        ("Speed", ct.c_float),
        ("Team", ct.c_int),
        ("Resv0", ct.c_float),
        ("Resv1", ct.c_float),
        ("Resv2", ct.c_float),
    ]


class VP(ct.Structure):
    _pack_ = 1
    _fields_ = [("X", ct.c_double), ("Y", ct.c_double), ("Z", ct.c_double)]


lib_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class AIPilot:
    def __init__(self, filename: str = "AIP_DCS_ownship.dll"):
        path_to_so_file = os.path.join(lib_path, filename)
        self.dll_path = os.path.abspath(path_to_so_file)
        self.debug_enabled = os.getenv("DOGFIGHT_BT_DEBUG", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not os.path.exists(self.dll_path):
            raise FileNotFoundError(f"BT DLL not found: {self.dll_path}")
        # 2026-05-26: Keep DLL path visible when diagnosing ctypes crashes.
        self._debug_log(f"loading dll={self.dll_path}")
        self.AIPilotDLL = ct.cdll.LoadLibrary(self.dll_path)

        self.AIPilotDLL.CreateBehaviorTree.argtypes = [ct.c_int, ct.c_int]
        self.AIPilotDLL.CreateBehaviorTree.restype = None

        self.AIPilotDLL.ChangeData.argtypes = [ct.c_int, ct.c_int, ct.c_float, ct.c_int, ct.POINTER(J_NavigationData)]
        self.AIPilotDLL.ChangeData.restype = OPlaneData

        self.AIPilotDLL.Step.argtypes = [ct.POINTER(OPlaneData), ct.c_int, ct.c_void_p, ct.c_bool, ct.c_void_p, ct.c_void_p]
        self.AIPilotDLL.Step.restype = ControlValue

        self.AIPilotDLL.GetVP.argtypes = [ct.POINTER(OPlaneData)]
        self.AIPilotDLL.GetVP.restype = VP

        self.AIPilotDLL.Reset.argtypes = []
        self.AIPilotDLL.Reset.restype = None

        self.AIPilotDLL.RemoveBT.argtypes = [ct.c_int]
        self.AIPilotDLL.RemoveBT.restype = None

    def _debug_log(self, message: str) -> None:
        if self.debug_enabled:
            print(f"[native_bt] {message}", file=sys.stderr)

    def CreateBehaviorTree(self, my_id, my_force_id):
        self._debug_log(
            f"before CreateBehaviorTree dll={self.dll_path} "
            f"my_id={my_id} force={my_force_id}"
        )
        try:
            self.AIPilotDLL.CreateBehaviorTree(my_id, my_force_id)
        except OSError as exc:
            raise OSError(
                "CreateBehaviorTree failed "
                f"dll={self.dll_path} my_id={my_id} force={my_force_id}"
            ) from exc
        self._debug_log(
            f"after CreateBehaviorTree my_id={my_id} force={my_force_id}"
        )

    @staticmethod
    def BuildPlaneData(location_xyz, rotation_rpy, speed: float, team: int) -> OPlaneData:
        plane_data = OPlaneData()
        plane_data.LocationX = float(location_xyz[0])
        plane_data.LocationY = float(location_xyz[1])
        plane_data.LocationZ = float(location_xyz[2])
        plane_data.Roll = float(rotation_rpy[0])
        plane_data.Pitch = float(rotation_rpy[1])
        plane_data.Yaw = float(rotation_rpy[2])
        plane_data.Speed = float(speed)
        plane_data.Team = int(team)
        plane_data.Resv0 = 0.0
        # plane_data.Resv1 = 0.0
        plane_data.Resv1 = 100.0
        plane_data.Resv2 = 0.0
        return plane_data

    def Step(self, my_id, my_force_id, tgt_id, tgt_force_id, my_navi, tgt_navi):
        b_lockon = False
        b_flare = ct.c_bool()
        b_launch_missile = ct.c_bool()
        try:
            my_opd = self.AIPilotDLL.ChangeData(
                my_id,
                my_force_id,
                100.0,
                0,
                ct.POINTER(J_NavigationData)(my_navi),
            )
            tgt_opd = self.AIPilotDLL.ChangeData(
                tgt_id,
                tgt_force_id,
                100.0,
                0,
                ct.POINTER(J_NavigationData)(tgt_navi),
            )
            target_buffer = self._pack_plane_data_buffer(tgt_opd)
            return self.AIPilotDLL.Step(
                ct.byref(my_opd),
                1,
                ct.cast(target_buffer, ct.c_void_p),
                b_lockon,
                ct.byref(b_flare),
                ct.byref(b_launch_missile),
            )
        except OSError as exc:
            raise OSError(
                "BT Step failed "
                f"dll={self.dll_path} my_id={my_id} force={my_force_id} "
                f"target_id={tgt_id} target_force={tgt_force_id}"
            ) from exc

    def StepWithPlaneData(
        self,
        my_plane: OPlaneData,
        target_plane: OPlaneData,
        is_locked_on: bool = False,
    ):
        b_flare = ct.c_bool()
        b_launch_missile = ct.c_bool()
        target_buffer = self._pack_plane_data_buffer(target_plane)
        try:
            return self.AIPilotDLL.Step(
                ct.byref(my_plane),
                1,
                ct.cast(target_buffer, ct.c_void_p),
                ct.c_bool(bool(is_locked_on)),
                ct.byref(b_flare),
                ct.byref(b_launch_missile),
            )
        except OSError as exc:
            raise OSError(
                "BT StepWithPlaneData failed "
                f"dll={self.dll_path} my_id={my_plane.Resv0} "
                f"force={my_plane.Team} target_id={target_plane.Resv0} "
                f"target_force={target_plane.Team}"
            ) from exc

    @staticmethod
    def _pack_plane_data_buffer(plane: OPlaneData):
        packed = struct.pack(
            "fffffffifff",
            plane.LocationX,
            plane.LocationY,
            plane.LocationZ,
            plane.Roll,
            plane.Pitch,
            plane.Yaw,
            plane.Speed,
            plane.Team,
            plane.Resv0,
            plane.Resv1,
            plane.Resv2,
        )
        return ct.create_string_buffer(packed)

    def GetVP(self, my_id, my_force_id, my_navi):
        try:
            my_opd = self.AIPilotDLL.ChangeData(
                my_id,
                my_force_id,
                100.0,
                0,
                ct.POINTER(J_NavigationData)(my_navi),
            )
            return self.AIPilotDLL.GetVP(ct.byref(my_opd))
        except OSError as exc:
            raise OSError(
                "BT GetVP failed "
                f"dll={self.dll_path} my_id={my_id} force={my_force_id}"
            ) from exc

    def GetVPWithPlaneData(self, my_plane: OPlaneData):
        try:
            return self.AIPilotDLL.GetVP(ct.byref(my_plane))
        except OSError as exc:
            raise OSError(
                "BT GetVPWithPlaneData failed "
                f"dll={self.dll_path} my_id={my_plane.Resv0} force={my_plane.Team}"
            ) from exc

    def Reset(self):
        self.AIPilotDLL.Reset()

    def RemoveBT(self, my_id):
        self._debug_log(f"RemoveBT my_id={my_id}")
        self.AIPilotDLL.RemoveBT(my_id)
