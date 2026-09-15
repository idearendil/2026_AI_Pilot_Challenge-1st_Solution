#include "reduced_predictor.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <filesystem>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "generated_f16_data.h"

namespace {

using namespace reduced_f16_data;

constexpr double kDt = 1.0 / 60.0;
constexpr double kPi = 3.14159265358979323846;
constexpr double kDegToRad = kPi / 180.0;
constexpr double kRadToDeg = 180.0 / kPi;
constexpr double kFtToM = 0.3048;
constexpr double kMToFt = 1.0 / kFtToM;
constexpr double kLbfToN = 4.4482216152605;
constexpr double kLbFtToNm = 1.3558179483314;
constexpr double kPaToPsf = 0.02088543423315;
constexpr double kGravity = 9.80665;

double clamp(double value, double low, double high) {
  return std::max(low, std::min(high, value));
}

double moveToward(double value, double target, double maximum_delta) {
  return value + clamp(target - value, -maximum_delta, maximum_delta);
}

struct Vec3 {
  double x = 0.0, y = 0.0, z = 0.0;
};

Vec3 operator+(const Vec3& a, const Vec3& b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
Vec3 operator-(const Vec3& a, const Vec3& b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
Vec3 operator-(const Vec3& value) { return {-value.x, -value.y, -value.z}; }
Vec3 operator*(const Vec3& value, double scalar) { return {value.x*scalar, value.y*scalar, value.z*scalar}; }
Vec3 operator*(double scalar, const Vec3& value) { return value*scalar; }
Vec3 operator/(const Vec3& value, double scalar) { return value*(1.0/scalar); }
Vec3& operator+=(Vec3& a, const Vec3& b) { a = a+b; return a; }

double dot(const Vec3& a, const Vec3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x};
}
double norm(const Vec3& value) { return std::sqrt(std::max(0.0, dot(value, value))); }
Vec3 normalized(const Vec3& value, const Vec3& fallback = {1.0, 0.0, 0.0}) {
  const double length = norm(value);
  return length > 1.0e-9 && std::isfinite(length) ? value/length : fallback;
}

struct Quaternion {
  double w = 1.0, x = 0.0, y = 0.0, z = 0.0;
};

Quaternion operator+(const Quaternion& a, const Quaternion& b) {
  return {a.w+b.w, a.x+b.x, a.y+b.y, a.z+b.z};
}
Quaternion operator*(const Quaternion& value, double scalar) {
  return {value.w*scalar, value.x*scalar, value.y*scalar, value.z*scalar};
}
Quaternion multiply(const Quaternion& a, const Quaternion& b) {
  return {
      a.w*b.w-a.x*b.x-a.y*b.y-a.z*b.z,
      a.w*b.x+a.x*b.w+a.y*b.z-a.z*b.y,
      a.w*b.y-a.x*b.z+a.y*b.w+a.z*b.x,
      a.w*b.z+a.x*b.y-a.y*b.x+a.z*b.w,
  };
}
Quaternion unit(Quaternion value) {
  const double length = std::sqrt(value.w*value.w+value.x*value.x+value.y*value.y+value.z*value.z);
  if (length < 1.0e-12 || !std::isfinite(length)) return {};
  return value*(1.0/length);
}

Quaternion eulerToQuaternion(double roll, double pitch, double yaw) {
  const double cr=std::cos(roll*0.5), sr=std::sin(roll*0.5);
  const double cp=std::cos(pitch*0.5), sp=std::sin(pitch*0.5);
  const double cy=std::cos(yaw*0.5), sy=std::sin(yaw*0.5);
  return unit({cy*cp*cr+sy*sp*sr, cy*cp*sr-sy*sp*cr,
               cy*sp*cr+sy*cp*sr, sy*cp*cr-cy*sp*sr});
}

Vec3 quaternionToEuler(const Quaternion& q) {
  const double roll = std::atan2(2.0*(q.w*q.x+q.y*q.z),
                                 1.0-2.0*(q.x*q.x+q.y*q.y));
  const double pitch = std::asin(clamp(2.0*(q.w*q.y-q.z*q.x), -1.0, 1.0));
  const double yaw = std::atan2(2.0*(q.w*q.z+q.x*q.y),
                                1.0-2.0*(q.y*q.y+q.z*q.z));
  return {roll, pitch, yaw};
}

Vec3 rotateBodyToNed(const Quaternion& q, const Vec3& value) {
  const double xx=q.x*q.x, yy=q.y*q.y, zz=q.z*q.z;
  const double xy=q.x*q.y, xz=q.x*q.z, yz=q.y*q.z;
  const double wx=q.w*q.x, wy=q.w*q.y, wz=q.w*q.z;
  return {
      (1.0-2.0*(yy+zz))*value.x + 2.0*(xy-wz)*value.y + 2.0*(xz+wy)*value.z,
      2.0*(xy+wz)*value.x + (1.0-2.0*(xx+zz))*value.y + 2.0*(yz-wx)*value.z,
      2.0*(xz-wy)*value.x + 2.0*(yz+wx)*value.y + (1.0-2.0*(xx+yy))*value.z,
  };
}

Vec3 rotateNedToBody(const Quaternion& q, const Vec3& value) {
  const Quaternion conjugate{q.w, -q.x, -q.y, -q.z};
  return rotateBodyToNed(conjugate, value);
}

struct Atmosphere {
  double density_kgm3 = 1.225;
  double density_ratio = 1.0;
  double sound_speed_mps = 340.294;
};

Atmosphere atmosphereAt(double altitude_m) {
  const double h = clamp(altitude_m, -1000.0, 30000.0);
  double temperature = 288.15;
  double pressure = 101325.0;
  if (h <= 11000.0) {
    temperature = 288.15 - 0.0065*h;
    pressure = 101325.0*std::pow(temperature/288.15, 5.2558797);
  } else {
    temperature = 216.65;
    const double pressure_11 = 22632.06;
    pressure = pressure_11*std::exp(-kGravity*(h-11000.0)/(287.05287*temperature));
  }
  Atmosphere result;
  result.density_kgm3 = pressure/(287.05287*temperature);
  result.density_ratio = result.density_kgm3/1.225;
  result.sound_speed_mps = std::sqrt(1.4*287.05287*temperature);
  return result;
}

double schedule1(const double* x, const double* y, std::size_t n, double value) {
  return interp1(x, y, n, value);
}

struct FCSState {
  double roll_integral=0.0, pitch_integral=0.0, yaw_integral=0.0;
  double previous_roll_error=0.0, previous_pitch_error=0.0, previous_yaw_error=0.0;
  double aileron_norm=0.0, elevator_norm=0.0, rudder_norm=0.0;
  double tef_norm=0.0, lef_norm=0.0;
  double engine_n2_norm=1.0;
  double previous_nz=1.0, previous_ny=0.0;
};

struct FCSOutput {
  double aileron_rad=0.0, elevator_rad=0.0, rudder_rad=0.0;
  double actual_aileron_rad=0.0;
  double lef_rad=0.0, flaperon_mix_rad=0.0, speedbrake_rad=0.0;
  double thrust_n=0.0;
};

struct ModelState {
  Vec3 position_ned_m;
  Quaternion attitude_body_to_ned;
  Vec3 velocity_body_mps;
  Vec3 omega_body_radps;
  FCSState fcs;
  double sim_time_s=0.0;
  MPCControl last_control{0.0,0.0,0.0,1.0};
};

struct FlightCondition {
  double speed_mps=0.0, alpha_rad=0.0, beta_rad=0.0, mach=0.0;
  double cas_knots=0.0, ground_speed_fps=0.0, altitude_m=0.0;
  Atmosphere atmosphere;
};

FlightCondition flightCondition(const ModelState& state) {
  FlightCondition out;
  const Vec3& v = state.velocity_body_mps;
  out.speed_mps = std::max(1.0, norm(v));
  out.alpha_rad = std::atan2(v.z, std::max(1.0e-6, v.x));
  out.beta_rad = std::asin(clamp(v.y/out.speed_mps, -1.0, 1.0));
  out.altitude_m = -state.position_ned_m.z;
  out.atmosphere = atmosphereAt(out.altitude_m);
  out.mach = out.speed_mps/out.atmosphere.sound_speed_mps;
  out.cas_knots = out.speed_mps*1.94384449244*std::sqrt(out.atmosphere.density_ratio);
  const Vec3 ned_velocity = rotateBodyToNed(state.attitude_body_to_ned, v);
  out.ground_speed_fps = std::hypot(ned_velocity.x, ned_velocity.y)*kMToFt;
  return out;
}

FCSOutput updateFcs(ModelState& state, const MPCControl& raw_control,
                    const FlightCondition& condition, double dt) {
  const MPCControl control{
      clamp(raw_control.roll,-1.0,1.0), clamp(raw_control.pitch,-1.0,1.0),
      clamp(raw_control.rudder,-1.0,1.0), clamp(raw_control.throttle,0.0,1.0)};
  FCSState& f = state.fcs;
  const Vec3 euler = quaternionToEuler(state.attitude_body_to_ned);

  const double roll_error = control.roll - 0.31821*state.omega_body_radps.x;
  f.roll_integral = clamp(
      f.roll_integral + (1.5*roll_error-0.5*f.previous_roll_error)*dt,
      -20.0, 20.0);
  const double roll_derivative = (roll_error-f.previous_roll_error)/dt;
  const double roll_pid = 3.0*roll_error + 0.00050*f.roll_integral - 0.00125*roll_derivative;
  const double roll_command = clamp(roll_pid+control.roll, -1.0, 1.0);
  f.previous_roll_error = roll_error;
  f.aileron_norm = moveToward(f.aileron_norm, roll_command, (2.0/0.3)*dt);
  const double mach_points[] = {0.0,1.0};
  const double mach_gain[] = {1.0,0.15};
  const double speed_compensated = f.aileron_norm*schedule1(mach_points,mach_gain,2,condition.mach);

  const double alpha_points[] = {-0.5236,-0.5,0.0,0.5,0.5236};
  const double alpha_gain[] = {0.0,0.11,1.0,0.11,0.0};
  const double limited_pitch_command = clamp(control.pitch, -1.0, 0.44);
  const double elevator_scheduled = limited_pitch_command*
      schedule1(alpha_points,alpha_gain,5,condition.alpha_rad);
  const double corrected_g = f.previous_nz-std::cos(euler.y)*std::cos(euler.x);
  const double pitch_error = elevator_scheduled + 6.2*state.omega_body_radps.y - 0.020*corrected_g;
  f.pitch_integral = clamp(
      f.pitch_integral + (1.5*pitch_error-0.5*f.previous_pitch_error)*dt,
      -20.0, 20.0);
  const double pitch_pid = clamp(
      0.3000*pitch_error + 0.0250*f.pitch_integral, -1.0, 1.0);
  const double pitch_target = clamp(elevator_scheduled + 1.0472*condition.alpha_rad + pitch_pid,
                                    -1.0, 1.0);
  f.previous_pitch_error = pitch_error;
  f.elevator_norm = moveToward(f.elevator_norm, pitch_target, (2.0/0.3)*dt);

  const double speed_points[] = {80.0,100.0,150.0};
  const double yaw_gain[] = {0.0,15.0,100.0};
  const double yaw_error = control.rudder +
      schedule1(speed_points,yaw_gain,3,condition.ground_speed_fps)*state.omega_body_radps.z +
      0.25*f.previous_ny;
  f.yaw_integral = clamp(
      f.yaw_integral+(1.5*yaw_error-0.5*f.previous_yaw_error)*dt,
      -20.0, 20.0);
  const double yaw_derivative = (yaw_error-f.previous_yaw_error)/dt;
  const double yaw_pid = clamp(
      0.105500*yaw_error + 0.000010*f.yaw_integral + 0.00005*yaw_derivative,
      -1.0, 1.0);
  const double yaw_target = clamp(control.rudder+yaw_pid, -1.0, 1.0);
  f.previous_yaw_error = yaw_error;
  // Both the PID and kinematic component write fcs/rudder-pos-norm in the
  // supplied XML. FGKinemat therefore starts each frame at the PID output,
  // not at its previous kinematic output.
  f.rudder_norm = moveToward(yaw_pid, yaw_target, (2.0/0.4)*dt);

  double tef_rad = 0.0;
  if (condition.cas_knots < 250.0) tef_rad = 0.349;
  if (condition.mach > 0.9) tef_rad = -0.0349;
  f.tef_norm = moveToward(f.tef_norm, clamp(tef_rad*2.864789,-1.0,1.0), (2.0/3.0)*dt);
  double lef_rad = 0.0;
  if (condition.alpha_rad > 0.0873) lef_rad = 0.262;
  if (condition.alpha_rad > 0.2618) lef_rad = 0.436;
  if (condition.mach > 0.9) lef_rad = -0.0349;
  f.lef_norm = moveToward(f.lef_norm, clamp(lef_rad*2.293578,-1.0,1.0), (2.0/3.0)*dt);

  const double throttle_position = 2.0*control.throttle;
  const double core_target = std::min(1.0, throttle_position);
  const double spool_base = 90.0/(0.4+3.0);
  const double n = std::min(1.0, f.engine_n2_norm+0.1);
  const double denominator = 1.0+3.0*std::pow(1.0-n,3.0)+(1.0-condition.atmosphere.density_ratio);
  const double spool_factor = core_target >= f.engine_n2_norm ? 1.0 : 3.0;
  const double spool_rate_normalized = spool_factor*spool_base/std::max(0.25,denominator)/47.0;
  f.engine_n2_norm = moveToward(f.engine_n2_norm, core_target, spool_rate_normalized*dt);
  const double idle = mil_thrust_lbf*idleThrustFactor(condition.mach,condition.altitude_m*kMToFt);
  const double mil = (mil_thrust_lbf-idle)*milThrustFactor(condition.mach,condition.altitude_m*kMToFt);
  double thrust_lbf = idle+mil*f.engine_n2_norm*f.engine_n2_norm;
  const double augment = clamp(throttle_position-1.0,0.0,1.0);
  if (augment > 0.0) {
    const double maximum = max_thrust_lbf*augThrustFactor(condition.mach,condition.altitude_m*kMToFt);
    thrust_lbf += augment*(maximum-thrust_lbf);
  }

  const double left_flaperon = clamp(-f.tef_norm-speed_compensated,-1.0,1.0);
  const double right_flaperon = clamp(f.tef_norm-speed_compensated,-1.0,1.0);
  FCSOutput out;
  // The aerodynamic table reads fcs/aileron-pos-rad, which is the immediate
  // roll-rate-command scale in the supplied XML.  The left/right surface
  // kinematics are used separately by the flaperon mixer.
  out.aileron_rad = 0.375*roll_command;
  out.actual_aileron_rad = 0.375*speed_compensated;
  out.elevator_rad = 0.436*f.elevator_norm;
  out.rudder_rad = 0.524*f.rudder_norm;
  out.lef_rad = lef_rad;
  out.flaperon_mix_rad = 1.4324*(left_flaperon+right_flaperon);
  out.speedbrake_rad = 0.0;
  out.thrust_n = thrust_lbf*kLbfToN;
  return out;
}

struct DynamicsResult {
  Vec3 position_dot_ned, velocity_dot_body, omega_dot_body;
  Quaternion attitude_dot;
  Vec3 aerodynamic_force_body;
};

Vec3 inertiaMultiply(const Vec3& omega) {
  return {
      ixx_kgm2*omega.x+ixy_kgm2*omega.y+ixz_kgm2*omega.z,
      ixy_kgm2*omega.x+iyy_kgm2*omega.y+iyz_kgm2*omega.z,
      ixz_kgm2*omega.x+iyz_kgm2*omega.y+izz_kgm2*omega.z,
  };
}

Vec3 inertiaSolve(const Vec3& rhs) {
  const double a=ixx_kgm2,b=ixy_kgm2,c=ixz_kgm2,d=iyy_kgm2,e=iyz_kgm2,f=izz_kgm2;
  const double determinant=a*(d*f-e*e)-b*(b*f-c*e)+c*(b*e-c*d);
  if (std::abs(determinant)<1.0e-9) return {};
  return {
      ((d*f-e*e)*rhs.x+(c*e-b*f)*rhs.y+(b*e-c*d)*rhs.z)/determinant,
      ((c*e-b*f)*rhs.x+(a*f-c*c)*rhs.y+(b*c-a*e)*rhs.z)/determinant,
      ((b*e-c*d)*rhs.x+(b*c-a*e)*rhs.y+(a*d-b*b)*rhs.z)/determinant,
  };
}

DynamicsResult dynamics(const ModelState& state, const FCSOutput& fcs) {
  const FlightCondition c = flightCondition(state);
  const double speed_fps = std::max(3.0,c.speed_mps*kMToFt);
  const double qbar_psf = 0.5*c.atmosphere.density_kgm3*c.speed_mps*c.speed_mps*kPaToPsf;
  AeroVariables variables{
      qbar_psf, wing_area_sqft, fcs.lef_rad, fcs.flaperon_mix_rad,
      0.0, fcs.speedbrake_rad, state.omega_body_radps.y, mean_chord_ft/(2.0*speed_fps),
      c.beta_rad, fcs.aileron_rad, fcs.rudder_rad, wing_span_ft/(2.0*speed_fps),
      state.omega_body_radps.x, state.omega_body_radps.z,
      groundEffect(std::max(0.0,c.altitude_m*kMToFt/mean_chord_ft)), wing_span_ft,
      c.alpha_rad, mean_chord_ft, fcs.elevator_rad, c.mach};
  const AeroLoads loads = evaluateAero(variables);
  const double ca=std::cos(c.alpha_rad), sa=std::sin(c.alpha_rad);
  const double cb=std::cos(c.beta_rad), sb=std::sin(c.beta_rad);
  const double drag=loads.drag_lbf*kLbfToN, side=loads.side_lbf*kLbfToN;
  const double lift=loads.lift_lbf*kLbfToN;
  Vec3 aero_force{
      -drag*ca*cb-side*ca*sb+lift*sa,
      -drag*sb+side*cb,
      -drag*sa*cb-side*sa*sb-lift*ca};
  Vec3 total_force=aero_force+Vec3{fcs.thrust_n,0.0,0.0};
  Vec3 moment{loads.roll_lbft*kLbFtToNm,loads.pitch_lbft*kLbFtToNm,
              loads.yaw_lbft*kLbFtToNm};
  moment += cross(Vec3{aero_arm_x_m,aero_arm_y_m,aero_arm_z_m},aero_force);

  const Vec3 gravity_body=rotateNedToBody(state.attitude_body_to_ned,{0.0,0.0,kGravity});
  DynamicsResult out;
  out.position_dot_ned=rotateBodyToNed(state.attitude_body_to_ned,state.velocity_body_mps);
  out.velocity_dot_body=total_force/mass_kg+gravity_body-
      cross(state.omega_body_radps,state.velocity_body_mps);
  out.omega_dot_body=inertiaSolve(moment-cross(state.omega_body_radps,
                                                inertiaMultiply(state.omega_body_radps)));
  out.attitude_dot=multiply(state.attitude_body_to_ned,
                            {0.0,state.omega_body_radps.x,state.omega_body_radps.y,
                             state.omega_body_radps.z})*0.5;
  out.aerodynamic_force_body=aero_force;
  return out;
}

class ReducedF16 {
 public:
  bool reset(const MPCPublicState& input) {
    state_={};
    state_.position_ned_m={input.north_m,input.east_m,input.down_m};
    state_.attitude_body_to_ned=eulerToQuaternion(input.roll_deg*kDegToRad,
                                                   input.pitch_deg*kDegToRad,
                                                   input.yaw_deg*kDegToRad);
    state_.velocity_body_mps={input.u_mps,input.v_mps,input.w_mps};
    state_.omega_body_radps={input.p_radps,input.q_radps,input.r_radps};
    state_.sim_time_s=input.sim_time_s;
    state_.last_control={input.last_roll_cmd,input.last_pitch_cmd,
                         input.last_rudder_cmd,input.last_throttle_cmd};
    state_.fcs.aileron_norm=clamp(input.last_roll_cmd,-1.0,1.0);
    state_.fcs.elevator_norm=clamp(input.last_pitch_cmd,-1.0,1.0);
    state_.fcs.rudder_norm=clamp(input.last_rudder_cmd,-1.0,1.0);
    state_.fcs.engine_n2_norm=std::min(1.0,2.0*clamp(input.last_throttle_cmd,0.0,1.0));
    ModelState previous_frame=state_;
    last_fcs_output_=updateFcs(previous_frame,state_.last_control,
                               flightCondition(previous_frame),kDt);
    return finite();
  }

  bool step(const MPCControl& control) {
    const FlightCondition condition=flightCondition(state_);
    const FCSOutput next_fcs=updateFcs(state_,control,condition,kDt);
    // JSBSim exposes the newly computed surfaces in the current frame while
    // the propagated public rates reflect the previous derivative history.
    // One frame of force/moment latency reproduces that causal ordering.
    const FCSOutput active_fcs=last_fcs_output_;
    const DynamicsResult first=dynamics(state_,active_fcs);
    ModelState midpoint=state_;
    midpoint.position_ned_m += first.position_dot_ned*(0.5*kDt);
    midpoint.velocity_body_mps += first.velocity_dot_body*(0.5*kDt);
    midpoint.omega_body_radps += first.omega_dot_body*(0.5*kDt);
    midpoint.attitude_body_to_ned=unit(midpoint.attitude_body_to_ned+first.attitude_dot*(0.5*kDt));
    const DynamicsResult middle=dynamics(midpoint,active_fcs);
    state_.position_ned_m += middle.position_dot_ned*kDt;
    state_.velocity_body_mps += middle.velocity_dot_body*kDt;
    state_.omega_body_radps += middle.omega_dot_body*kDt;
    state_.attitude_body_to_ned=unit(state_.attitude_body_to_ned+middle.attitude_dot*kDt);
    state_.sim_time_s += kDt;
    state_.last_control={clamp(control.roll,-1.0,1.0),clamp(control.pitch,-1.0,1.0),
                         clamp(control.rudder,-1.0,1.0),clamp(control.throttle,0.0,1.0)};
    state_.fcs.previous_nz=-middle.aerodynamic_force_body.z/(mass_kg*kGravity);
    state_.fcs.previous_ny=middle.aerodynamic_force_body.y/(mass_kg*kGravity);
    last_fcs_output_=next_fcs;
    return finite();
  }

  bool finite() const {
    const double speed=norm(state_.velocity_body_mps);
    return std::isfinite(state_.position_ned_m.x)&&std::isfinite(state_.position_ned_m.y)&&
        std::isfinite(state_.position_ned_m.z)&&std::isfinite(speed)&&speed<1000.0&&
        norm(state_.omega_body_radps)<30.0&&-state_.position_ned_m.z>-2000.0;
  }
  double altitudeFt() const { return -state_.position_ned_m.z*kMToFt; }
  double speedMps() const { return norm(state_.velocity_body_mps); }
  double calibratedSpeedMps() const {
    return flightCondition(state_).cas_knots/1.94384449244;
  }
  double alphaDeg() const { return flightCondition(state_).alpha_rad*kRadToDeg; }
  double betaDeg() const { return flightCondition(state_).beta_rad*kRadToDeg; }
  double simTime() const { return state_.sim_time_s; }
  Vec3 positionNed() const { return state_.position_ned_m; }
  Vec3 velocityNed() const {
    return rotateBodyToNed(state_.attitude_body_to_ned, state_.velocity_body_mps);
  }
  Vec3 forwardNed() const { return rotateBodyToNed(state_.attitude_body_to_ned,{1.0,0.0,0.0}); }
  MPCPublicState publicState() const {
    const Vec3 euler=quaternionToEuler(state_.attitude_body_to_ned);
    return {state_.position_ned_m.x,state_.position_ned_m.y,state_.position_ned_m.z,
            euler.x*kRadToDeg,euler.y*kRadToDeg,euler.z*kRadToDeg,
            state_.velocity_body_mps.x,state_.velocity_body_mps.y,state_.velocity_body_mps.z,
            state_.omega_body_radps.x,state_.omega_body_radps.y,state_.omega_body_radps.z,
            state_.sim_time_s,state_.last_control.roll,state_.last_control.pitch,
            state_.last_control.rudder,state_.last_control.throttle};
  }
  MPCDebugState debugState() const {
    return {publicState(), last_fcs_output_.actual_aileron_rad*kRadToDeg,
            last_fcs_output_.elevator_rad*kRadToDeg,
            last_fcs_output_.rudder_rad*kRadToDeg,
            53.0+47.0*state_.fcs.engine_n2_norm, alphaDeg(), betaDeg()};
  }
 private:
  ModelState state_;
  FCSOutput last_fcs_output_;
};

struct GeometrySample {
  double range_ft=0.0, ata_deg=180.0, enemy_ata_deg=180.0, closure_mps=0.0;
};

double angleDeg(const Vec3& a,const Vec3& b) {
  return std::acos(clamp(dot(normalized(a),normalized(b)),-1.0,1.0))*kRadToDeg;
}

Vec3 targetForward(const MPCTargetSample& target) {
  const Vec3 velocity{target.vel_n_mps,target.vel_e_mps,target.vel_d_mps};
  if (norm(velocity)>10.0) return normalized(velocity);
  const Quaternion attitude=eulerToQuaternion(target.roll_deg*kDegToRad,
                                               target.pitch_deg*kDegToRad,
                                               target.yaw_deg*kDegToRad);
  return rotateBodyToNed(attitude,{1.0,0.0,0.0});
}

GeometrySample geometry(const ReducedF16& own,const MPCTargetSample& target) {
  const Vec3 own_position=own.positionNed();
  const Vec3 target_position{target.north_m,target.east_m,target.down_m};
  const Vec3 los=target_position-own_position;
  const Vec3 target_velocity{target.vel_n_mps,target.vel_e_mps,target.vel_d_mps};
  const Vec3 relative_velocity=target_velocity-own.velocityNed();
  const double closure=-dot(normalized(los),relative_velocity);
  return {norm(los)*kMToFt,angleDeg(own.forwardNed(),los),
          angleDeg(targetForward(target),-los),closure};
}

double damageRate(double ata_deg,double range_ft,double time_s) {
  if (range_ft>=500.0&&range_ft<=3000.0&&ata_deg<1.0) return (3000.0-range_ft)/2500.0;
  if (time_s>=100.0&&range_ft>=500.0&&range_ft<=3500.0&&ata_deg<2.0)
    return 0.3*(3500.0-range_ft)/3000.0;
  if (time_s>=150.0&&range_ft>=500.0&&range_ft<=4000.0&&ata_deg<3.0)
    return 0.1*(4000.0-range_ft)/3500.0;
  return 0.0;
}
double attackPotential(double ata) {
  const double bounded=clamp(ata,0.0,180.0);
  // A smooth, globally defined alignment score.  It has one maximum at
  // zero ATA, remains differentiable through the head-on region, and does
  // not encode any opponent-specific turn preference.
  const double broad=0.5*(1.0+std::cos(bounded*kDegToRad));
  const double sharp=std::exp(-0.5*std::pow(bounded/18.0,2.0));
  return 0.45*broad+0.55*sharp;
}

double threatPotential(double enemy_ata,double range_ft) {
  const double aim=std::exp(-0.5*std::pow(clamp(enemy_ata,0.0,180.0)/28.0,2.0));
  // Threat is relevant throughout a generic BFM engagement, but fades
  // smoothly when the aircraft are outside a broad visual engagement range.
  const double range_weight=clamp((12000.0-range_ft)/10000.0,0.0,1.0);
  return aim*range_weight;
}

double controlZonePotential(const GeometrySample& g,double time_s) {
  const double aim=std::exp(-0.5*std::pow(g.ata_deg/12.0,2.0));
  const double active_range=time_s<100.0?3000.0:(time_s<150.0?3500.0:4000.0);
  const double desired_range=clamp(0.55*active_range,1400.0,2300.0);
  const double range_sigma=std::max(700.0,0.30*active_range);
  const double range=std::exp(-0.5*std::pow((g.range_ft-desired_range)/range_sigma,2.0));
  const double rear=0.4+0.6*clamp((g.enemy_ata_deg-60.0)/120.0,0.0,1.0);
  return aim*range*rear;
}

double noseAdvantagePotential(const GeometrySample& g) {
  // Positive means ownship has the nose advantage; negative means the
  // opponent has it. This comparison is symmetric and uses only relative
  // geometry.
  return std::tanh((g.enemy_ata_deg-g.ata_deg)/45.0);
}

double closureUtility(const GeometrySample& g) {
  double utility=0.0;
  if (g.range_ft>=3500.0) {
    utility=clamp(g.closure_mps/200.0,-1.0,1.0);
  } else if (g.range_ft<=2500.0) {
    utility=clamp(1.0-std::abs(g.closure_mps-20.0)/150.0,-1.0,1.0);
  } else {
    const double blend=(g.range_ft-2500.0)/1000.0;
    const double near_value=clamp(1.0-std::abs(g.closure_mps-20.0)/150.0,-1.0,1.0);
    const double far_value=clamp(g.closure_mps/200.0,-1.0,1.0);
    utility=(1.0-blend)*near_value+blend*far_value;
  }
  if (g.range_ft<1200.0&&g.closure_mps>80.0)
    utility-=clamp((g.closure_mps-80.0)/150.0,0.0,1.5);
  return clamp(utility,-2.0,1.0);
}

double scoreStep(const ReducedF16& sim,const MPCTargetSample& target,
                 const MPCControl& control,const MPCControl& previous,
                 const MPCCostWeights& weights,MPCRolloutDiagnostic& diag) {
  const GeometrySample g=geometry(sim,target);
  const double dealt=damageRate(g.ata_deg,g.range_ft,sim.simTime());
  const double taken=damageRate(g.enemy_ata_deg,g.range_ft,sim.simTime());
  // Smoothly discourage leaving the engagement without introducing a hard
  // threshold tied to one opponent's observed behaviour.
  const double far=std::pow(clamp((g.range_ft-10000.0)/10000.0,0.0,1.5),2.0);
  const double overshoot= g.range_ft<1800.0 && g.closure_mps>60.0
      ? std::pow((g.closure_mps-60.0)/100.0,2.0)
          *clamp((1800.0-g.range_ft)/1200.0,0.0,1.0)
      : 0.0;
  double ground=0.0;
  if (sim.altitudeFt()<4000.0) {
    const double margin=clamp((4000.0-sim.altitudeFt())/3000.0,0.0,2.0);
    ground=margin*margin;
  }
  double envelope=0.0;
  if (sim.speedMps()<160.0) envelope+=std::pow((160.0-sim.speedMps())/60.0,2.0);
  if (sim.speedMps()>420.0) envelope+=std::pow((sim.speedMps()-420.0)/80.0,2.0);
  if (sim.calibratedSpeedMps()<130.0)
    envelope+=std::pow((130.0-sim.calibratedSpeedMps())/40.0,2.0);
  if (std::abs(sim.alphaDeg())>35.0) envelope+=std::pow((std::abs(sim.alphaDeg())-35.0)/20.0,2.0);
  if (std::abs(sim.betaDeg())>25.0) envelope+=std::pow((std::abs(sim.betaDeg())-25.0)/20.0,2.0);
  const double sink_mps=std::max(0.0,sim.velocityNed().z);
  if (sim.altitudeFt()<8000.0&&sink_mps>0.0) {
    const double proximity=clamp((8000.0-sim.altitudeFt())/6000.0,0.0,1.0);
    envelope+=proximity*std::pow(sink_mps/100.0,2.0);
  }
  const double slew=std::pow(control.roll-previous.roll,2.0)+
      std::pow(control.pitch-previous.pitch,2.0)+0.5*std::pow(control.rudder-previous.rudder,2.0)+
      0.25*std::pow(control.throttle-previous.throttle,2.0);
  diag.predicted_damage_dealt+=dealt*kDt;
  diag.predicted_damage_taken+=taken*kDt;
  diag.min_altitude_ft=std::min(diag.min_altitude_ft,sim.altitudeFt());
  diag.min_range_ft=std::min(diag.min_range_ft,g.range_ft);
  diag.final_ata_deg=g.ata_deg;
  diag.final_enemy_ata_deg=g.enemy_ata_deg;
  diag.final_speed_mps=sim.speedMps();
  return kDt*(weights.damage_dealt*dealt-weights.damage_taken*taken+
      weights.attack_geometry*attackPotential(g.ata_deg)+
      weights.control_zone*controlZonePotential(g,sim.simTime())+
      weights.closure*closureUtility(g)+
      weights.nose_advantage*noseAdvantagePotential(g)-
      weights.threat_geometry*threatPotential(g.enemy_ata_deg,g.range_ft)-
      weights.far_range*far-weights.overshoot*overshoot-
      weights.ground*ground-weights.envelope*envelope)-
      weights.control_slew*slew;
}

double terminalScore(const ReducedF16& sim,const MPCTargetSample& target,
                     const MPCCostWeights& weights) {
  const GeometrySample g=geometry(sim,target);
  return weights.terminal_geometry*(0.70*attackPotential(g.ata_deg)+
      1.10*controlZonePotential(g,sim.simTime())+
      0.45*noseAdvantagePotential(g)+0.25*closureUtility(g)-
      threatPotential(g.enemy_ata_deg,g.range_ft));
}

struct Predictor {
  explicit Predictor(const std::filesystem::path& root,int maximum)
      : max_candidates(std::max(1,maximum)) {
    if (!std::filesystem::exists(root/"aircraft"/"f16"/"f16.xml")||
        !std::filesystem::exists(root/"engine"/"F100-PW-229.xml"))
      throw std::runtime_error("copied F-16 XML assets are missing");
  }
  int max_candidates;
  std::string last_error;
  std::mutex mutex;
};

void evaluateCandidate(const MPCPublicState& initial,const MPCTargetSample* trajectory,
                       const MPCControl* controls,int candidate,int knot_count,
                       int steps_per_knot,
                       const MPCCostWeights& weights,
                       MPCRolloutDiagnostic& diag) {
  ReducedF16 sim;
  MPCTargetSample target_state=trajectory[0];
  diag={};
  diag.min_altitude_ft=std::numeric_limits<double>::infinity();
  diag.min_range_ft=std::numeric_limits<double>::infinity();
  diag.valid=sim.reset(initial)?1:0;
  MPCControl previous{initial.last_roll_cmd,initial.last_pitch_cmd,
                      initial.last_rudder_cmd,initial.last_throttle_cmd};
  int step_index=0;
  for (int knot=0;knot<knot_count&&diag.valid;++knot) {
    const MPCControl current=controls[candidate*knot_count+knot];
    for (int step=0;step<steps_per_knot;++step) {
      ++step_index;
      if (!sim.step(current)) { diag.valid=0; break; }
      // The target trajectory is generated from causal public observations by
      // the Python predictor. Do not alter it with a hidden pursuit model or
      // an opponent-specific response rule inside the rollout.
      target_state=trajectory[std::min(step_index, knot_count*steps_per_knot)];
      diag.score+=scoreStep(sim,target_state,current,previous,weights,diag);
      previous=current;
      if (sim.altitudeFt()<900.0) { diag.score-=1000.0;diag.valid=0;break; }
    }
  }
  if (diag.valid) diag.score+=terminalScore(sim,target_state,weights);
  if (!diag.valid||!std::isfinite(diag.score)) diag.score=-1.0e12;
}

}  // namespace

extern "C" {

void* MPC_Create(const char* asset_root_utf8,int max_candidates) {
  try {
    if (!asset_root_utf8) return nullptr;
    return new Predictor(std::filesystem::u8path(asset_root_utf8),max_candidates);
  } catch (...) { return nullptr; }
}
void MPC_Destroy(void* handle) { delete static_cast<Predictor*>(handle); }
const char* MPC_LastError(void* handle) {
  return handle?static_cast<Predictor*>(handle)->last_error.c_str():"invalid predictor handle";
}
int MPC_MaxCandidates(void* handle) {
  return handle?static_cast<Predictor*>(handle)->max_candidates:0;
}

int MPC_EvaluateBatch(void* handle,const MPCPublicState* initial_state,
                      const MPCTargetSample* target_trajectory,int trajectory_steps,
                      const MPCControl* controls,int candidate_count,int knot_count,
                      int steps_per_knot,
                      const MPCCostWeights* weights,
                      MPCRolloutDiagnostic* diagnostics) {
  if (!handle||!initial_state||!target_trajectory||!controls||
      !weights||!diagnostics) return 0;
  auto* predictor=static_cast<Predictor*>(handle);
  std::lock_guard<std::mutex> lock(predictor->mutex);
  if (candidate_count<1||candidate_count>predictor->max_candidates||knot_count<1||
      steps_per_knot<1||trajectory_steps<knot_count*steps_per_knot+1) return 0;
  try {
    std::atomic<int> next{0};
    const unsigned hardware=std::max(1u,std::thread::hardware_concurrency());
    const int worker_count=std::min(candidate_count,static_cast<int>(hardware));
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (int worker=0;worker<worker_count;++worker) {
      workers.emplace_back([&]() {
        while (true) {
          const int candidate=next.fetch_add(1);
          if (candidate>=candidate_count) break;
          try {
            evaluateCandidate(*initial_state,target_trajectory,controls,candidate,knot_count,
                              steps_per_knot,*weights,diagnostics[candidate]);
          } catch (...) {
            diagnostics[candidate]={};
            diagnostics[candidate].score=-1.0e12;
            diagnostics[candidate].valid=0;
          }
        }
      });
    }
    for (auto& worker:workers) worker.join();
    predictor->last_error.clear();
    return 1;
  } catch (const std::exception& error) {
    predictor->last_error=error.what();return 0;
  } catch (...) { predictor->last_error="unknown reduced predictor error";return 0; }
}

int MPC_RolloutOne(void* handle,const MPCPublicState* initial_state,
                   const MPCControl* controls,int control_count,int steps_per_control,
                   MPCPublicState* final_state) {
  if (!handle||!initial_state||!controls||!final_state||control_count<1||steps_per_control<1) return 0;
  auto* predictor=static_cast<Predictor*>(handle);
  std::lock_guard<std::mutex> lock(predictor->mutex);
  try {
    ReducedF16 sim;
    if (!sim.reset(*initial_state)) return 0;
    for (int control=0;control<control_count;++control)
      for (int step=0;step<steps_per_control;++step)
        if (!sim.step(controls[control])) return 0;
    *final_state=sim.publicState();
    predictor->last_error.clear();
    return 1;
  } catch (const std::exception& error) { predictor->last_error=error.what();return 0; }
}

int MPC_RolloutDebug(void* handle,const MPCPublicState* initial_state,
                     const MPCControl* controls,int control_count,int steps_per_control,
                     MPCDebugState* final_state) {
  if (!handle||!initial_state||!controls||!final_state||control_count<1||steps_per_control<1) return 0;
  auto* predictor=static_cast<Predictor*>(handle);
  std::lock_guard<std::mutex> lock(predictor->mutex);
  try {
    ReducedF16 sim;
    if (!sim.reset(*initial_state)) return 0;
    for (int control=0;control<control_count;++control)
      for (int step=0;step<steps_per_control;++step)
        if (!sim.step(controls[control])) return 0;
    *final_state=sim.debugState();
    predictor->last_error.clear();
    return 1;
  } catch (const std::exception& error) { predictor->last_error=error.what();return 0; }
}

const char* MPC_Version(void) {
  return "Release_MPC independent reduced F-16 6DoF predictor 0.5 / generic robust geometry cost / XML-derived / no JSBSim link";
}

}
