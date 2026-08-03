#pragma once

#include <cstdint>

#ifdef _WIN32
#  ifdef MPC_PREDICTOR_EXPORTS
#    define MPC_API __declspec(dllexport)
#  else
#    define MPC_API __declspec(dllimport)
#  endif
#else
#  define MPC_API
#endif

#pragma pack(push, 8)

struct MPCPublicState {
  double north_m, east_m, down_m;
  double roll_deg, pitch_deg, yaw_deg;
  double u_mps, v_mps, w_mps;
  double p_radps, q_radps, r_radps;
  double sim_time_s;
  double last_roll_cmd, last_pitch_cmd, last_rudder_cmd, last_throttle_cmd;
};

struct MPCTargetSample {
  double north_m, east_m, down_m;
  double vel_n_mps, vel_e_mps, vel_d_mps;
  double roll_deg, pitch_deg, yaw_deg;
};

struct MPCControl { double roll, pitch, rudder, throttle; };

struct MPCCostWeights {
  double damage_dealt, damage_taken;
  double attack_geometry, control_zone, closure, threat_geometry;
  double nose_advantage, far_range, overshoot;
  double ground, envelope, terminal_geometry, control_slew;
};

struct MPCRolloutDiagnostic {
  double score, predicted_damage_dealt, predicted_damage_taken;
  double min_altitude_ft, min_range_ft;
  double final_ata_deg, final_enemy_ata_deg, final_speed_mps;
  std::int32_t valid;
};

struct MPCDebugState {
  MPCPublicState public_state;
  double aileron_deg, elevator_deg, rudder_deg;
  double engine_n2_percent, alpha_deg, beta_deg;
};

#pragma pack(pop)

extern "C" {
MPC_API void* MPC_Create(const char* asset_root_utf8, int max_candidates);
MPC_API void MPC_Destroy(void* handle);
MPC_API const char* MPC_LastError(void* handle);
MPC_API int MPC_MaxCandidates(void* handle);
MPC_API int MPC_EvaluateBatch(
    void* handle, const MPCPublicState* initial_state,
    const MPCTargetSample* target_trajectory, int trajectory_steps,
    const MPCControl* controls, int candidate_count, int knot_count,
    int steps_per_knot,
    const MPCCostWeights* weights,
    MPCRolloutDiagnostic* diagnostics);
MPC_API int MPC_RolloutOne(
    void* handle, const MPCPublicState* initial_state,
    const MPCControl* controls, int control_count, int steps_per_control,
    MPCPublicState* final_state);
MPC_API int MPC_RolloutDebug(
    void* handle, const MPCPublicState* initial_state,
    const MPCControl* controls, int control_count, int steps_per_control,
    MPCDebugState* final_state);
MPC_API const char* MPC_Version(void);
}
