// GPU 커널: 스레드=env. 각 env 가 N step 을 내부루프로 수행.
// env0 은 전 궤적(traj, N x 10)을 기록(검증용). 모든 env 는 최종 상태(finalout, nenv x 10).
#include "fdm.cuh"

extern "C" __global__ void fdm_run(FdmState* st, const double* actions, int N, int nenv,
                                   double* traj, double* finalout){
  int e = blockIdx.x*blockDim.x + threadIdx.x;
  if(e >= nenv) return;
  FdmState s = st[e];
  FdmOut o;
  o.eci_pos[0]=o.eci_pos[1]=o.eci_pos[2]=0.0; o.alpha=0.0;
  for(int k=0;k<N;k++){
    fdm_step(&s, actions[k*4+0],actions[k*4+1],actions[k*4+2],actions[k*4+3], &o);
    if(e==0){
      traj[k*10+0]=o.eci_pos[0]; traj[k*10+1]=o.eci_pos[1]; traj[k*10+2]=o.eci_pos[2];
      traj[k*10+3]=o.euler[0];   traj[k*10+4]=o.euler[1];   traj[k*10+5]=o.euler[2];
      traj[k*10+6]=o.vUVW[0];    traj[k*10+7]=o.vUVW[1];    traj[k*10+8]=o.vUVW[2];
      traj[k*10+9]=o.alpha;
    }
  }
  finalout[e*10+0]=o.eci_pos[0]; finalout[e*10+1]=o.eci_pos[1]; finalout[e*10+2]=o.eci_pos[2];
  finalout[e*10+3]=o.euler[0];   finalout[e*10+4]=o.euler[1];   finalout[e*10+5]=o.euler[2];
  finalout[e*10+6]=o.vUVW[0];    finalout[e*10+7]=o.vUVW[1];    finalout[e*10+8]=o.vUVW[2];
  finalout[e*10+9]=o.alpha;
  st[e]=s;
}

// ===== 배치 스텝 커널 (RL env 용): 스레드=aircraft. =====
// nac 개 aircraft 각각 substeps 회 fdm_step(action_repeat, 액션 고정).
// obs[a*OBSN + *]: eci_pos3, eci_vel3, euler3, vUVW3, alpha, beta, mach, Vt, alt_asl (17)
#define OBSN 17
extern "C" __global__ void fdm_step_batch(FdmState* st, const double* actions,
                                          int nac, int substeps, double* obs){
  int a = blockIdx.x*blockDim.x + threadIdx.x;
  if(a >= nac) return;
  FdmState s = st[a];
  double ail=actions[a*4+0], el=actions[a*4+1], rud=actions[a*4+2], thr=actions[a*4+3];
  FdmOut o;
  for(int i=0;i<substeps;i++) fdm_step(&s, ail,el,rud,thr, &o);
  st[a]=s;
  double* p = obs + a*OBSN;
  p[0]=o.eci_pos[0]; p[1]=o.eci_pos[1]; p[2]=o.eci_pos[2];
  p[3]=o.eci_vel[0]; p[4]=o.eci_vel[1]; p[5]=o.eci_vel[2];
  p[6]=o.euler[0];   p[7]=o.euler[1];   p[8]=o.euler[2];
  p[9]=o.vUVW[0];    p[10]=o.vUVW[1];   p[11]=o.vUVW[2];
  p[12]=o.alpha; p[13]=o.beta; p[14]=o.mach; p[15]=o.Vt; p[16]=o.alt_asl;
}
