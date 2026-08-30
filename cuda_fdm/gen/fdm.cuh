// F16 JSBSim v1.0.0 FDM — host/CUDA 공용 단일기 단일스텝 (검증된 jsb_*.py 이식).
// 같은 소스를 (1) g++ 호스트 컴파일 (2) NVRTC 디바이스 컴파일 로 사용.
#ifndef FDM_CUH
#define FDM_CUH

// -------- host/device 매크로 --------
#if defined(__CUDACC_RTC__) || defined(__CUDACC__)
  #define HOSTDEV __device__
#else
  #include <math.h>
  #define HOSTDEV static inline
  #define __constant__
  #define __device__
  #define __global__
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// FDM 적분 dt = 1/60, FCS 컴포넌트 dt = 1/120 (load_model 시점 캡처 gotcha)
#define FDM_DT (1.0/60.0)
#define FCS_DT (1.0/120.0)
#define AERO_HB 765.0        // aero/h_b-mac-ft 상수
#define GEAR_POS_FORCE 0.30  // 외부 강제 gear-pos-norm

// -------- 저수준 헬퍼 --------
HOSTDEV double clipd(double x, double lo, double hi){ return x<lo?lo:(x>hi?hi:x); }
HOSTDEV double signd(double x){ return x>=0.0?1.0:-1.0; }
HOSTDEV double maxd(double a,double b){ return a>b?a:b; }

HOSTDEV void v_cross(const double a[3], const double b[3], double o[3]){
  o[0]=a[1]*b[2]-a[2]*b[1]; o[1]=a[2]*b[0]-a[0]*b[2]; o[2]=a[0]*b[1]-a[1]*b[0];
}
HOSTDEV void m_vec(const double A[3][3], const double v[3], double o[3]){
  o[0]=A[0][0]*v[0]+A[0][1]*v[1]+A[0][2]*v[2];
  o[1]=A[1][0]*v[0]+A[1][1]*v[1]+A[1][2]*v[2];
  o[2]=A[2][0]*v[0]+A[2][1]*v[1]+A[2][2]*v[2];
}
HOSTDEV void m_mul(const double A[3][3], const double B[3][3], double O[3][3]){
  for(int i=0;i<3;i++)for(int j=0;j<3;j++){
    O[i][j]=A[i][0]*B[0][j]+A[i][1]*B[1][j]+A[i][2]*B[2][j];
  }
}
HOSTDEV void m_T(const double A[3][3], double O[3][3]){
  for(int i=0;i<3;i++)for(int j=0;j<3;j++) O[i][j]=A[j][i];
}

// -------- FGTable 보간 (jsb_tables.py 정확복제) --------
HOSTDEV double table1d(const double* x, const double* y, int n, double key){
  if(key<=x[0]) return y[0];
  if(key>=x[n-1]) return y[n-1];
  int r=1;
  while(r<n-1 && x[r]<key) r++;
  double span=x[r]-x[r-1];
  double factor;
  if(span!=0.0){ factor=(key-x[r-1])/span; if(factor>1.0) factor=1.0; }
  else factor=1.0;
  return factor*(y[r]-y[r-1])+y[r-1];
}
HOSTDEV double table2d(const double* rx, int nr, const double* cx, int nc,
                       const double* d, double rowKey, double colKey){
  int r=1;
  while(r<nr-1 && rx[r]<rowKey) r++;
  while(r>1 && rx[r-1]>rowKey) r--;
  int c=1;
  while(c<nc-1 && cx[c]<colKey) c++;
  while(c>1 && cx[c-1]>colKey) c--;
  double rF=(rowKey-rx[r-1])/(rx[r]-rx[r-1]);
  double cF=(colKey-cx[c-1])/(cx[c]-cx[c-1]);
  rF = rF>1.0?1.0:(rF<0.0?0.0:rF);
  cF = cF>1.0?1.0:(cF<0.0?0.0:cF);
  // d[row*nc+col]
  double d_r1_cm1=d[r*nc+(c-1)], d_rm1_cm1=d[(r-1)*nc+(c-1)];
  double d_r1_c=d[r*nc+c],       d_rm1_c=d[(r-1)*nc+c];
  double col1=rF*(d_r1_cm1-d_rm1_cm1)+d_rm1_cm1;
  double col2=rF*(d_r1_c-d_rm1_c)+d_rm1_c;
  return col1+cF*(col2-col1);
}

// 상수/테이블/공력 (자동생성)
#include "f16_gen.cuh"

// -------- 사원수 / 프레임 (jsb_frames.py) --------
HOSTDEV void quat_to_T(const double q[4], double T[3][3]){
  double q0=q[0],q1=q[1],q2=q[2],q3=q[3];
  double q0q0=q0*q0,q1q1=q1*q1,q2q2=q2*q2,q3q3=q3*q3;
  double q0q1=q0*q1,q0q2=q0*q2,q0q3=q0*q3,q1q2=q1*q2,q1q3=q1*q3,q2q3=q2*q3;
  T[0][0]=q0q0+q1q1-q2q2-q3q3; T[0][1]=2.0*(q1q2+q0q3); T[0][2]=2.0*(q1q3-q0q2);
  T[1][0]=2.0*(q1q2-q0q3); T[1][1]=q0q0-q1q1+q2q2-q3q3; T[1][2]=2.0*(q2q3+q0q1);
  T[2][0]=2.0*(q1q3+q0q2); T[2][1]=2.0*(q2q3-q0q1); T[2][2]=q0q0-q1q1-q2q2+q3q3;
}
HOSTDEV void quat_normalize(double q[4]){
  double mag=sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
  if(mag==0.0) return;
  double inv=1.0/mag; q[0]*=inv;q[1]*=inv;q[2]*=inv;q[3]*=inv;
}
HOSTDEV void quat_qdot(const double q[4], const double pqr[3], double o[4]){
  double p=pqr[0],qq=pqr[1],r=pqr[2];
  o[0]=-0.5*(q[1]*p+q[2]*qq+q[3]*r);
  o[1]= 0.5*(q[0]*p-q[3]*qq+q[2]*r);
  o[2]= 0.5*(q[3]*p+q[0]*qq-q[1]*r);
  o[3]= 0.5*(-q[2]*p+q[1]*qq+q[0]*r);
}
HOSTDEV void euler_to_quat(double phi,double tht,double psi,double q[4]){
  double thtd2=0.5*tht,psid2=0.5*psi,phid2=0.5*phi;
  double St=sin(thtd2),Sp=sin(psid2),Sf=sin(phid2);
  double Ct=cos(thtd2),Cp=cos(psid2),Cf=cos(phid2);
  double CfCt=Cf*Ct,CfSt=Cf*St,SfSt=Sf*St,SfCt=Sf*Ct;
  q[0]=CfCt*Cp+SfSt*Sp; q[1]=SfCt*Cp-CfSt*Sp; q[2]=CfSt*Cp+SfCt*Sp; q[3]=CfCt*Sp-SfSt*Cp;
  quat_normalize(q);
}
HOSTDEV void mat_to_quat(const double m[3][3], double Q[4]){
  // FGMatrix33 data[] column-major: d[k]=m(i,j), i-1=k%3, j-1=k//3
  double d[9]={m[0][0],m[1][0],m[2][0], m[0][1],m[1][1],m[2][1], m[0][2],m[1][2],m[2][2]};
  double t[4]={1.0+d[0]+d[4]+d[8], 1.0+d[0]-d[4]-d[8], 1.0-d[0]+d[4]-d[8], 1.0-d[0]-d[4]+d[8]};
  int idx=0; for(int i=1;i<4;i++) if(t[i]>t[idx]) idx=i;
  if(idx==0){ Q[0]=0.5*sqrt(t[0]); Q[1]=0.25*(d[7]-d[5])/Q[0]; Q[2]=0.25*(d[2]-d[6])/Q[0]; Q[3]=0.25*(d[3]-d[1])/Q[0]; }
  else if(idx==1){ Q[1]=0.5*sqrt(t[1]); Q[0]=0.25*(d[7]-d[5])/Q[1]; Q[2]=0.25*(d[3]+d[1])/Q[1]; Q[3]=0.25*(d[2]+d[6])/Q[1]; }
  else if(idx==2){ Q[2]=0.5*sqrt(t[2]); Q[0]=0.25*(d[2]-d[6])/Q[2]; Q[1]=0.25*(d[3]+d[1])/Q[2]; Q[3]=0.25*(d[7]+d[5])/Q[2]; }
  else { Q[3]=0.5*sqrt(t[3]); Q[0]=0.25*(d[3]-d[1])/Q[3]; Q[1]=0.25*(d[6]+d[2])/Q[3]; Q[2]=0.25*(d[7]+d[5])/Q[3]; }
}
HOSTDEV void mat_to_euler(const double m[3][3], double e[3]){
  double d6=m[0][2]; double theta; int gimbal=0;
  if(d6<=-1.0){ theta=0.5*M_PI; gimbal=1; }
  else if(1.0<=d6){ theta=-0.5*M_PI; gimbal=1; }
  else theta=asin(-d6);
  double phi,psi;
  if(gimbal){ phi=atan2(-m[2][1],m[1][1]); psi=0.0; }
  else { phi=atan2(m[1][2],m[2][2]); psi=atan2(m[0][1],m[0][0]); if(psi<0.0) psi+=2.0*M_PI; }
  e[0]=phi; e[1]=theta; e[2]=psi;
}
HOSTDEV void Ti2ec_from_epa(double epa, double T[3][3]){
  double ce=cos(epa),se=sin(epa);
  T[0][0]=ce;T[0][1]=se;T[0][2]=0.0;
  T[1][0]=-se;T[1][1]=ce;T[1][2]=0.0;
  T[2][0]=0.0;T[2][1]=0.0;T[2][2]=1.0;
}
// location_derived: 필요한 것 radius,mLat,Tl2ec
HOSTDEV void location_derived(const double ecef[3], double* radius, double* mLat, double Tl2ec[3][3]){
  double x=ecef[0],y=ecef[1],z=ecef[2];
  double rad=sqrt(x*x+y*y+z*z);
  double rxy=sqrt(x*x+y*y);
  double sinLon,cosLon,sinLat,cosLat;
  if(rxy==0.0){ sinLon=0.0; cosLon=1.0; } else { sinLon=y/rxy; cosLon=x/rxy; }
  if(rad==0.0){ sinLat=0.0; cosLat=1.0; } else { sinLat=z/rad; cosLat=rxy/rad; }
  double mlat=(rxy==0.0 && z==0.0)?0.0:atan2(z,rxy);
  double Tec2l[3][3]={
    {-cosLon*sinLat, -sinLon*sinLat, cosLat},
    {-sinLon, cosLon, 0.0},
    {-cosLon*cosLat, -sinLon*cosLat, -sinLat}};
  m_T(Tec2l, Tl2ec);
  *radius=rad; *mLat=mlat;
}
HOSTDEV void gravity_j2(const double ecef[3], double latitude, double o[3]){
  double x=ecef[0],y=ecef[1],z=ecef[2];
  double r=sqrt(x*x+y*y+z*z);
  double sinLat=sin(latitude);
  double adivr=SEMI_MAJOR/r;
  double preCommon=1.5*J2const*adivr*adivr;
  double xy=1.0-5.0*(sinLat*sinLat);
  double zz=3.0-5.0*(sinLat*sinLat);
  double GMOverr2=GM/(r*r);
  o[0]=-GMOverr2*((1.0+(preCommon*xy))*x/r);
  o[1]=-GMOverr2*((1.0+(preCommon*xy))*y/r);
  o[2]=-GMOverr2*((1.0+(preCommon*zz))*z/r);
}

// -------- 표준대기 (jsb_atmos.py) --------
HOSTDEV double atm_geopot(double h){ double ER=EarthRadius; return (h*ER)/(ER+h); }
HOSTDEV double atm_geometric(double gp){ double ER=EarthRadius; return (gp*ER)/(ER-gp); }
HOSTDEV double atm_temperature(double altitude){
  double gp=atm_geopot(altitude);
  if(gp>=0.0) return table1d(ATM_ALT,ATM_TMP,NATM,gp);
  return table1d(ATM_ALT,ATM_TMP,NATM,0.0)+gp*ATM_LAPSE[0];
}
HOSTDEV double atm_pressure(double altitude){
  double gp=atm_geopot(altitude);
  double BaseAlt=ATM_ALT[0]; int b=0;
  for(int bb=0;bb<NATM-2;bb++){
    double testAlt=ATM_ALT[bb+1];
    if(gp<testAlt){ b=bb; break; }
    BaseAlt=testAlt; b=bb+1;
  }
  double Tmb=atm_temperature(atm_geometric(BaseAlt));
  double deltaH=gp-BaseAlt; double Lmb=ATM_LAPSE[b];
  if(Lmb!=0.0){ double Exp=g0*Mair/(Rstar*Lmb); double factor=Tmb/(Tmb+Lmb*deltaH); return ATM_PBRK[b]*pow(factor,Exp); }
  return ATM_PBRK[b]*exp(-g0*Mair*deltaH/(Rstar*Tmb));
}
HOSTDEV double atm_density_altitude(double density){
  int b=0;
  for(int bb=0;bb<NATM-2;bb++){
    if(density>=ATM_DBRK[bb+1]){ b=bb; break; }
    b=bb+1;
  }
  double Tmb=ATM_TMP[b],Hb=ATM_ALT[b],Lmb=ATM_LAPSE[b],pb=ATM_DBRK[b];
  double da;
  if(Lmb!=0.0){ double Exp=-1.0/(1.0+(g0*Mair)/(Rstar*Lmb)); da=Hb+(Tmb/Lmb)*(pow(density/pb,Exp)-1.0); }
  else { double Factor=-(Rstar*Tmb)/(g0*Mair); da=Hb+Factor*log(density/pb); }
  return atm_geometric(da);
}
typedef struct { double T,P,rho,a,sigma,densalt; } Atmos;
HOSTDEV void atm_calculate(double altitude, Atmos* o){
  double T=atm_temperature(altitude);
  double P=atm_pressure(altitude);
  double rho=P/(Reng*T);
  o->T=T; o->P=P; o->rho=rho;
  o->a=sqrt(SHRatio*Reng*T);
  double rho_sl=StdDaySLpressure/(Reng*StdDaySLtemperature);
  o->sigma=rho/rho_sl;
  o->densalt=atm_density_altitude(rho);
}

// -------- 질량/관성 (jsb_massbalance.py) --------
#define EMPTY_WT 17400.0
#define PILOT_WT 230.0
typedef struct { double cg[3]; double mass_slug; double J[3][3]; double Jinv[3][3]; double RPBody[3]; } Mass;
HOSTDEV void mb_stb(const double r[3], const double cg[3], double o[3]){
  o[0]=INCHTOFT*(cg[0]-r[0]); o[1]=INCHTOFT*(r[1]-cg[1]); o[2]=INCHTOFT*(cg[2]-r[2]);
}
HOSTDEV void mb_pm_inertia(double mass_sl, const double r[3], const double cg[3], double M[3][3]){
  double vx=INCHTOFT*(cg[0]-r[0]), vy=INCHTOFT*(r[1]-cg[1]), vz=INCHTOFT*(cg[2]-r[2]);
  double sx=mass_sl*vx, sy=mass_sl*vy, sz=mass_sl*vz;
  double xx=sx*vx, yy=sy*vy, zz=sz*vz, xy=-sx*vy, xz=-sx*vz, yz=-sy*vz;
  M[0][0]=yy+zz; M[0][1]=xy; M[0][2]=xz;
  M[1][0]=xy; M[1][1]=xx+zz; M[1][2]=yz;
  M[2][0]=xz; M[2][1]=yz; M[2][2]=xx+yy;
}
HOSTDEV void mb_compute(double fuel_total_lbs, Mass* o){
  const double BASE_CG[3]={-193.0,0.0,-5.1};
  const double PILOT_LOC[3]={-336.2,0.0,0.0};
  const double TANK0[3]={-174.4,65.0,5.0};
  const double TANK1[3]={-174.4,-65.0,5.0};
  const double AERORP[3]={-189.5,0.0,3.9};
  const double BASEJ[3][3]={{9496.0,0.0,-982.0},{0.0,55814.0,0.0},{-982.0,0.0,63100.0}};
  double t0=fuel_total_lbs/2.0, t1=t0;
  double weight=EMPTY_WT+(t0+t1)+PILOT_WT;
  for(int i=0;i<3;i++)
    o->cg[i]=(EMPTY_WT*BASE_CG[i]+PILOT_WT*PILOT_LOC[i]+t0*TANK0[i]+t1*TANK1[i])/weight;
  double mJ[3][3]; for(int i=0;i<3;i++)for(int j=0;j<3;j++) mJ[i][j]=BASEJ[i][j];
  double tmp[3][3];
  mb_pm_inertia(LBTOSLUG*EMPTY_WT,BASE_CG,o->cg,tmp); for(int i=0;i<3;i++)for(int j=0;j<3;j++) mJ[i][j]+=tmp[i][j];
  mb_pm_inertia(LBTOSLUG*PILOT_WT,PILOT_LOC,o->cg,tmp); for(int i=0;i<3;i++)for(int j=0;j<3;j++) mJ[i][j]+=tmp[i][j];
  mb_pm_inertia(LBTOSLUG*t0,TANK0,o->cg,tmp); for(int i=0;i<3;i++)for(int j=0;j<3;j++) mJ[i][j]+=tmp[i][j];
  mb_pm_inertia(LBTOSLUG*t1,TANK1,o->cg,tmp); for(int i=0;i<3;i++)for(int j=0;j<3;j++) mJ[i][j]+=tmp[i][j];
  double Ixx=mJ[0][0],Iyy=mJ[1][1],Izz=mJ[2][2];
  double Ixy=-mJ[0][1],Ixz=-mJ[0][2],Iyz=-mJ[1][2];
  double k1=(Iyy*Izz-Iyz*Iyz), k2=(Iyz*Ixz+Ixy*Izz), k3=(Ixy*Iyz+Iyy*Ixz);
  double denom=1.0/(Ixx*k1-Ixy*k2-Ixz*k3);
  k1*=denom;k2*=denom;k3*=denom;
  double k4=(Izz*Ixx-Ixz*Ixz)*denom, k5=(Ixy*Ixz+Iyz*Ixx)*denom, k6=(Ixx*Iyy-Ixy*Ixy)*denom;
  o->Jinv[0][0]=k1;o->Jinv[0][1]=k2;o->Jinv[0][2]=k3;
  o->Jinv[1][0]=k2;o->Jinv[1][1]=k4;o->Jinv[1][2]=k5;
  o->Jinv[2][0]=k3;o->Jinv[2][1]=k5;o->Jinv[2][2]=k6;
  o->J[0][0]=Ixx;o->J[0][1]=-Ixy;o->J[0][2]=-Ixz;
  o->J[1][0]=-Ixy;o->J[1][1]=Iyy;o->J[1][2]=-Iyz;
  o->J[2][0]=-Ixz;o->J[2][1]=-Iyz;o->J[2][2]=Izz;
  o->mass_slug=LBTOSLUG*weight;
  mb_stb(AERORP,o->cg,o->RPBody);
}

// -------- FCS (jsb_fcs.py) --------
HOSTDEV int equal_roundoff(double a,double b){ return fabs(a-b)<=1e-9*maxd(maxd(1.0,fabs(a)),fabs(b)); }
HOSTDEV double kinemat_run(const double* det,const double* tim,int n,double* output,
                           double inp,int doscale,int has_seed,double seed){
  double dt0=FCS_DT;
  double Input=inp;
  if(doscale) Input*=det[n-1];
  double Output = has_seed?seed:(*output);
  Input=clipd(Input,det[0],det[n-1]);
  while(dt0>0.0 && !equal_roundoff(Input,Output)){
    int ind=1;
    while(1){
      int cond = (Input<Output)?(det[ind]<Output):(det[ind]<=Output);
      if(!cond) break;
      ind++;
      if(ind>=n) break;
    }
    if(ind>=n) ind=n-1;
    if(tim[ind]<=0.0){ Output=Input; break; }
    double Rate=(det[ind]-det[ind-1])/tim[ind];
    double ThisInput=clipd(Input,det[ind-1],det[ind]);
    double ThisDt=fabs((ThisInput-Output)/Rate);
    if(dt0<ThisDt){ ThisDt=dt0; if(Output<Input) Output+=ThisDt*Rate; else Output-=ThisDt*Rate; }
    else Output=ThisInput;
    dt0-=ThisDt;
  }
  *output=Output;
  return Output;
}
// PID 상태: ip=input_prev, ip2=input_prev2, iout=i_out_total
HOSTDEV double pid_run(double kp,double ki,double kd,double* ip,double* ip2,double* iout,
                       double Input,double trigger,int has_clip,double clo,double chi){
  double Dval=(Input-*ip)/FCS_DT;
  double test=trigger, I_out_delta=0.0;
  if(fabs(test)<0.000001) I_out_delta=1.5*Input-0.5*(*ip);
  if(test<0.0) *iout=0.0;
  *iout += ki*FCS_DT*I_out_delta;
  double Output=kp*Input + (*iout) + kd*Dval;
  *ip2 = (test<0.0)?0.0:(*ip);
  *ip = Input;
  if(has_clip) Output=clipd(Output,clo,chi);
  return Output;
}
HOSTDEV double aerosurface_zc(double Input,double InMin,double InMax,double OutMin,double OutMax){
  if(Input==0.0) return 0.0;
  if(Input>0.0) return (Input/InMax)*OutMax;
  return (Input/InMin)*OutMin;
}

// FCS 상태 (FdmState 내에)
typedef struct {
  double k_tef,k_aileron,k_elevator,k_rudder,k_gear,k_lef;   // kinemat outputs
  double roll_ip,roll_ip2,roll_iout;
  double gload_ip,gload_ip2,gload_iout;
  double yaw_ip,yaw_ip2,yaw_iout;
} FcsState;
// FCS 출력 (aero 입력)
typedef struct {
  double aileron_pos_rad,elevator_pos_rad,rudder_pos_rad,lef_pos_rad;
  double flaperon_mix_rad,speedbrake_pos_rad,gear_pos_norm;
} FcsOut;

HOSTDEV void fcs_step(FcsState* f,
    double c_aileron,double c_elevator,double c_rudder,double c_gear,
    double c_pitch_trim,double c_yaw_trim,
    double alpha,double mach,double vc,double vg,
    double n_pilot_y,double n_pilot_z,double p_aero,double q_aero,double r_aero,
    double pitch_rad,double roll_rad, FcsOut* out){
  // detents/times
  const double d_tef[3]={-1.0,0.0,1.0}, t_tef[3]={3.0,0.0,3.0};
  const double d_ail[2]={-1.0,1.0}, t_ail[2]={0.3,0.3};
  const double d_ele[2]={-1.0,1.0}, t_ele[2]={0.3,0.3};
  const double d_rud[2]={-1.0,1.0}, t_rud[2]={0.4,0.4};
  const double d_gear[2]={0.0,1.0}, t_gear[2]={0.0,5.0};
  const double d_lef[2]={-1.0,1.0}, t_lef[2]={3.0,3.0};

  // ===== Flaps =====
  double tef_pos_rad;
  if(vc<250.0) tef_pos_rad=0.349;
  else if(mach>0.9) tef_pos_rad=-0.0349;
  else tef_pos_rad=0.0;
  double tef_pos_norm=2.864789*tef_pos_rad;
  double tef_control=kinemat_run(d_tef,t_tef,3,&f->k_tef,tef_pos_norm,1,0,0.0);

  // ===== Roll =====
  double roll_rate_norm=0.31821*p_aero;
  double roll_trim_error=c_aileron-roll_rate_norm;
  double ail_trig=(vc<20.0)?0.0:1.0;
  double roll_rate_pid=pid_run(3.0,0.0005,-0.00125,&f->roll_ip,&f->roll_ip2,&f->roll_iout,roll_trim_error,ail_trig,0,0,0);
  double roll_rate_command=clipd(roll_rate_pid+c_aileron,-1.0,1.0);
  double left_aileron_pos_norm=kinemat_run(d_ail,t_ail,2,&f->k_aileron,roll_rate_command,1,0,0.0);
  double aileron_speed_comp=FCS_AIL_SPEED(mach)*left_aileron_pos_norm;
  double left_flaperon_norm=clipd(-tef_control-aileron_speed_comp,-1.0,1.0);
  double right_flaperon_norm=clipd(tef_control-aileron_speed_comp,-1.0,1.0);
  double flaperon_summer=left_flaperon_norm+right_flaperon_norm;
  out->flaperon_mix_rad=1.4324*flaperon_summer;
  double left_aileron_pos_rad=aerosurface_zc(aileron_speed_comp,-1.0,1.0,-0.375,0.375);
  double right_aileron_pos_rad=aerosurface_zc(-aileron_speed_comp,-1.0,1.0,-0.375,0.375);
  out->aileron_pos_rad=aerosurface_zc(roll_rate_command,-1.0,1.0,-0.375,0.375);

  // ===== Pitch =====
  double n_pilot_z_corr=cos(pitch_rad)*cos(roll_rad);
  double g_load_corrected=n_pilot_z-n_pilot_z_corr;
  double elevator_cmd_limiter=clipd(c_elevator+c_pitch_trim,-1.0,0.44);
  double elevator_scheduler=FCS_ELEV_SCHED(alpha)*elevator_cmd_limiter;
  double alpha_limiter_norm=1.0472*alpha;
  double pitch_rate_norm=6.2*q_aero;
  double g_load_norm=0.020*g_load_corrected;
  double pitch_trim_error=elevator_scheduler+pitch_rate_norm-g_load_norm;
  double elev_trig=(vc<5.0)?0.0:1.0;
  double g_load_pid=pid_run(0.3,0.025,0.0,&f->gload_ip,&f->gload_ip2,&f->gload_iout,pitch_trim_error,elev_trig,1,-1.0,1.0);
  double pitch_scheduler=clipd(elevator_scheduler+alpha_limiter_norm+g_load_pid,-1.0,1.0);
  double elevator_pos_norm=kinemat_run(d_ele,t_ele,2,&f->k_elevator,pitch_scheduler,1,0,0.0);
  out->elevator_pos_rad=aerosurface_zc(elevator_pos_norm,-1.0,1.0,-0.436,0.436);

  // ===== Yaw =====
  double yaw_rate_norm=FCS_YAW_RATE(vg)*r_aero;
  double yaw_load_norm=0.25*n_pilot_y;
  double yaw_trim_error=c_rudder+yaw_rate_norm+yaw_load_norm;
  double rud_trig=(vc<10.0)?0.0:1.0;
  double yaw_load_pid=pid_run(0.1055,0.00001,0.00005,&f->yaw_ip,&f->yaw_ip2,&f->yaw_iout,yaw_trim_error,rud_trig,1,-1.0,1.0);
  double yaw_scheduler=clipd(c_rudder+c_yaw_trim+yaw_load_pid,-1.0,1.0);
  double rudder_pos_norm=kinemat_run(d_rud,t_rud,2,&f->k_rudder,yaw_scheduler,1,1,yaw_load_pid);
  out->rudder_pos_rad=aerosurface_zc(rudder_pos_norm,-1.0,1.0,-0.524,0.524);

  // ===== Landing Gear =====
  double gear_wow=0.0;
  double gear_pos_norm=kinemat_run(d_gear,t_gear,2,&f->k_gear,c_gear,1,1,GEAR_POS_FORCE);
  out->gear_pos_norm=gear_pos_norm;

  // ===== LEF =====
  double lef_pos_rad;
  if(gear_wow==1.0 && gear_pos_norm>0.0) lef_pos_rad=-0.0349;
  else if(gear_pos_norm==0.0 && alpha>0.2618) lef_pos_rad=0.436;
  else if(gear_wow==0.0 && alpha>0.0873) lef_pos_rad=0.262;
  else if(mach>0.9) lef_pos_rad=-0.0349;
  else lef_pos_rad=0.0;
  out->lef_pos_rad=lef_pos_rad;
  double lef_pos_norm=2.293578*lef_pos_rad;
  kinemat_run(d_lef,t_lef,2,&f->k_lef,lef_pos_norm,1,0,0.0);

  // ===== Speedbrake =====
  out->speedbrake_pos_rad=0.0;
}

// -------- 엔진 (jsb_propulsion.py) --------
typedef struct { double N1,N2,N2norm,FuelFlow_pph; } EngState;
typedef struct { double thrust,fuel_burn; double forces[3]; double moments[3]; } EngOut;
HOSTDEV double eng_seek(double v,double target,double accel,double decel,double dt){
  if(v>target){ v-=dt*decel; if(v<target) v=target; }
  else if(v<target){ v+=dt*accel; if(v>target) v=target; }
  return v;
}
HOSTDEV double eng_spool(double delay,double densratio,double N2norm){
  double n=N2norm+0.1; if(n>1.0) n=1.0;
  double om=1.0-n;
  return delay/(1.0+3.0*om*om*om+(1.0-densratio));
}
HOSTDEV void eng_step(EngState* e,double throttle_pos_norm,double mach,double densalt,
                      double T,double densratio,double dt,const double cg[3],EngOut* o){
  double ThrottlePos=throttle_pos_norm, AugmentCmd;
  if(ThrottlePos>1.0){ AugmentCmd=ThrottlePos-1.0; ThrottlePos-=AugmentCmd; }
  else AugmentCmd=0.0;
  double N1_factor=MAXN1-IDLEN1, N2_factor=MAXN2-IDLEN2, d=BYPASSRATIO;
  double base=90.0/(d+3.0);
  double n2up=eng_spool(1.0*base,densratio,e->N2norm), n2dn=eng_spool(3.0*base,densratio,e->N2norm);
  double n1up=eng_spool(1.0*base,densratio,e->N2norm), n1dn=eng_spool(2.4*base,densratio,e->N2norm);
  e->N2=eng_seek(e->N2,IDLEN2+ThrottlePos*N2_factor,n2up,n2dn,dt);
  e->N1=eng_seek(e->N1,IDLEN1+ThrottlePos*N1_factor,n1up,n1dn,dt);
  e->N2norm=(e->N2-IDLEN2)/N2_factor;
  double idlethrust=MILTHRUST*ENG_IDLE(mach,densalt);
  double milthrust=(MILTHRUST-idlethrust)*ENG_MIL(mach,densalt);
  double thrust=idlethrust+milthrust*e->N2norm*e->N2norm;
  double om=1.0-e->N2norm;
  double correctedTSFC=TSFC*sqrt(T/389.7)*(0.84+om*om);
  e->FuelFlow_pph=eng_seek(e->FuelFlow_pph,thrust*correctedTSFC,1000.0,10000.0,dt);
  if(e->FuelFlow_pph<IdleFF) e->FuelFlow_pph=IdleFF;
  thrust=thrust*(1.0-BLEEDDEMAND);
  if(AugmentCmd>0.0){
    double tdiff=(MAXTHRUST*ENG_AUG(mach,densalt))-thrust;
    thrust+=tdiff*AugmentCmd;
    e->FuelFlow_pph=eng_seek(e->FuelFlow_pph,thrust*ATSFC,5000.0,10000.0,dt);
  }
  double rz=INCHTOFT*(cg[2]-0.0), ry=INCHTOFT*(0.0-cg[1]);
  o->thrust=thrust;
  o->forces[0]=thrust; o->forces[1]=0.0; o->forces[2]=0.0;
  o->moments[0]=0.0; o->moments[1]=rz*thrust; o->moments[2]=-ry*thrust;
  o->fuel_burn=e->FuelFlow_pph/3600.0*dt;
}

// -------- Auxiliary (jsb_auxiliary.py) --------
HOSTDEV double pitot_total_pressure(double mach,double p){
  if(mach<0.0) return p;
  if(mach<1.0){ double m2=1.0+0.2*mach*mach; return p*m2*m2*m2*sqrt(m2); } // ^3.5
  return p*166.92158009316827*pow(mach,7.0)/pow(7.0*mach*mach-1.0,2.5);
}
HOSTDEV double mach_from_impact_pressure(double qc,double p){
  double A=qc/p+1.0;
  double M=sqrt(5.0*(pow(A,1.0/3.5)-1.0));
  if(M>1.0){ for(int i=0;i<10;i++){ double t=1.0-1.0/(7.0*M*M); M=0.8812848543473311*sqrt(A*pow(t,2.5)); } }
  return M;
}
HOSTDEV double vcalibrated_from_mach(double mach,double p){
  double qc=pitot_total_pressure(mach,p)-p;
  return STD_SL_SOUNDSPEED*mach_from_impact_pressure(qc,StdDaySLpressure);
}
typedef struct {
  double alpha,beta,Vt,qbar,mach,p_aero,q_aero,r_aero,vg,n_pilot_y,n_pilot_z,vc_kts;
} Aux;
HOSTDEV void auxiliary(const double vUVW[3],const double vPQR[3],const double vVel_ned[3],
    double rho,double a,double P,const double cg[3],
    const double prev_ba[3],const double prev_pqridot[3],const double prev_pqri[3],
    double SLGrav, Aux* o){
  double u=vUVW[0],v=vUVW[1],w=vUVW[2];
  double AeroU2=u*u,AeroV2=v*v,AeroW2=w*w;
  double mUW=AeroU2+AeroW2; double Vt2=mUW+AeroV2; double Vt=sqrt(Vt2);
  double alpha=0.0,beta=0.0;
  if(Vt>0.001){ beta=atan2(v,sqrt(mUW)); if(mUW>=1e-6) alpha=atan2(w,u); }
  o->alpha=alpha; o->beta=beta; o->Vt=Vt;
  o->qbar=0.5*rho*Vt2; o->mach=Vt/a;
  o->vg=sqrt(vVel_ned[0]*vVel_ned[0]+vVel_ned[1]*vVel_ned[1]);
  const double EYEPOINT[3]={-336.2,0.0,29.5};
  double ToEyePt[3]; mb_stb(EYEPOINT,cg,ToEyePt);
  double c1[3],c2[3],c3[3];
  v_cross(prev_pqridot,ToEyePt,c1);
  v_cross(prev_pqri,ToEyePt,c2); v_cross(prev_pqri,c2,c3);
  double ay=prev_ba[1]+c1[1]+c3[1];
  double az=prev_ba[2]+c1[2]+c3[2];
  o->n_pilot_y=ay/SLGrav; o->n_pilot_z=az/SLGrav;
  double mach=o->mach;
  double vcas=(fabs(mach)>0.0)?vcalibrated_from_mach(mach,P):0.0;
  o->vc_kts=vcas*FPSTOKTS;
  o->p_aero=vPQR[0]; o->q_aero=vPQR[1]; o->r_aero=vPQR[2];
}

// -------- Accelerations (jsb_accel.py) --------
typedef struct { double vUVWidot[3],vPQRidot[3],vBodyAccel[3],vPQRi[3]; } Accel;
HOSTDEV void accelerations(const double force[3],const double moment[3],double mass,
    const double J[3][3],const double Jinv[3][3],const double vUVW[3],const double vPQR[3],
    const double Ti2b[3][3],const double eci_pos[3],const double vGrav[3], Accel* o){
  double omega[3]={0.0,0.0,ROTATION_RATE};
  double Ti2b_omega[3]; m_vec(Ti2b,omega,Ti2b_omega);
  double vPQRi[3]={vPQR[0]+Ti2b_omega[0],vPQR[1]+Ti2b_omega[1],vPQR[2]+Ti2b_omega[2]};
  double JvPQRi[3]; m_vec(J,vPQRi,JvPQRi);
  double cr[3]; v_cross(vPQRi,JvPQRi,cr);
  double mm[3]={moment[0]-cr[0],moment[1]-cr[1],moment[2]-cr[2]};
  m_vec(Jinv,mm,o->vPQRidot);
  double vBodyAccel[3]={force[0]/mass,force[1]/mass,force[2]/mass};
  for(int i=0;i<3;i++) o->vBodyAccel[i]=vBodyAccel[i];
  for(int i=0;i<3;i++) o->vPQRi[i]=vPQRi[i];
  // vUVWidot = Tb2i*vBodyAccel + vGrav
  double Tb2i[3][3]; m_T(Ti2b,Tb2i);
  double tmp[3]; m_vec(Tb2i,vBodyAccel,tmp);
  for(int i=0;i<3;i++) o->vUVWidot[i]=tmp[i]+vGrav[i];
}

// ======== FdmState + step ========
typedef struct {
  double eci_pos[3], eci_vel[3], q[4], vPQRi[3], epa, fuel_total;
  double in_vUVWidot[3], in_vPQRidot[3], vQtrndot[4];
  double deq_q[3][4], deq_pqri[3][3], deq_ivel[3][3], deq_uvwidot[3][3];
  double prev_vBodyAccel[3], prev_vPQRidot[3], prev_vPQRi[3];
  // prev_aux
  double pa_alpha,pa_mach,pa_vc,pa_vg,pa_npy,pa_npz,pa_p,pa_q,pa_r;
  EngState eng;
  FcsState fcs;
} FdmState;

typedef struct {
  double eci_pos[3], eci_vel[3], euler[3], vUVW[3], vPQR[3];
  double alpha,beta,Vt,mach,qbar,thrust,fuel,alt_asl;
} FdmOut;

// deque: shift-right, insert at [0] (jsb_fdm.py deque appendleft+pop 의 [0..2] 재현)
HOSTDEV void deq_push3(double dq[3][3],const double v[3]){
  for(int i=0;i<3;i++){ dq[2][i]=dq[1][i]; dq[1][i]=dq[0][i]; dq[0][i]=v[i]; }
}
HOSTDEV void deq_push4(double dq[3][4],const double v[4]){
  for(int i=0;i<4;i++){ dq[2][i]=dq[1][i]; dq[1][i]=dq[0][i]; dq[0][i]=v[i]; }
}

// kinematics: 현재 ECI state -> vUVW,vPQR,vVel_ned,ecef,Ti2b,Tec2i,radius,mLat,euler
HOSTDEV void kinematics(const FdmState* s, double vUVW[3],double vPQR[3],double vVel_ned[3],
    double ecef[3],double Ti2b[3][3],double Tec2i[3][3],double* radius,double* mLat,double euler[3]){
  double Ti2ec[3][3]; Ti2ec_from_epa(s->epa,Ti2ec);
  m_T(Ti2ec,Tec2i);
  m_vec(Ti2ec,s->eci_pos,ecef);
  double Tl2ec[3][3]; location_derived(ecef,radius,mLat,Tl2ec);
  double Tl2i[3][3]; m_mul(Tec2i,Tl2ec,Tl2i);
  quat_to_T(s->q,Ti2b);
  double Tb2i[3][3]; m_T(Ti2b,Tb2i);
  double Tl2b[3][3]; m_mul(Ti2b,Tl2i,Tl2b);
  double Tb2l[3][3]; m_T(Tl2b,Tb2l);
  double omega[3]={0.0,0.0,ROTATION_RATE};
  double cr[3]; v_cross(omega,s->eci_pos,cr);
  double d[3]={s->eci_vel[0]-cr[0],s->eci_vel[1]-cr[1],s->eci_vel[2]-cr[2]};
  m_vec(Ti2b,d,vUVW);
  double Ti2b_omega[3]; m_vec(Ti2b,omega,Ti2b_omega);
  for(int i=0;i<3;i++) vPQR[i]=s->vPQRi[i]-Ti2b_omega[i];
  m_vec(Tb2l,vUVW,vVel_ned);
  mat_to_euler(Tl2b,euler);
}

// 힘경로 -> Accel(+aux,eng) ; 미분/prev 갱신은 호출측
HOSTDEV void force_path(FdmState* s, FcsState* fcs, EngState* eng, double fuel,
    double c_ail,double c_ele,double c_rud,double c_thr,
    const double vUVW[3],const double vPQR[3],const double vVel_ned[3],
    const double ecef[3],const double Ti2b[3][3],const double Tec2i[3][3],
    double radius,double mLat,const double euler[3],
    Accel* acc, Aux* aux, EngOut* engo, double* alt_asl_out){
  double alt_asl=radius-SEA_LEVEL_RADIUS; *alt_asl_out=alt_asl;
  Atmos at; atm_calculate(alt_asl,&at);
  Mass mb; mb_compute(fuel,&mb);
  FcsOut fo;
  fcs_step(fcs, c_ail,c_ele,c_rud,0.0, 0.0,0.0,
    s->pa_alpha,s->pa_mach,s->pa_vc,s->pa_vg,s->pa_npy,s->pa_npz,s->pa_p,s->pa_q,s->pa_r,
    euler[1],euler[0], &fo);
  auxiliary(vUVW,vPQR,vVel_ned,at.rho,at.a,at.P,mb.cg,
    s->prev_vBodyAccel,s->prev_vPQRidot,s->prev_vPQRi,SLGRAVITY,aux);
  double thr_pos=2.0*c_thr;
  eng_step(eng,thr_pos,aux->mach,at.densalt,at.T,at.sigma,FDM_DT,mb.cg,engo);
  // aero
  double Vt=aux->Vt; double twovel=2.0*Vt;
  double bi2vel=(twovel!=0.0)?30.0/twovel:0.0;
  double ci2vel=(twovel!=0.0)?11.32/twovel:0.0;
  double kclge=KCLGE(AERO_HB);
  double DR,SI,LI,RO,PI_,YA;
  f16_aero(aux->qbar,aux->alpha,aux->beta,aux->mach,bi2vel,ci2vel,kclge,
    aux->p_aero,aux->q_aero,aux->r_aero,
    fo.aileron_pos_rad,fo.elevator_pos_rad,fo.rudder_pos_rad,fo.lef_pos_rad,
    fo.flaperon_mix_rad,fo.speedbrake_pos_rad,fo.gear_pos_norm,
    &DR,&SI,&LI,&RO,&PI_,&YA);
  double drag=-DR, side=SI, lift=-LI;
  double a=aux->alpha,b=aux->beta;
  double ca=cos(a),sa=sin(a),cb=cos(b),sb=sin(b);
  double Fx=ca*cb*drag+(-ca*sb)*side+(-sa)*lift;
  double Fy=sb*drag+cb*side;
  double Fz=sa*cb*drag+(-sa*sb)*side+ca*lift;
  double rx=mb.RPBody[0],ry=mb.RPBody[1],rz=mb.RPBody[2];
  double crossM[3]={ry*Fz-rz*Fy, rz*Fx-rx*Fz, rx*Fy-ry*Fx};
  double force[3]={Fx+engo->forces[0],Fy+engo->forces[1],Fz+engo->forces[2]};
  double moment[3]={RO+crossM[0]+engo->moments[0],PI_+crossM[1]+engo->moments[1],YA+crossM[2]+engo->moments[2]};
  double grav_ecef[3]; gravity_j2(ecef,mLat,grav_ecef);
  double vGrav[3]; m_vec(Tec2i,grav_ecef,vGrav);
  accelerations(force,moment,mb.mass_slug,mb.J,mb.Jinv,vUVW,vPQR,Ti2b,s->eci_pos,vGrav,acc);
}

HOSTDEV void fdm_step(FdmState* s,double aileron,double elevator,double rudder,double throttle,FdmOut* out){
  double DT=FDM_DT;
  // --- Propagate 적분 (순서 준수) ---
  // q (RectEuler + normalize)
  deq_push4(s->deq_q,s->vQtrndot);
  for(int i=0;i<4;i++) s->q[i]=s->q[i]+DT*s->deq_q[0][i];
  quat_normalize(s->q);
  // vPQRi (RectEuler)
  deq_push3(s->deq_pqri,s->in_vPQRidot);
  for(int i=0;i<3;i++) s->vPQRi[i]=s->vPQRi[i]+DT*s->deq_pqri[0][i];
  // eci_pos (AB3) — uses current eci_vel
  deq_push3(s->deq_ivel,s->eci_vel);
  for(int i=0;i<3;i++)
    s->eci_pos[i]=s->eci_pos[i]+(DT/12.0)*(23.0*s->deq_ivel[0][i]-16.0*s->deq_ivel[1][i]+5.0*s->deq_ivel[2][i]);
  // eci_vel (AB2)
  deq_push3(s->deq_uvwidot,s->in_vUVWidot);
  for(int i=0;i<3;i++)
    s->eci_vel[i]=s->eci_vel[i]+DT*(1.5*s->deq_uvwidot[0][i]-0.5*s->deq_uvwidot[1][i]);
  s->epa+=ROTATION_RATE*DT;

  // 운동학
  double vUVW[3],vPQR[3],vVel_ned[3],ecef[3],Ti2b[3][3],Tec2i[3][3],radius,mLat,euler[3];
  kinematics(s,vUVW,vPQR,vVel_ned,ecef,Ti2b,Tec2i,&radius,&mLat,euler);
  // vQtrndot (다음 스텝)
  quat_qdot(s->q,s->vPQRi,s->vQtrndot);

  // 힘경로
  Accel acc; Aux aux; EngOut engo; double alt_asl;
  force_path(s,&s->fcs,&s->eng,s->fuel_total,aileron,elevator,rudder,throttle,
    vUVW,vPQR,vVel_ned,ecef,Ti2b,Tec2i,radius,mLat,euler,&acc,&aux,&engo,&alt_asl);
  s->fuel_total-=engo.fuel_burn;
  for(int i=0;i<3;i++){ s->in_vUVWidot[i]=acc.vUVWidot[i]; s->in_vPQRidot[i]=acc.vPQRidot[i]; }
  for(int i=0;i<3;i++){ s->prev_vBodyAccel[i]=acc.vBodyAccel[i]; s->prev_vPQRidot[i]=acc.vPQRidot[i]; s->prev_vPQRi[i]=acc.vPQRi[i]; }
  s->pa_alpha=aux.alpha; s->pa_mach=aux.mach; s->pa_vc=aux.vc_kts; s->pa_vg=aux.vg;
  s->pa_npy=aux.n_pilot_y; s->pa_npz=aux.n_pilot_z; s->pa_p=aux.p_aero; s->pa_q=aux.q_aero; s->pa_r=aux.r_aero;

  for(int i=0;i<3;i++){ out->eci_pos[i]=s->eci_pos[i]; out->eci_vel[i]=s->eci_vel[i];
    out->euler[i]=euler[i]; out->vUVW[i]=vUVW[i]; out->vPQR[i]=vPQR[i]; }
  out->alpha=aux.alpha; out->beta=aux.beta; out->Vt=aux.Vt; out->mach=aux.mach;
  out->qbar=aux.qbar; out->thrust=engo.thrust; out->fuel=s->fuel_total; out->alt_asl=alt_asl;
}

#endif // FDM_CUH
