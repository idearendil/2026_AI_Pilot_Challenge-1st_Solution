// -*- c -*-
// claude164r 관측 + my_reward 보상 융합 커널 (fp64).
// cuda_fdm/obs_reward.py(torch, CPU 참조와 비트일치 검증됨)의 정확한 포팅.
// 두 커널: advance_kernel(1 thread/env: push+advance+종료+reward), build_obs_kernel(1 thread/기체).
// FDM 커널과 동일 규약(--fmad=false). NVRTC 단일소스(include 없음).

#define D2R 0.017453292519943295
#define R2D 57.29577951308232
#define PI2 6.283185307179586
#define FT2M 0.3048
#define M2FT 3.28084
#define SEA_LEVEL_RADIUS_FT 20925646.32546
#define ROT 7.292115e-5
#define WA 6378137.0
#define WE2 0.0066943799901411
// obs 상수(my_observation)
#define MAX_SPEED 600.0
#define MAX_RANGE_M 2500.0
#define MAX_CLOSURE 1000.0
#define VSPEED_SCALE 100.0
#define PQR_SCALE 4.0
#define ACCEL_SCALE 150.0

#define AOA_SCALE 30.0
#define SIDESLIP_SCALE 15.0
#define MIN_ALT_M 300.0
#define ALT_DANGER 300.0
#define MAX_ALT_M 15000.0
#define ENERGY_ADV 5000.0
#define PURSUIT_ATA 30.0
#define PURSUIT_RANGE 3000.0
#define FUEL_BURN 8.0e-5
#define FUEL_REF 300.0
#define REL_VEL 600.0
#define GACC 9.80665
#define EPISODE_MAX 200.0
// damage
#define MIN_DMG_R_FT 500.0
#define T1_MAX 3000.0
#define T2_MAX 3500.0
#define T3_MAX 4000.0
#define T1_CONE 1.0
#define T2_CONE 2.0
#define T3_CONE 3.0
#define T2_START 100.0
#define T3_START 150.0

#define CLAMP(x,a,b) ((x)<(a)?(a):((x)>(b)?(b):(x)))

__device__ __forceinline__ void mm(const double A[3][3], const double B[3][3], double C[3][3]) {
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            double s = 0.0;
            for (int k = 0; k < 3; k++) s += A[i][k] * B[k][j];
            C[i][j] = s;
        }
}
__device__ __forceinline__ void mv(const double A[3][3], const double v[3], double o[3]) {
    for (int i = 0; i < 3; i++) o[i] = A[i][0]*v[0] + A[i][1]*v[1] + A[i][2]*v[2];
}
__device__ __forceinline__ void mvT(const double A[3][3], const double v[3], double o[3]) {
    for (int i = 0; i < 3; i++) o[i] = A[0][i]*v[0] + A[1][i]*v[1] + A[2][i]*v[2];
}
__device__ __forceinline__ double vnorm(const double v[3]) {
    return sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

// roll/pitch/yaw[deg] -> R_ned_to_body = Tx@Ty@Tz
__device__ void ned2body(double roll, double pitch, double yaw, double R[3][3]) {
    double r = roll*D2R, p = pitch*D2R, y = yaw*D2R;
    double cr = cos(r), sr = sin(r), cp = cos(p), sp = sin(p), cy = cos(y), sy = sin(y);
    double Tx[3][3] = {{1,0,0},{0,cr,sr},{0,-sr,cr}};
    double Ty[3][3] = {{cp,0,-sp},{0,1,0},{sp,0,cp}};
    double Tz[3][3] = {{cy,sy,0},{-sy,cy,0},{0,0,1}};
    double TyTz[3][3];
    mm(Ty, Tz, TyTz);
    mm(Tx, TyTz, R);
}

__device__ void quatT(double q0, double q1, double q2, double q3, double T[3][3]) {
    double q0q0 = q0*q0, q1q1 = q1*q1, q2q2 = q2*q2, q3q3 = q3*q3;
    double q0q1 = q0*q1, q0q2 = q0*q2, q0q3 = q0*q3;
    double q1q2 = q1*q2, q1q3 = q1*q3, q2q3 = q2*q3;
    T[0][0] = q0q0+q1q1-q2q2-q3q3; T[0][1] = 2.0*(q1q2+q0q3); T[0][2] = 2.0*(q1q3-q0q2);
    T[1][0] = 2.0*(q1q2-q0q3); T[1][1] = q0q0-q1q1+q2q2-q3q3; T[1][2] = 2.0*(q2q3+q0q1);
    T[2][0] = 2.0*(q1q3+q0q2); T[2][1] = 2.0*(q2q3-q0q1); T[2][2] = q0q0-q1q1-q2q2+q3q3;
}

// state[101] -> 대회 9-DOF (N/E/D[m], roll/pitch/yaw[deg], body u/v/w[m/s])
__device__ void kin9(const double* st, double OX, double OY, double OZ,
                     double OSLAT, double OCLAT, double OSLON, double OCLON, double s9[9]) {
    double px = st[0], py = st[1], pz = st[2];
    double vx = st[3], vy = st[4], vz = st[5];
    double q0 = st[6], q1 = st[7], q2 = st[8], q3 = st[9];
    double epa = st[13];
    double ce = cos(epa), se = sin(epa);
    double ex = ce*px + se*py, ey = -se*px + ce*py, ez = pz;
    double radius = sqrt(ex*ex + ey*ey + ez*ez);
    double rxy = sqrt(ex*ex + ey*ey);
    double sinLat = ez/radius, cosLat = rxy/radius, sinLon = ey/rxy, cosLon = ex/rxy;
    double Tec2l[3][3] = {
        {-cosLon*sinLat, -sinLon*sinLat, cosLat},
        {-sinLon, cosLon, 0.0},
        {-cosLon*cosLat, -sinLon*cosLat, -sinLat}};
    double Tl2ec[3][3];
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) Tl2ec[i][j] = Tec2l[j][i];
    double Tec2i[3][3] = {{ce,-se,0},{se,ce,0},{0,0,1}};
    double Tl2i[3][3]; mm(Tec2i, Tl2ec, Tl2i);
    double Ti2b[3][3]; quatT(q0, q1, q2, q3, Ti2b);
    double Tl2b[3][3]; mm(Ti2b, Tl2i, Tl2b);
    double d02 = CLAMP(Tl2b[0][2], -1.0, 1.0);
    double theta = asin(-d02);
    double phi = atan2(Tl2b[1][2], Tl2b[2][2]);
    double psi = atan2(Tl2b[0][1], Tl2b[0][0]);
    if (psi < 0.0) psi += PI2;
    double dv[3] = {vx - (-ROT*py), vy - (ROT*px), vz};
    double vb[3]; mv(Ti2b, dv, vb);
    double alt_asl_m = (radius - SEA_LEVEL_RADIUS_FT) * FT2M;
    // NED 브릿지: ecef(m) -> geodetic lat/lon(Bowring) -> alt_asl 로 재임베드 -> origin NED
    double exm = ex*FT2M, eym = ey*FT2M, ezm = ez*FT2M;
    double b = WA * sqrt(1.0 - WE2);
    double ep2 = (WA*WA - b*b) / (b*b);
    double pxy = sqrt(exm*exm + eym*eym);
    double lon = atan2(eym, exm);
    double th = atan2(ezm*WA, pxy*b);
    double s3 = sin(th)*sin(th)*sin(th), c3 = cos(th)*cos(th)*cos(th);
    double lat = atan2(ezm + ep2*b*s3, pxy - WE2*WA*c3);
    double slat = sin(lat), clat = cos(lat), slon = sin(lon), clon = cos(lon);
    double Nrad = WA / sqrt(1.0 - WE2*slat*slat);
    double hx = (Nrad + alt_asl_m)*clat*clon;
    double hy = (Nrad + alt_asl_m)*clat*slon;
    double hz = (Nrad*(1.0 - WE2) + alt_asl_m)*slat;
    double dx = hx - OX, dy = hy - OY, dz = hz - OZ;
    s9[0] = -OSLAT*OCLON*dx - OSLAT*OSLON*dy + OCLAT*dz;
    s9[1] = -OSLON*dx + OCLON*dy;
    s9[2] = -(OCLAT*OCLON*dx + OCLAT*OSLON*dy + OSLAT*dz);
    s9[3] = phi*R2D; s9[4] = theta*R2D; s9[5] = psi*R2D;
    s9[6] = vb[0]*FT2M; s9[7] = vb[1]*FT2M; s9[8] = vb[2]*FT2M;
}

// 3D ATA(own->tgt), 0~180 부호없음
__device__ double ata_deg(const double so[9], const double st[9]) {
    double p[3] = {st[0]-so[0], st[1]-so[1], st[2]-so[2]};
    double n = vnorm(p);
    if (n > 0.0) { p[0]/=n; p[1]/=n; p[2]/=n; }
    double R[3][3]; ned2body(so[3], so[4], so[5], R);
    double pt[3]; mv(R, p, pt);
    return acos(CLAMP(pt[0], -1.0, 1.0)) * R2D;
}

// 3D aspect(부호있음)
__device__ double aspect_deg(const double so[9], const double st[9]) {
    double R[3][3]; ned2body(st[3], st[4], st[5], R);
    double p[3] = {so[0]-st[0], so[1]-st[1], so[2]-st[2]};
    double n = vnorm(p);
    if (n > 0.0) { p[0]/=n; p[1]/=n; p[2]/=n; }
    double b[3]; mv(R, p, b);
    double pt0 = -b[0], pt1 = -b[1], pt2 = b[2];
    double sign = (pt1 > 0.0) ? 1.0 : ((pt1 < 0.0) ? -1.0 : ((pt2 >= 0.0) ? 1.0 : -1.0));
    return sign * acos(CLAMP(pt0, -1.0, 1.0)) * R2D;
}

__device__ void los_az_el(const double so[9], const double st[9], double* az, double* el) {
    double d[3] = {st[0]-so[0], st[1]-so[1], st[2]-so[2]};
    double n = vnorm(d);
    if (n > 0.0) { d[0]/=n; d[1]/=n; d[2]/=n; }
    double R[3][3]; ned2body(so[3], so[4], so[5], R);
    double db[3]; mv(R, d, db);
    *az = atan2(db[1], db[0]) * R2D;
    *el = -asin(CLAMP(db[2], -1.0, 1.0)) * R2D;
}

__device__ void dir_frame(const double x[3], double R[3][3]) {
    double nx = vnorm(x);
    if (nx < 1e-8) {
        R[0][0]=1;R[0][1]=0;R[0][2]=0; R[1][0]=0;R[1][1]=1;R[1][2]=0; R[2][0]=0;R[2][1]=0;R[2][2]=1;
        return;
    }
    double xn[3] = {x[0]/nx, x[1]/nx, x[2]/nx};
    double refs[3][3] = {{0,0,1},{1,0,0},{0,1,0}};   // down, north, east
    double z[3]; int chosen = 2;
    for (int r = 0; r < 3; r++) {
        double d = refs[r][0]*xn[0] + refs[r][1]*xn[1] + refs[r][2]*xn[2];
        double zz[3] = {refs[r][0]-d*xn[0], refs[r][1]-d*xn[1], refs[r][2]-d*xn[2]};
        double nn = vnorm(zz);
        if (r < 2 && nn < 1e-6) continue;
        z[0]=zz[0]; z[1]=zz[1]; z[2]=zz[2]; chosen = r; break;
    }
    (void)chosen;
    double nz = vnorm(z); z[0]/=nz; z[1]/=nz; z[2]/=nz;
    double y[3] = {z[1]*xn[2]-z[2]*xn[1], z[2]*xn[0]-z[0]*xn[2], z[0]*xn[1]-z[1]*xn[0]};
    double ny = vnorm(y); y[0]/=ny; y[1]/=ny; y[2]/=ny;
    z[0]=xn[1]*y[2]-xn[2]*y[1]; z[1]=xn[2]*y[0]-xn[0]*y[2]; z[2]=xn[0]*y[1]-xn[1]*y[0];
    R[0][0]=xn[0];R[0][1]=xn[1];R[0][2]=xn[2];
    R[1][0]=y[0];R[1][1]=y[1];R[1][2]=y[2];
    R[2][0]=z[0];R[2][1]=z[1];R[2][2]=z[2];
}

__device__ void bank_sincos(const double Rb2n[3][3], const double dir[3], double* s, double* c) {
    double Rf[3][3]; dir_frame(dir, Rf);
    double by[3] = {Rb2n[0][1], Rb2n[1][1], Rb2n[2][1]};
    double yv[3]; mv(Rf, by, yv);
    double mu = atan2(yv[2], yv[1]);
    *s = sin(mu); *c = cos(mu);
}

__device__ void log_so3(const double R[3][3], double dt, double out[3]) {
    double tr = R[0][0] + R[1][1] + R[2][2];
    double cos_t = CLAMP((tr - 1.0)*0.5, -1.0, 1.0);
    double theta = acos(cos_t);
    double ax[3] = {R[2][1]-R[1][2], R[0][2]-R[2][0], R[1][0]-R[0][1]};
    double denom = 2.0*sin(theta);
    double scale = 0.0;
    if (theta >= 1e-8 && fabs(denom) >= 1e-8) scale = theta / denom;
    double inv = 1.0 / (dt > 1e-8 ? dt : 1e-8);
    out[0] = ax[0]*scale*inv; out[1] = ax[1]*scale*inv; out[2] = ax[2]*scale*inv;
}

__device__ __forceinline__ double normz(double x, double lo, double hi) {
    if (hi <= lo) return 0.0;
    double mid = (hi+lo)*0.5, half = (hi-lo)*0.5;
    return (CLAMP(x, lo, hi) - mid) / half;
}

__device__ double damage(double r_ft, double ata_abs, double t) {
    double a = fabs(ata_abs);
    if (r_ft >= MIN_DMG_R_FT && r_ft <= T1_MAX && a < T1_CONE)
        return 1.0*(T1_MAX - r_ft)/(T1_MAX - MIN_DMG_R_FT);
    if (t >= T2_START && r_ft >= MIN_DMG_R_FT && r_ft <= T2_MAX && a < T2_CONE)
        return 0.3*(T2_MAX - r_ft)/(T2_MAX - MIN_DMG_R_FT);
    if (t >= T3_START && r_ft >= MIN_DMG_R_FT && r_ft <= T3_MAX && a < T3_CONE)
        return 0.1*(T3_MAX - r_ft)/(T3_MAX - MIN_DMG_R_FT);
    return 0.0;
}

// 거리/조준 포텐셜(고도항 제거: 고도 관련 shaping 은 전면 삭제). a1=아군→상대 |ATA|,
// a2=상대→아군 |ATA|. 값이 클수록 유리. 경계 500ft·15000ft 에서 연속.
__device__ double shaping_potential(double dist_ft, double a1, double a2) {
    double x;
    if (dist_ft <= 500.0) {
        double base = dist_ft + 14000.0;
        x = base*(90.0-a1)/90.0*2.5 - base*(90.0-a2)/90.0*2.5 + 999500.0;
    } else if (dist_ft <= 15000.0) {
        double base = 15000.0 - dist_ft;
        x = base + base*(90.0-a1)/90.0*2.5 - base*(90.0-a2)/90.0*2.5 + 985000.0;
    } else {
        x = 1000000.0 - dist_ft;
    }
    return x;
}

#define STORE(dst, val) { double _v = (val); \
    if (!isfinite(_v)) _v = (_v > 0.0) ? 10.0 : ((_v < 0.0) ? -10.0 : 0.0); \
    (dst) = (float)_v; }

// 관점 기체 obs 214 를 out 에 기록(50 scalar + 144 vector[48행,가속도 포함] + 20 action-hist). so=own9, st=tgt9.
__device__ void build_obs_one(const double so[9], const double st[9],
                              double hp_o, double hp_t, double fuel_o, double fuel_t,
                              const double pqr_o[3], const double pqr_p[3],
                              const double accel_o[3], const double accel_p[3],
                              double dmg_dealt, double dmg_taken, double t_ac,
                              const double* acth, float* out) {
    double Rnb_o[3][3], Rnb_t[3][3];
    ned2body(so[3], so[4], so[5], Rnb_o);
    ned2body(st[3], st[4], st[5], Rnb_t);
    double ovb[3] = {so[6], so[7], so[8]}, tvb[3] = {st[6], st[7], st[8]};
    double own_vn[3], tgt_vn[3];
    mvT(Rnb_o, ovb, own_vn);
    mvT(Rnb_t, tvb, tgt_vn);
    double rel_vn[3] = {tgt_vn[0]-own_vn[0], tgt_vn[1]-own_vn[1], tgt_vn[2]-own_vn[2]};
    double own_spd = vnorm(ovb), tgt_spd = vnorm(tvb);
    double own_alt = -so[2], tgt_alt = -st[2];
    double delta[3] = {st[0]-so[0], st[1]-so[1], st[2]-so[2]};
    double dist = vnorm(delta);
    double los_u[3] = {0,0,0};
    double closure = 0.0;
    if (dist > 1e-6) {
        los_u[0]=delta[0]/dist; los_u[1]=delta[1]/dist; los_u[2]=delta[2]/dist;
        closure = (own_vn[0]-tgt_vn[0])*los_u[0] + (own_vn[1]-tgt_vn[1])*los_u[1]
                + (own_vn[2]-tgt_vn[2])*los_u[2];
    }
    double ata = ata_deg(so, st);
    double enemy_ata = ata_deg(st, so);
    double aa = aspect_deg(so, st);
    double az, el; los_az_el(so, st, &az, &el);

    double u = ovb[0], v = ovb[1], w = ovb[2];
    double aoa = 0.0, sslip = 0.0;
    if (own_spd >= 1.0) {
        aoa = atan2(w, u) * R2D;
        sslip = atan2(v, sqrt(u*u + w*w)) * R2D;
    }
    double vspeed = -own_vn[2];
    double e_own = own_alt + own_spd*own_spd/(2.0*GACC);
    double e_tgt = tgt_alt + tgt_spd*tgt_spd/(2.0*GACC);
    double e_adv = e_own - e_tgt;

    double cone = (t_ac >= T3_START) ? T3_CONE : ((t_ac >= T2_START) ? T2_CONE : T1_CONE);
    double maxrng_ft = (t_ac >= T3_START) ? T3_MAX : ((t_ac >= T2_START) ? T2_MAX : T1_MAX);
    double aim_sharp = 2.0*exp(-((ata/3.0)*(ata/3.0))) - 1.0;
    double aim_margin = tanh((cone - fabs(ata)) / (cone > 1e-6 ? cone : 1e-6));
    double en_aim_sharp = 2.0*exp(-((enemy_ata/3.0)*(enemy_ata/3.0))) - 1.0;
    double en_aim_margin = tanh((cone - fabs(enemy_ata)) / (cone > 1e-6 ? cone : 1e-6));
    double min_r_m = MIN_DMG_R_FT*FT2M, max_r_m = maxrng_ft*FT2M;
    double span = (max_r_m - min_r_m); if (span < 1e-6) span = 1e-6;
    double rm_near = tanh((dist - min_r_m)/span);
    double rm_far = tanh((max_r_m - dist)/span);

    double ovbank_s, ovbank_c, tvbank_s, tvbank_c;
    double Rb2n_o[3][3], Rb2n_t[3][3];
    for (int i=0;i<3;i++) for (int j=0;j<3;j++){ Rb2n_o[i][j]=Rnb_o[j][i]; Rb2n_t[i][j]=Rnb_t[j][i]; }
    bank_sincos(Rb2n_o, own_vn, &ovbank_s, &ovbank_c);
    bank_sincos(Rb2n_t, tgt_vn, &tvbank_s, &tvbank_c);

    double pf_ata = 1.0 - fabs(ata)/PURSUIT_ATA; if (pf_ata < 0.0) pf_ata = 0.0;
    double pf_rng = 1.0 - dist/PURSUIT_RANGE; if (pf_rng < 0.0) pf_rng = 0.0;
    double pursuit = 2.0*(pf_ata*pf_rng) - 1.0;

    // ── 스칼라 50 ──
    STORE(out[0], normz(own_spd, 0.0, MAX_SPEED));
    STORE(out[1], normz(tgt_spd, 0.0, MAX_SPEED));
    STORE(out[2], tanh(aoa/AOA_SCALE));
    STORE(out[3], tanh(sslip/SIDESLIP_SCALE));
    STORE(out[4], tanh((own_alt - MIN_ALT_M)/ALT_DANGER));
    STORE(out[5], normz(vspeed, -VSPEED_SCALE, VSPEED_SCALE));
    STORE(out[6], normz(hp_o, 0.0, 1.0));
    STORE(out[7], normz(hp_t, 0.0, 1.0));
    STORE(out[8], hp_o - hp_t);
    STORE(out[9], e_adv/(fabs(e_adv) + ENERGY_ADV));
    STORE(out[10], normz(dist, 0.0, MAX_RANGE_M));
    STORE(out[11], normz(closure, -MAX_CLOSURE, MAX_CLOSURE));
    STORE(out[12], sin(ata*D2R)); STORE(out[13], cos(ata*D2R));
    STORE(out[14], sin(aa*D2R));  STORE(out[15], cos(aa*D2R));
    STORE(out[16], sin(az*D2R));  STORE(out[17], cos(az*D2R));
    STORE(out[18], sin(el*D2R));  STORE(out[19], cos(el*D2R));
    STORE(out[20], aim_sharp);
    STORE(out[21], aim_margin);
    STORE(out[22], en_aim_sharp);
    STORE(out[23], en_aim_margin);
    STORE(out[24], rm_near);
    STORE(out[25], rm_far);
    STORE(out[26], normz(t_ac, 0.0, EPISODE_MAX));
    STORE(out[27], sin(so[3]*D2R)); STORE(out[28], cos(so[3]*D2R));
    STORE(out[29], sin(so[4]*D2R)); STORE(out[30], cos(so[4]*D2R));
    STORE(out[31], sin(so[5]*D2R)); STORE(out[32], cos(so[5]*D2R));
    STORE(out[33], sin(st[3]*D2R)); STORE(out[34], cos(st[3]*D2R));
    STORE(out[35], sin(st[4]*D2R)); STORE(out[36], cos(st[4]*D2R));
    STORE(out[37], sin(st[5]*D2R)); STORE(out[38], cos(st[5]*D2R));
    STORE(out[39], ovbank_s); STORE(out[40], ovbank_c);
    STORE(out[41], tvbank_s); STORE(out[42], tvbank_c);
    STORE(out[43], normz(fuel_o, 0.0, 1.0));
    STORE(out[44], normz(fuel_t, 0.0, 1.0));
    STORE(out[45], CLAMP(2.0*dmg_dealt - 1.0, -1.0, 1.0));
    STORE(out[46], CLAMP(2.0*dmg_taken - 1.0, -1.0, 1.0));
    STORE(out[47], pursuit);
    STORE(out[48], normz(own_alt, 0.0, MAX_ALT_M));
    STORE(out[49], normz(tgt_alt, 0.0, MAX_ALT_M));

    // ── 벡터 114 (VEC_LAYOUT 순서) ──
    double own_om_n[3], tgt_om_n[3];
    mvT(Rnb_o, pqr_o, own_om_n);
    mvT(Rnb_t, pqr_p, tgt_om_n);
    double I3[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
    double Fmy[3][3], Fopp[3][3], Flos[3][3];
    dir_frame(own_vn, Fmy); dir_frame(tgt_vn, Fopp); dir_frame(delta, Flos);
    // frame ptr 배열
    const double (*F[6])[3] = {I3, Rnb_o, Rnb_t, Fmy, Fopp, Flos};
    double grav[3] = {0,0,1};
    // V[7]=own_accel, V[8]=tgt_accel (이미 NED). kind=3 → accel(normz ±ACCEL_SCALE).
    const double* V[9] = {grav, los_u, own_vn, tgt_vn, rel_vn, own_om_n, tgt_om_n,
                          accel_o, accel_p};
    const int LAY[48][3] = {
        {0,1,0},{0,2,0},{0,3,0},{0,4,0},{0,5,0},
        {1,0,0},{1,1,0},{1,2,0},{1,3,0},{1,4,0},
        {2,0,1},{2,1,1},{2,2,1},{2,4,1},{2,5,1},
        {3,0,1},{3,1,1},{3,2,1},{3,3,1},{3,5,1},
        {4,0,1},{4,1,1},{4,2,1},{4,3,1},{4,4,1},{4,5,1},
        {5,0,2},{5,1,2},{5,2,2},{5,3,2},{5,4,2},{5,5,2},
        {6,0,2},{6,1,2},{6,2,2},{6,3,2},{6,4,2},{6,5,2},
        // own_accel: world,mybody,oppbody,oppvel,los (myvel=3 제외) — 속도 항과 동일 패턴
        {7,0,3},{7,1,3},{7,2,3},{7,4,3},{7,5,3},
        // tgt_accel: world,mybody,oppbody,myvel,los (oppvel=4 제외)
        {8,0,3},{8,1,3},{8,2,3},{8,3,3},{8,5,3}};
    int base = 50;
    for (int i = 0; i < 48; i++) {
        int vi = LAY[i][0], fi = LAY[i][1], kind = LAY[i][2];
        const double (*Rm)[3] = F[fi];
        const double* vv = V[vi];
        double comp[3];
        for (int r = 0; r < 3; r++) comp[r] = Rm[r][0]*vv[0] + Rm[r][1]*vv[1] + Rm[r][2]*vv[2];
        for (int r = 0; r < 3; r++) {
            double val;
            if (kind == 0) val = comp[r];
            else if (kind == 1) val = normz(comp[r], -REL_VEL, REL_VEL);
            else if (kind == 3) val = normz(comp[r], -ACCEL_SCALE, ACCEL_SCALE);
            else val = tanh(comp[r]/PQR_SCALE);
            STORE(out[base + i*3 + r], val);
        }
    }
    // ── action history 20 (벡터블록 48*3=144 뒤, base 50 → 50+144=194) ──
    for (int j = 0; j < 20; j++) STORE(out[194 + j], acth[j]);
}

// ────────────────────────────────────────────────────────────────────────────
extern "C" __global__ void advance_kernel(
    const double* states, const double* actions,
    double* hp, double* fuel, double* t_sec,
    double* prev_att, unsigned char* prev_valid, double* pqr,
    double* last_dmg_dealt, double* last_dmg_taken, double* hp_loss,
    double* act_hist, double* prev_x, unsigned char* prev_x_valid,
    double* prev_vel, double* accel,
    double* reward, unsigned char* term, unsigned char* trunc_,
    int nenv,
    double OX, double OY, double OZ, double OSLAT, double OCLAT, double OSLON, double OCLON,
    double dt, double min_alt, double max_time,
    double own_w, double dmg_scale, double shap_scale,
    double win_r, double loss_r, double own_alt_r, double tgt_alt_r,
    int reward_mode, double alt_hunt_coef) {
    // reward_mode 0 = main(거리/조준 shaping), 1 = exploiter(shaping 제거 + 상대고도 log 사냥).
    // own_alt_r/tgt_alt_r 는 더 이상 쓰이지 않는다(고도이탈 종료 보상 = 남은 HP 전량 상실
    // = ±hp*dmg_scale 로 동적 계산). 인자 순서 안정성 위해 시그니처에는 남겨둔다.
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nenv) return;
    int o = 2*e, p = 2*e + 1;

    // action history push(roll +1, row0=action)
    for (int t = 0; t < 2; t++) {
        int a = (t == 0) ? o : p;
        double* ah = act_hist + a*20;
        for (int k = 4; k >= 1; k--)
            for (int c = 0; c < 4; c++) ah[k*4 + c] = ah[(k-1)*4 + c];
        for (int c = 0; c < 4; c++) ah[c] = actions[a*4 + c];
    }

    double s9o[9], s9p[9];
    kin9(states + o*101, OX, OY, OZ, OSLAT, OCLAT, OSLON, OCLON, s9o);
    kin9(states + p*101, OX, OY, OZ, OSLAT, OCLAT, OSLON, OCLON, s9p);

    double dpos[3] = {s9p[0]-s9o[0], s9p[1]-s9o[1], s9p[2]-s9o[2]};
    double dist = vnorm(dpos);
    double r_ft = dist * M2FT;
    double ata_op = ata_deg(s9o, s9p);
    double ata_po = ata_deg(s9p, s9o);
    double t_old = t_sec[e];
    double rate_o = damage(r_ft, ata_op, t_old);
    double rate_p = damage(r_ft, ata_po, t_old);

    double hp_o_old = hp[o], hp_p_old = hp[p];
    double hp_o_new = hp_o_old - rate_p*dt; if (hp_o_new < 0.0) hp_o_new = 0.0;
    double hp_p_new = hp_p_old - rate_o*dt; if (hp_p_new < 0.0) hp_p_new = 0.0;
    double loss_o = hp_o_old - hp_o_new, loss_p = hp_p_old - hp_p_new;
    hp[o] = hp_o_new; hp[p] = hp_p_new;
    hp_loss[o] = loss_o; hp_loss[p] = loss_p;
    last_dmg_dealt[o] = rate_o; last_dmg_taken[o] = rate_p;
    last_dmg_dealt[p] = rate_p; last_dmg_taken[p] = rate_o;

    double spd_o = sqrt(s9o[6]*s9o[6] + s9o[7]*s9o[7] + s9o[8]*s9o[8]);
    double spd_p = sqrt(s9p[6]*s9p[6] + s9p[7]*s9p[7] + s9p[8]*s9p[8]);
    double fo = fuel[o] - FUEL_BURN*(spd_o/FUEL_REF)*dt; if (fo < 0.0) fo = 0.0; fuel[o] = fo;
    double fp = fuel[p] - FUEL_BURN*(spd_p/FUEL_REF)*dt; if (fp < 0.0) fp = 0.0; fuel[p] = fp;

    // pqr (SO3 log)
    for (int t = 0; t < 2; t++) {
        int a = (t == 0) ? o : p;
        const double* s9a = (t == 0) ? s9o : s9p;
        double Nprev[3][3], Ncurr[3][3];
        ned2body(prev_att[a*3+0], prev_att[a*3+1], prev_att[a*3+2], Nprev);
        ned2body(s9a[3], s9a[4], s9a[5], Ncurr);
        // r_delta = Nprev @ Ncurr^T
        double rd[3][3];
        for (int i=0;i<3;i++) for (int j=0;j<3;j++) {
            double s=0; for (int k=0;k<3;k++) s += Nprev[i][k]*Ncurr[j][k]; rd[i][j]=s;
        }
        double lg[3];
        if (prev_valid[a]) { log_so3(rd, dt, lg); }
        else { lg[0]=lg[1]=lg[2]=0.0; }
        pqr[a*3+0]=lg[0]; pqr[a*3+1]=lg[1]; pqr[a*3+2]=lg[2];
        // 선가속도: v_ned = Rb2n·vbody = mvT(Ncurr, vbody). accel = (v_ned - prev_vel)/dt.
        // prev_valid(=pqr 과 동일 유효성)로 첫 step 0. prev_valid 세팅 전에 읽어야 정합.
        double vb[3] = {s9a[6], s9a[7], s9a[8]};
        double vned[3]; mvT(Ncurr, vb, vned);
        if (prev_valid[a]) {
            double invdt = 1.0 / (dt > 1e-8 ? dt : 1e-8);
            accel[a*3+0] = (vned[0]-prev_vel[a*3+0])*invdt;
            accel[a*3+1] = (vned[1]-prev_vel[a*3+1])*invdt;
            accel[a*3+2] = (vned[2]-prev_vel[a*3+2])*invdt;
        } else { accel[a*3+0]=accel[a*3+1]=accel[a*3+2]=0.0; }
        prev_vel[a*3+0]=vned[0]; prev_vel[a*3+1]=vned[1]; prev_vel[a*3+2]=vned[2];
        prev_att[a*3+0]=s9a[3]; prev_att[a*3+1]=s9a[4]; prev_att[a*3+2]=s9a[5];
        prev_valid[a]=1;
    }

    double t_new = t_old + dt; t_sec[e] = t_new;

    // 종료
    double alt_o = -s9o[2], alt_p = -s9p[2];
    int finite = 1;
    for (int k=0;k<9;k++){ if(!isfinite(s9o[k])||!isfinite(s9p[k])) finite=0; }
    unsigned char te = (alt_o < min_alt) || (alt_p < min_alt)
                    || (hp_o_new <= 0.0) || (hp_p_new <= 0.0) || (!finite);
    unsigned char tr = (!te) && (t_new > max_time);
    term[e] = te; trunc_[e] = tr;

    // reward (관점 o, p)
    double dist_ft = dist / FT2M;
    double a1 = fabs(ata_op), a2 = fabs(ata_po);
    // 이전 step 포텐셜(telescoping) 을 두 관점 갱신 전에 원본으로 읽어둔다.
    double px_o_old = prev_x[o], px_p_old = prev_x[p];
    unsigned char pv_o = prev_x_valid[o], pv_p = prev_x_valid[p];
    // 이번 step 포텐셜. main(0)=거리/조준 shaping, exploiter(1)=상대고도 log(단위·1000 은
    // 차분에서 상쇄되므로 무해; alt<=0 방어로 하한 clamp).
    double pot_o, pot_p;
    if (reward_mode == 0) {
        pot_o = shaping_potential(dist_ft, a1, a2);
        pot_p = shaping_potential(dist_ft, a2, a1);
    } else {
        double aof = alt_o/FT2M/1000.0; if (aof < 1e-4) aof = 1e-4;
        double apf = alt_p/FT2M/1000.0; if (apf < 1e-4) apf = 1e-4;
        pot_o = log(aof); pot_p = log(apf);
    }
    // persp o
    {
        double r_dam = (hp_o_new > 0.0 && hp_p_new > 0.0)
                     ? (loss_p - loss_o*own_w)*dmg_scale : 0.0;
        double r_ex;
        if (reward_mode == 0)
            r_ex = (pv_o && shap_scale != 0.0) ? (pot_o - px_o_old)*shap_scale : 0.0;
        else   // 상대(p) 고도 하강 사냥: C*(ln(상대 이전고도) - ln(상대 현재고도)).
            r_ex = pv_p ? (px_p_old - pot_p)*alt_hunt_coef : 0.0;
        double r_t = 0.0;
        if (te) {
            if (alt_o < min_alt) r_t += -hp_o_new*dmg_scale;      // 내 고도이탈 = 남은 HP 전량 상실
            else if (alt_p < min_alt) r_t += hp_p_new*dmg_scale;  // 상대 고도이탈 = 상대 남은 HP 전량 소멸
            if (hp_p_new <= 0.0) r_t += win_r;
            if (hp_o_new <= 0.0) r_t += loss_r;
        }
        reward[o] = r_dam + r_ex + r_t;
    }
    // persp p
    {
        double r_dam = (hp_o_new > 0.0 && hp_p_new > 0.0)
                     ? (loss_o - loss_p*own_w)*dmg_scale : 0.0;
        double r_ex;
        if (reward_mode == 0)
            r_ex = (pv_p && shap_scale != 0.0) ? (pot_p - px_p_old)*shap_scale : 0.0;
        else
            r_ex = pv_o ? (px_o_old - pot_o)*alt_hunt_coef : 0.0;
        double r_t = 0.0;
        if (te) {
            if (alt_p < min_alt) r_t += -hp_p_new*dmg_scale;
            else if (alt_o < min_alt) r_t += hp_o_new*dmg_scale;
            if (hp_o_new <= 0.0) r_t += win_r;
            if (hp_p_new <= 0.0) r_t += loss_r;
        }
        reward[p] = r_dam + r_ex + r_t;
    }
    // telescoping 포텐셜 갱신(항상 저장 → 위상 정합; shaping 계수와 무관).
    prev_x[o] = pot_o; prev_x[p] = pot_p; prev_x_valid[o] = 1; prev_x_valid[p] = 1;
}

extern "C" __global__ void build_obs_kernel(
    const double* states,
    const double* hp, const double* fuel, const double* t_sec,
    const double* pqr, const double* accel, const double* last_dmg_dealt, const double* last_dmg_taken,
    const double* act_hist, float* obs, int nac,
    double OX, double OY, double OZ, double OSLAT, double OCLAT, double OSLON, double OCLON) {
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= nac) return;
    int par = a ^ 1;
    int e = a >> 1;
    double s9o[9], s9t[9];
    kin9(states + a*101, OX, OY, OZ, OSLAT, OCLAT, OSLON, OCLON, s9o);
    kin9(states + par*101, OX, OY, OZ, OSLAT, OCLAT, OSLON, OCLON, s9t);
    double pqr_o[3] = {pqr[a*3+0], pqr[a*3+1], pqr[a*3+2]};
    double pqr_p[3] = {pqr[par*3+0], pqr[par*3+1], pqr[par*3+2]};
    double acc_o[3] = {accel[a*3+0], accel[a*3+1], accel[a*3+2]};
    double acc_p[3] = {accel[par*3+0], accel[par*3+1], accel[par*3+2]};
    build_obs_one(s9o, s9t, hp[a], hp[par], fuel[a], fuel[par],
                  pqr_o, pqr_p, acc_o, acc_p, last_dmg_dealt[a], last_dmg_taken[a],
                  t_sec[e], act_hist + a*20, obs + a*214);
}
