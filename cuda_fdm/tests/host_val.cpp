// 호스트 검증: seed.bin -> FdmState, actions.bin, N step, out.bin (N x 10) 작성.
// g++ -O2 -I../gen host_val.cpp -o host_val   (math.h, double)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "fdm.cuh"

static_assert(sizeof(FdmState) == 101*sizeof(double), "FdmState layout must be 101 packed doubles");

int main(int argc, char** argv){
  const char* dir = (argc>1)? argv[1] : "_bin";
  char path[512];
  // meta
  snprintf(path,sizeof(path),"%s/meta.txt",dir);
  FILE* fm=fopen(path,"r"); if(!fm){ fprintf(stderr,"no meta\n"); return 1; }
  int N=0; if(fscanf(fm,"%d",&N)!=1){ return 1; } fclose(fm);
  // seed
  FdmState s;
  snprintf(path,sizeof(path),"%s/seed.bin",dir);
  FILE* fs=fopen(path,"rb"); if(!fs){ fprintf(stderr,"no seed\n"); return 1; }
  double buf[101];
  if(fread(buf,sizeof(double),101,fs)!=101){ fprintf(stderr,"seed read\n"); return 1; }
  fclose(fs);
  // buf -> FdmState (memcpy: 동일 레이아웃 101 double)
  memcpy(&s, buf, sizeof(double)*101);
  // actions
  snprintf(path,sizeof(path),"%s/actions.bin",dir);
  FILE* fa=fopen(path,"rb"); if(!fa){ fprintf(stderr,"no actions\n"); return 1; }
  std::vector<double> act(N*4);
  if((int)fread(act.data(),sizeof(double),N*4,fa)!=N*4){ fprintf(stderr,"act read\n"); return 1; }
  fclose(fa);
  // run
  snprintf(path,sizeof(path),"%s/out_c.bin",dir);
  FILE* fo=fopen(path,"wb"); if(!fo){ return 1; }
  for(int k=0;k<N;k++){
    FdmOut o;
    fdm_step(&s, act[k*4+0],act[k*4+1],act[k*4+2],act[k*4+3], &o);
    double row[10]={o.eci_pos[0],o.eci_pos[1],o.eci_pos[2],
                    o.euler[0],o.euler[1],o.euler[2],
                    o.vUVW[0],o.vUVW[1],o.vUVW[2],o.alpha};
    fwrite(row,sizeof(double),10,fo);
  }
  fclose(fo);
  printf("host_val: ran %d steps -> out_c.bin\n", N);
  return 0;
}
