#include <hexl/hexl.hpp>
#include <chrono>
#include <cstdio>
#include <vector>
using namespace intel::hexl;
int main(){
  const uint64_t N=32768;
  for(size_t bits: {55u,58u,61u}){
    auto ps=GeneratePrimes(1,bits,true,N);
    uint64_t q=ps[0];
    NTT ntt(N,q);
    AlignedVector64<uint64_t> a(N),b(N),c(N);
    for(size_t i=0;i<N;i++){a[i]=(i*2654435761ull)%q;b[i]=(i*40503ull)%q;}
    int R=50;
    auto t0=std::chrono::steady_clock::now();
    for(int r=0;r<R;r++) ntt.ComputeForward(a.data(),a.data(),1,1);
    auto t1=std::chrono::steady_clock::now();
    for(int r=0;r<R;r++) ntt.ComputeInverse(a.data(),a.data(),1,1);
    auto t2=std::chrono::steady_clock::now();
    for(int r=0;r<R;r++) EltwiseMultMod(c.data(),a.data(),b.data(),N,q,1);
    auto t3=std::chrono::steady_clock::now();
    for(int r=0;r<R;r++) EltwiseFMAMod(c.data(),a.data(),12345,b.data(),N,q,1);
    auto t4=std::chrono::steady_clock::now();
    for(int r=0;r<R;r++) EltwiseAddMod(c.data(),a.data(),b.data(),N,q);
    auto t5=std::chrono::steady_clock::now();
    auto us=[&](auto x,auto y){return std::chrono::duration<double,std::micro>(y-x).count()/R;};
    printf("bits=%zu q=%lu fwd=%.1fus inv=%.1fus mult=%.1fus fma=%.1fus add=%.1fus\n",bits,q,us(t0,t1),us(t1,t2),us(t2,t3),us(t3,t4),us(t4,t5));
  }
}
