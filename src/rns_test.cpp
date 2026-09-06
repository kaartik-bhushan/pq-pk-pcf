// Dumps random RNS values and the results of Rescaler / Lifter / poly_mod_small
// so that check_rns.py can verify them against Python big integers.
#include "pcf.hpp"
using namespace pcf;
int main() {
  Params P; P.N = 4096; P.n = 64;
  Context C(P);
  Rng rng;
  FILE* f = fopen("rns_dump.txt", "w");
  fprintf(f, "primes");
  for (u64 q : C.Bg.q) fprintf(f, " %lu", q);
  fprintf(f, "\nkb %zu kg %zu alpha %lu\n", C.kb, C.kg, C.P.alpha);
  const size_t J = 64;  // coefficients to dump
  // ---- rescale
  RNSPoly x(C.kg, C.N); sample_uniform(x, C.Bg, rng);
  RNSPoly xin = x;
  C.resc_g.apply(x);
  for (size_t j = 0; j < J; j++) {
    fprintf(f, "rescale");
    for (size_t l = 0; l < C.kg; l++) fprintf(f, " %lu", xin.limb(l)[j]);
    for (size_t l = 0; l < C.kb; l++) fprintf(f, " %lu", x.limb(l)[j]);
    fprintf(f, "\n");
  }
  // ---- lift (input: kb limbs uniform), also mod alpha
  RNSPoly h(C.kg, C.N); sample_uniform(h, C.Bg, rng);
  RNSPoly hin = h;
  C.lift_g.apply(h, h);
  AlignedBuf ma(C.N);
  poly_mod_small(ma.p, hin, C.Bg, C.kb, C.P.alpha, C.scratch);
  for (size_t j = 0; j < J; j++) {
    fprintf(f, "lift");
    for (size_t l = 0; l < C.kb; l++) fprintf(f, " %lu", hin.limb(l)[j]);
    for (size_t l = 0; l < C.kg; l++) fprintf(f, " %lu", h.limb(l)[j]);
    fprintf(f, " %lu\n", ma.p[j]);
  }
  // ---- NTT round trip / slot packing sanity
  std::vector<u64> s(C.N), c(C.N), s2(C.N);
  for (size_t i = 0; i < C.N; i++) s[i] = rng.u64() & 1;
  C.from_slots(c.data(), s.data());
  C.to_slots(s2.data(), c.data());
  size_t bad = 0; for (size_t i = 0; i < C.N; i++) bad += (s[i] != s2[i]);
  printf("slot round trip: %s\n", bad ? "FAIL" : "ok");
  fclose(f);
  printf("dumped rns_dump.txt\n");
  return bad ? 1 : 0;
}
