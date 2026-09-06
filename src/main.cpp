#include "pcf.hpp"
#include <cstring>
#include <string>

using namespace pcf;

static void usage() {
  printf("usage: pcf_bench [--N 32768] [--n 4900] [--evals 3] [--lazy0] [--rand-limbs k] [--no-check] [--ifma] [--prime-bits b] [--kQ k] [--kQs k] [--threads t]\n");
}

// Debug: verify memory-share invariants M01 - M11 = s * g  (mod beta) for every
// memory cell after an evaluation of both parties on the same input.
static bool check_invariants(Context& C, const SecretKey0& sk0, Evaluator0& E0, Evaluator1& E1) {
  size_t N = C.N, kb = C.kb;
  RNSPoly s_b(kb, N);
  poly_from_signed(s_b, sk0.s.data(), C.Bg);
  poly_ntt(s_b, C.Bg);
  bool ok = true;
  for (int k = 0; k < CELL_COUNT; k++) {
    if (k == CELL_A || k == CELL_D || k == CELL_P || k == CELL_PB) continue;   // values kept on Q_s limbs only; covered by check_outputs
    RNSPoly sg(kb, N);
    for (size_t l = 0; l < kb; l++) hx::EltwiseMultMod(sg.limb(l), s_b.limb(l), E1.mem[k].g.limb(l), N, C.Bg.q[l], 1);
    poly_intt(sg, C.Bg);
    for (size_t l = 0; l < kb; l++) {
      const u64 *a = E0.mem[k].limb(l), *b = E1.mem[k].z.limb(l), *c = sg.limb(l);
      u64 q = C.Bg.q[l];
      size_t bad = 0;
      for (size_t j = 0; j < N; j++) if ((a[j] + q - b[j]) % q != c[j]) bad++;
      if (bad) { printf("  invariant FAIL mem[%d] limb %zu: %zu/%zu coefficients\n", k, l, bad, N); ok = false; break; }
    }
  }
  return ok;
}

// Debug: party 1's cleartext value of P (resp. Pbar) must vanish exactly in the
// slots where the wPRF evaluates to 0 (resp. 1).
static bool check_values(Context& C, Evaluator1& E1, const Output1& o1) {
  size_t bad = 0;
  for (size_t i = 0; i < C.N; i++) {
    bool p0 = E1.mem[CELL_P].gs.p[i] == 0, p1 = E1.mem[CELL_PB].gs.p[i] == 0;
    if (p0 == p1 || p0 != (o1.b[i] == 0)) bad++;
  }
  if (bad) printf("  value check FAIL: %zu slots\n", bad);
  return bad == 0;
}

// Debug: output shares satisfy y_0 - y_1 = s_l * P_l and Y_0^(j) - Y_1^(j) = theta^(j)_l * P_l
// (mod alpha) in every slot, for both branches.
static bool check_outputs(Context& C, const SecretKey0& sk0, Evaluator0& E0, Evaluator1& E1) {
  size_t N = C.N; u64 al = C.P.alpha;
  std::vector<u64> co(N), sl(N);
  auto slots_of = [&](const std::vector<int64_t>& v) {
    for (size_t i = 0; i < N; i++) co[i] = v[i] >= 0 ? (u64)v[i] : al - (u64)(-v[i]);
    C.to_slots(sl.data(), co.data());
    return sl;
  };
  size_t bad = 0;
  for (int b = 0; b < 2; b++) {
    const u64* P = E1.mem[b == 0 ? CELL_P : CELL_PB].gs.p;
    std::vector<u64> ss = slots_of(sk0.s);
    size_t b0 = 0, bnz = 0;
    for (size_t i = 0; i < N; i++)
      if ((E0.dbg_y[b][i] + al - E1.dbg_y[b][i]) % al != (ss[i] * P[i]) % al) { bad++; b0++; }
    for (int j = 0; j < 7; j++) {
      std::vector<u64> th = slots_of(sk0.theta[j]);
      size_t bj = 0, off1 = 0;
      for (size_t i = 0; i < N; i++) {
        u64 got = (E0.dbg_ys[b][j * N + i] + al - E1.dbg_ys[b][j * N + i]) % al, want = (th[i] * P[i]) % al;
        if (got != want) { bad++; bj++; if (P[i] != 0) bnz++; if ((got + 1) % al == want || (want + 1) % al == got) off1++; }
      }
      if (bj) printf("   b=%d j=%d: %zu bad (%zu off-by-one)\n", b, j, bj, off1);
    }
    if (b0) printf("   b=%d main: %zu bad\n", b, b0);
  }
  if (bad) printf("  output check FAIL: %zu (slot, output) pairs\n", bad);
  return bad == 0;
}

int main(int argc, char** argv) {
  Params P;
  int evals = 3;
  bool check = true;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&]() { if (i + 1 >= argc) { usage(); exit(1); } return std::string(argv[++i]); };
    if (a == "--N") P.N = std::stoul(next());
    else if (a == "--n") P.n = std::stoul(next());
    else if (a == "--evals") evals = std::stoi(next());
    else if (a == "--lazy0") P.eager0 = false;
    else if (a == "--rand-limbs") P.n_rand_limbs = std::stoul(next());
    else if (a == "--no-check") check = false;
    else if (a == "--ifma") P.use_ifma_preset();
    else if (a == "--threads") thread_count() = std::stoi(next());
    else if (a == "--prime-bits") P.prime_bits = std::stoul(next());
    else if (a == "--kQ") P.k_Q = std::stoul(next());
    else if (a == "--kQs") P.k_Qs = std::stoul(next());
    else { usage(); return 1; }
  }
  size_t sn = P.sqrt_n();
  if (sn * sn != P.n) { printf("n must be a perfect square\n"); return 1; }

  Context C(P);
  Rng rng;
  auto now = []() { return std::chrono::steady_clock::now(); };
  auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };

  PublicParams pp = Setup(rng);
  SecretKey0 sk0; PublicKey0 pk0; SecretKey1 sk1; PublicKey1 pk1;
  auto t0 = now(); KeyGen0(C, pp, rng, sk0, pk0); auto t1 = now();
  printf("KeyGen0: %.1f ms  (pk0: %zu A_j + 3 u + 14 v ring elements)\n", ms(t0, t1), pk0.A.size());
  t0 = now(); KeyGen1(C, pp, rng, sk1, pk1); t1 = now();
  printf("KeyGen1: %.1f ms  (pk1: %zu ring elements)\n", ms(t0, t1), pk1.v.size());
  EvalKey0 ek0; EvalKey1 ek1;
  t0 = now(); KeyDer0(C, sk0, pk0, pk1, ek0); t1 = now();
  printf("KeyDer0: %.1f ms\n", ms(t0, t1));
  t0 = now(); KeyDer1(C, sk1, pk1, pk0, ek1); t1 = now();
  printf("KeyDer1: %.1f ms\n", ms(t0, t1));

  Evaluator0 E0(C, ek0);
  Evaluator1 E1(C, ek1);
  E0.keep_debug = E1.keep_debug = check;
  Output0 o0; Output1 o1;
  double tot0 = 0, tot1 = 0;
  size_t total_ok = 0, total_bad = 0;
  for (int e = 0; e < evals; e++) {
    uint8_t x[16]; rng.fill(x, 16);
    t0 = now(); E0.eval(x, o0); t1 = now(); double d0 = ms(t0, t1);
    auto t2 = now(); E1.eval(x, o1); auto t3 = now(); double d1 = ms(t2, t3);
    tot0 += d0; tot1 += d1;
    // correctness: r'_i == r_i^{(b_i)}; and r_i^{(1-b_i)} != r'_i
    size_t ok = 0, bad = 0, coll = 0, ones = 0;
    for (size_t i = 0; i < C.N; i++) {
      const uint8_t* rb = (o1.b[i] ? o0.r1.data() : o0.r0.data()) + 16 * i;
      const uint8_t* ro = (o1.b[i] ? o0.r0.data() : o0.r1.data()) + 16 * i;
      if (memcmp(rb, o1.r.data() + 16 * i, 16) == 0) ok++; else bad++;
      if (memcmp(ro, o1.r.data() + 16 * i, 16) == 0) {
        coll++;
        if (coll <= 2) {
          printf("  collision at slot %zu (b=%d): P slots %lu %lu | ys0: ", i, o1.b[i], E1.mem[CELL_P].gs.p[i], E1.mem[CELL_PB].gs.p[i]);
          for (int j = 0; j < 7; j++) printf("%lu ", E0.dbg_ys[(1 - o1.b[i])][j * C.N + i]);
          printf("y0=%lu | ys1: ", E0.dbg_y[(1 - o1.b[i])][i]);
          for (int j = 0; j < 7; j++) printf("%lu ", E1.dbg_ys[o1.b[i]][j * C.N + i]);
          printf("y1=%lu\n", E1.dbg_y[o1.b[i]][i]);
        }
      }
      ones += o1.b[i];
    }
    total_ok += ok; total_bad += bad;
    printf("eval %d: party0 %.1f ms, party1 %.1f ms | OTs correct %zu/%zu, other-string collisions %zu, b=1 fraction %.3f\n",
           e, d0, d1, ok, C.N, coll, (double)ones / C.N);
    if (check && e == 0) printf("  memory-share invariants: %s, RMS values: %s, output shares: %s\n", check_invariants(C, sk0, E0, E1) ? "OK" : "FAILED",
                                 check_values(C, E1, o1) ? "OK" : "FAILED", check_outputs(C, sk0, E0, E1) ? "OK" : "FAILED");
  }
  if (check) {  // dump the cleartext coefficients of the degree-7 value (M+Mb)*S_X^3 for a magnitude check
    RNSPoly g = E1.mem[CELL_A3].g; poly_intt(g, C.Bg);
    FILE* f = fopen("value_dump.txt", "w");
    fprintf(f, "primes"); for (u64 q : C.Bg.q) fprintf(f, " %lu", q); fprintf(f, "\n");
    for (size_t j = 0; j < C.N; j += 64) { for (size_t l = 0; l < C.kg; l++) fprintf(f, "%lu ", g.limb(l)[j]); fprintf(f, "\n"); }
    fclose(f);
  }
  E0.T.print("party 0", evals);
  E1.T.print("party 1", evals);
  double per0 = tot0 / evals, per1 = tot1 / evals;
  printf("\nRESULT: N=%zu slots/eval, party0 %.1f ms/eval, party1 %.1f ms/eval\n", C.N, per0, per1);
  printf("        OTs/s (%d thread%s per party, bounded by the slower party): %.0f   [party0 alone: %.0f, party1 alone: %.0f]\n",
         thread_count(), thread_count() > 1 ? "s" : "", C.N / (std::max(per0, per1) / 1000.0), C.N / (per0 / 1000.0), C.N / (per1 / 1000.0));
  printf("        correctness: %zu/%zu OTs consistent\n", total_ok, total_ok + total_bad);
  return total_bad ? 2 : 0;
}
