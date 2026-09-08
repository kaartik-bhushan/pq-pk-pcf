// Packed public-key PCF for OT (Construction 5), instantiated with
//   * Construction 2 (succinct half-chosen VOLE with local reconstruction),
//   * Construction 4 (compact lattice-based packed public-key aHMAC:
//     SP-RLWE public samples, KDM-Enc1, KDM-Enc-Pack, InpMemMult, rounded Output),
//   * the XOR5-MAJ7 GAR-wPRF written as an RMS program (Section 8).
//
// Moduli (all limbs except alpha are NTT-friendly primes of `prime_bits` bits,
// q = 1 mod 2^16; alpha = 65537 is itself a limb):
//   alpha  = 65537                      packing modulus (X^N+1 splits into N linear factors)
//   beta   = prod of k_beta primes      memory-share modulus                    (~2^290)
//   gamma  = beta * alpha * Q~          input-share modulus, Q~ = k_Q primes     (~2^770)
//   beta'  = first k_betap beta-primes  modulus of the VOLE output shares        (~2^116)
//   p      = beta' * Q'                 VOLE modulus, Q' = k_Qp primes           (~2^290)
// Because alpha | gamma, floor(gamma/alpha) = beta * Q~ and floor(gamma/beta) =
// alpha * Q~ are exact, and both roundings of the construction,
//   floor(x)_{gamma/beta} = round(x / (alpha Q~))   (InpMemMult, keeps the beta limbs)
//   floor(x)_{gamma/alpha} = round(x / (beta Q~))   (Output, keeps the alpha limb)
// are exact RNS rescalings.  Optionally (k_Qs < k_Q) the KDM-Enc-Pack
// ciphertexts and the Output rounding use the divisor gamma_o = alpha * beta *
// Q_s of gamma (Q_s = first k_Qs primes of Q~), i.e. the same construction over
// a smaller modulus; the default is gamma_o = gamma (the literal construction).
//
// RNS limb layout of a gamma-element: [beta_1..beta_kb | alpha | Q~_1..Q~_kQ].
#pragma once
#include "rns.hpp"
#include "sampling.hpp"
#include <chrono>
#include <cstdio>
#include <functional>
#include <map>

namespace pcf {

struct Params {
  size_t N = 1 << 15;      // ring dimension d (= number of packed OT slots)
  size_t n = 4900;         // wPRF key length (perfect square)
  u64 alpha = 65537;
  size_t k_beta = 5, k_Q = 8, k_betap = 2, k_Qp = 3;
  size_t k_Qs = 0;         // 0 = all of Q~ (literal construction); k < k_Q: Output stage over gamma_o = alpha*beta*Q_s
  size_t prime_bits = 58;  // all RNS primes lie in (2^(prime_bits-1), 2^prime_bits); use 50 on AVX512-IFMA CPUs
  int eta = 21;            // CBD parameter: |e| <= 21 = B_e
  size_t n_rand_limbs = 2; // how many Q~-limbs of the shared PRF value r are randomised (InpMemMult)
  bool eager0 = true;      // precompute party-0 memory shares at KeyDer
  size_t sqrt_n() const { return (size_t)std::lround(std::sqrt((double)n)); }
  // Preset for AVX512-IFMA machines (HEXL uses ~2x faster kernels for moduli < 2^50):
  void use_ifma_preset() { prime_bits = 50; k_beta = 6; k_Q = 9; k_betap = 3; k_Qp = 3; }
};

// ----------------------------------------------------------------- timing
struct Timer {
  std::map<std::string, double> acc;
  std::chrono::steady_clock::time_point t0;
  void start() { t0 = std::chrono::steady_clock::now(); }
  void stop(const char* name) {
    auto t1 = std::chrono::steady_clock::now();
    acc[name] += std::chrono::duration<double, std::milli>(t1 - t0).count();
    t0 = t1;
  }
  void print(const char* title, int div = 1) const {
    printf("  [%s] per-eval breakdown (ms):\n", title);
    double tot = 0;
    for (auto& kv : acc) { printf("    %-24s %8.2f\n", kv.first.c_str(), kv.second / div); tot += kv.second / div; }
    printf("    %-24s %8.2f\n", "TOTAL", tot);
  }
};

// ----------------------------------------------------------------- context
struct Context {
  Params P;
  size_t N, kb, kg, kbp, kp, ka;         // limb counts: beta, gamma, beta', p; ka = index of the alpha limb (= kb)
  std::vector<u64> primes;
  Base Bg;                 // [beta | alpha | Q~]
  Base Bp;                 // [beta' | Q']
  std::vector<size_t> idx_beta, idx_Q, idx_Qs, idx_o, idx_drop_rms, idx_drop_out, idx_all;
  Rescaler resc_g;         // round(x / (alpha Q~))         -> beta limbs      (InpMemMult)
  Rescaler resc_p;         // round(x / Q')                  -> beta' limbs     (VOLE Recon)
  Rescaler resc_o;         // round(x / (beta Q_s))          -> alpha limb      (Output)
  Lifter lift_g;           // beta limbs -> all limbs
  Lifter lift_s;           // beta' limbs -> limb kbp (exact sums of up to 7 input shares fit in kbp+1 limbs)
  Lifter lift_m;           // kbp+1 limbs -> all limbs
  Lifter lift_o;           // beta limbs -> {alpha} u Q_s limbs (Output stage)
  std::vector<u64> Q_mod_g;              // floor(gamma/beta) = alpha Q~ mod q_i
  std::vector<u64> Qp_mod_p;             // Q' mod q_i over Bp
  std::vector<u64> payload_mod_g;        // floor(gamma_o/alpha) = beta Q_s mod q_i   (KDM-Enc-Pack payload scale)
  AlignedBuf scratch;
  FastMod fma;                           // fast reduction mod alpha

  explicit Context(const Params& p) : P(p), fma(p.alpha) {
    N = P.N; kb = P.k_beta; kbp = P.k_betap; kp = kbp + P.k_Qp; ka = kb; kg = kb + 1 + P.k_Q;
    if (kbp + 1 > kb) throw std::runtime_error("k_beta must exceed k_betap");
    size_t kQs = P.k_Qs == 0 ? P.k_Q : P.k_Qs;
    if (kQs > P.k_Q) throw std::runtime_error("k_Qs must not exceed k_Q");
    size_t total = kb + P.k_Q + P.k_Qp;
    primes = hx::GeneratePrimes(total, P.prime_bits - 1, false, N);  // HEXL: largest primes below 2^prime_bits
    for (u64 q : primes) if (q <= (1ull << (P.prime_bits - 1)) || q >= (1ull << P.prime_bits)) throw std::runtime_error("prime size");
    std::vector<u64> beta(primes.begin(), primes.begin() + kb);
    std::vector<u64> Qv(primes.begin() + kb, primes.begin() + kb + P.k_Q);
    std::vector<u64> Qpv(primes.begin() + kb + P.k_Q, primes.end());
    std::vector<u64> g = beta; g.push_back(P.alpha); g.insert(g.end(), Qv.begin(), Qv.end());
    std::vector<u64> pp(beta.begin(), beta.begin() + kbp); pp.insert(pp.end(), Qpv.begin(), Qpv.end());
    Bg.init(N, g);
    Bp.init(N, pp);
    // index sets
    for (size_t i = 0; i < kb; i++) idx_beta.push_back(i);
    for (size_t i = kb + 1; i < kg; i++) idx_Q.push_back(i);
    idx_Qs.assign(idx_Q.begin(), idx_Q.begin() + kQs);
    for (size_t i = 0; i < kg; i++) idx_all.push_back(i);
    idx_drop_rms.push_back(ka); idx_drop_rms.insert(idx_drop_rms.end(), idx_Q.begin(), idx_Q.end());
    idx_drop_out = idx_beta; idx_drop_out.insert(idx_drop_out.end(), idx_Qs.begin(), idx_Qs.end());
    idx_o.push_back(ka); idx_o.insert(idx_o.end(), idx_drop_out.begin(), idx_drop_out.end());
    resc_g.init(Bg, idx_beta, idx_drop_rms);
    resc_o.init(Bg, {ka}, idx_drop_out);
    { std::vector<size_t> keep, drop; for (size_t i = 0; i < kbp; i++) keep.push_back(i); for (size_t i = kbp; i < kp; i++) drop.push_back(i); resc_p.init(Bp, keep, drop); }
    lift_g.init(Bg, idx_beta, idx_drop_rms);
    { std::vector<size_t> src; for (size_t i = 0; i < kbp; i++) src.push_back(i); lift_s.init(Bg, src, {kbp}); }
    { std::vector<size_t> src, tgt; for (size_t i = 0; i <= kbp; i++) src.push_back(i); for (size_t i = kbp + 1; i < kg; i++) tgt.push_back(i); lift_m.init(Bg, src, tgt); }
    { std::vector<size_t> tgt = {ka}; tgt.insert(tgt.end(), idx_Qs.begin(), idx_Qs.end()); lift_o.init(Bg, idx_beta, tgt); }
    BigUInt Qg(P.alpha); for (u64 q : Qv) Qg.mul_small(q);                 // gamma / beta
    BigUInt Qp(1); for (u64 q : Qpv) Qp.mul_small(q);
    BigUInt pay(1); for (u64 q : beta) pay.mul_small(q); for (size_t i : idx_Qs) pay.mul_small(Bg.q[i]);   // gamma_o / alpha
    Q_mod_g.resize(kg); for (size_t i = 0; i < kg; i++) Q_mod_g[i] = Qg.mod_small(Bg.q[i]);
    Qp_mod_p.resize(kp); for (size_t i = 0; i < kp; i++) Qp_mod_p[i] = Qp.mod_small(Bp.q[i]);
    payload_mod_g.resize(kg); for (size_t i = 0; i < kg; i++) payload_mod_g[i] = pay.mod_small(Bg.q[i]);
    scratch.alloc(N);
    BigUInt betaB(1); for (u64 q : beta) betaB.mul_small(q);
    BigUInt betapB(1); for (size_t i = 0; i < kbp; i++) betapB.mul_small(beta[i]);
    BigUInt gamma = Bg.product(), gamma_o = pay; gamma_o.mul_small(P.alpha);
    printf("Params: N=%zu n=%zu alpha=%lu  log2(beta)=%zu log2(gamma)=%zu log2(gamma_o)=%zu log2(beta')=%zu log2(p)=%zu  (limbs beta/gamma/beta'/p = %zu/%zu/%zu/%zu; output limbs %zu)\n",
           N, P.n, P.alpha, betaB.bits(), gamma.bits(), gamma_o.bits(), betapB.bits(), Bp.product().bits(), kb, kg, kbp, kp, idx_o.size());
  }

  // NTT-domain polynomial from small signed coefficients, over base B.
  RNSPoly ntt_of_signed(const int64_t* c, const Base& B) const {
    RNSPoly r(B.k(), N); poly_from_signed(r, c, B); poly_ntt(r, B); return r;
  }
  RNSPoly ntt_of_small(const u64* c, const Base& B) const {
    RNSPoly r(B.k(), N); poly_from_small(r, c, B); poly_ntt(r, B); return r;
  }
  // Shared PRF r = PRF(K, id) (Construction 4, InpMemMult): adds uniform values
  // to the first n_rand_limbs Q~-limbs of x (coefficient form, base Bg).
  void add_prf(RNSPoly& x, const AES128& prf, u64 id) {
    const u64 mask = (1ull << P.prime_bits) - 1;
    for (size_t l = 0; l < P.n_rand_limbs; l++) {
      size_t i = idx_Q[l];
      u64* t = scratch.p;
      prf.ctr_fill(reinterpret_cast<uint8_t*>(t), 8 * N, (id << 8) | i, 0);
      for (size_t j = 0; j < N; j++) t[j] &= mask;
      hx::EltwiseReduceMod(t, t, N, Bg.q[i], 2, 1);
      hx::EltwiseAddMod(x.limb(i), x.limb(i), t, N, Bg.q[i]);
    }
  }
  // slots = psi^{-1}(poly mod alpha): forward NTT mod alpha of `coeffs` (values in [0,alpha)).
  void to_slots(u64* slots, const u64* coeffs) const { Bg.ntt[ka]->ComputeForward(slots, coeffs, 1, 1); }
  void from_slots(u64* coeffs, const u64* slots) const { Bg.ntt[ka]->ComputeInverse(coeffs, slots, 1, 1); }
};

// =============================================================== keys
struct PublicParams { uint8_t seed[16]; };

struct PublicKey0 {
  RNSPoly a_p;                  // a in NTT form mod p (from pp)
  std::vector<RNSPoly> A;       // s~_1 : A_j, j in [0, 2 sqrt(n)], NTT mod p
  RNSPoly a_g, u1, u2, u3;      // SP-RLWE public 'a' and KDM-Enc1 outputs, NTT mod gamma
  std::vector<RNSPoly> v1, v2;  // KDM-Enc-Pack outputs (7 each), NTT mod gamma (only the gamma_o limbs are used)
  uint8_t K[16];                // PRF key (only needed for correctness)
};
struct SecretKey0 {
  std::vector<int64_t> s;
  std::vector<std::vector<u64>> delta;    // payloads Delta^(j) in R_alpha (coefficients in [0, alpha))
};
struct PublicKey1 {
  std::vector<RNSPoly> v;       // x~_1 : v_t for t in [sqrt n], NTT mod p
};
struct SecretKey1 {
  std::vector<u64> bits;                  // n * (N/64) words: bit i-th key, slot l
  std::vector<RNSPoly> u0, u1;            // x~_0 noise, NTT mod p, per group t
};

// Evaluation keys (the state each party keeps after KeyDer).
struct EvalKey0 {
  AES128 prf;
  RNSPoly s_beta;               // s mod beta (coefficient form, kb limbs)
  RNSPoly negI1;                // -(u1 * s)  : negated input share of constant 1, NTT mod gamma
  RNSPoly nu1;                  // -u1, NTT mod gamma
  std::vector<RNSPoly> nv1, nv2;      // negated KDM-Enc-Pack ciphertexts, NTT mod gamma
  // memory-share source
  std::vector<RNSPoly> z0;      // eager table: z0^(i) in [0, beta'), coefficient form (kbp limbs)
  std::vector<RNSPoly> W;       // lazy: a^j * s NTT mod p, j in [0, sqrt n)
  std::vector<RNSPoly> v;       // lazy: x~_1
};
struct EvalKey1 {
  AES128 prf;
  std::vector<u64> bits;
  RNSPoly nu1, nu2, nu3;        // negated u's, NTT mod gamma
  std::vector<RNSPoly> nv1, nv2;
  std::vector<RNSPoly> z1;      // eager table: z1^(i) in [0, beta'), coefficient form (kbp limbs)
};

// =============================================================== Setup / KeyGen
inline PublicParams Setup(Rng& rng) { PublicParams pp; rng.fill(pp.seed, 16); return pp; }

inline RNSPoly expand_a(const Context& C, const PublicParams& pp, const Base& B, u64 nonce) {
  Rng r(pp.seed, nonce);
  RNSPoly a(B.k(), C.N);
  sample_uniform(a, B, r);   // uniform in NTT domain == uniform ring element
  return a;
}

// PKPCF.KeyGen(0): samples s <- chi_s and Delta^(j) <- R_alpha, then
// Pack-PK-aHMAC.KeyGen0(s, {Delta}): Share0(s), SP-RLWE-Gen, KDM-Enc1, KDM-Enc-Pack.
inline void KeyGen0(Context& C, const PublicParams& pp, Rng& rng, SecretKey0& sk, PublicKey0& pk) {
  const size_t N = C.N, sn = C.P.sqrt_n();
  sk.s.resize(N); sample_ternary(sk.s.data(), N, rng);
  sk.delta.assign(7, std::vector<u64>(N));
  for (auto& d : sk.delta) sample_uniform_small(d.data(), N, C.P.alpha, rng);

  // ---- SuccHCVOLE.Share0(s) over R_p
  pk.a_p = expand_a(C, pp, C.Bp, 0);
  RNSPoly s_p = C.ntt_of_signed(sk.s.data(), C.Bp);
  RNSPoly apow(C.kp, N);
  for (size_t i = 0; i < C.kp; i++) for (size_t j = 0; j < N; j++) apow.limb(i)[j] = 1;  // a^0
  pk.A.resize(2 * sn + 1);
  std::vector<int64_t> e(N);
  for (size_t j = 0; j <= 2 * sn; j++) {
    RNSPoly Aj(C.kp, N);
    poly_mul(Aj, apow, s_p, C.Bp);                      // a^j s
    sample_cbd(e.data(), N, C.P.eta, rng);
    RNSPoly en = C.ntt_of_signed(e.data(), C.Bp);
    poly_add(Aj, Aj, en, C.Bp);                          // + e'_j
    if (j == sn + 1) {                                    // + floor(p/beta') * s = Q' s
      RNSPoly t(C.kp, N);
      poly_fma_scalar(t, s_p, C.Qp_mod_p, nullptr, C.Bp);
      poly_add(Aj, Aj, t, C.Bp);
    }
    pk.A[j] = std::move(Aj);
    poly_mul(apow, apow, pk.a_p, C.Bp);                  // a^{j+1}
  }

  // ---- SP-RLWE-Gen(s) over R_gamma
  pk.a_g = expand_a(C, pp, C.Bg, 1);
  RNSPoly s_g = C.ntt_of_signed(sk.s.data(), C.Bg);
  RNSPoly b1(C.kg, N), b2(C.kg, N);
  {
    poly_mul(b1, pk.a_g, s_g, C.Bg);
    sample_cbd(e.data(), N, C.P.eta, rng); RNSPoly e1 = C.ntt_of_signed(e.data(), C.Bg); poly_add(b1, b1, e1, C.Bg);
    poly_mul(b2, b1, s_g, C.Bg);
    sample_cbd(e.data(), N, C.P.eta, rng); RNSPoly e2 = C.ntt_of_signed(e.data(), C.Bg); poly_add(b2, b2, e2, C.Bg);
  }
  // ---- KDM-Enc1(s, a, (b1, b2))
  {
    std::vector<int64_t> rr(N); sample_ternary(rr.data(), N, rng);
    RNSPoly r = C.ntt_of_signed(rr.data(), C.Bg);
    pk.u1 = RNSPoly(C.kg, N); pk.u2 = RNSPoly(C.kg, N); pk.u3 = RNSPoly(C.kg, N);
    poly_mul(pk.u1, r, pk.a_g, C.Bg);
    sample_cbd(e.data(), N, C.P.eta, rng); { RNSPoly t = C.ntt_of_signed(e.data(), C.Bg); poly_add(pk.u1, pk.u1, t, C.Bg); }
    poly_mul(pk.u2, r, b1, C.Bg);
    sample_cbd(e.data(), N, C.P.eta, rng); { RNSPoly t = C.ntt_of_signed(e.data(), C.Bg); poly_add(pk.u2, pk.u2, t, C.Bg); }
    poly_mul(pk.u3, r, b2, C.Bg);
    sample_cbd(e.data(), N, C.P.eta, rng); { RNSPoly t = C.ntt_of_signed(e.data(), C.Bg); poly_add(pk.u3, pk.u3, t, C.Bg); }
    RNSPoly t(C.kg, N); poly_fma_scalar(t, s_g, C.Q_mod_g, nullptr, C.Bg); poly_add(pk.u3, pk.u3, t, C.Bg);  // + s * floor(gamma/beta)
  }
  // ---- KDM-Enc-Pack(a, b1, {Delta^(j)}):  theta^(j) <- chi_s,
  //      v1 = theta a + e1,  v2 = theta b1 + e2 + Delta^(j) * floor(gamma_o/alpha)   (mod gamma)
  {
    pk.v1.resize(7); pk.v2.resize(7);
    std::vector<int64_t> th(N);
    for (size_t j = 0; j < 7; j++) {
      sample_ternary(th.data(), N, rng);
      RNSPoly theta = C.ntt_of_signed(th.data(), C.Bg);
      pk.v1[j] = RNSPoly(C.kg, N); pk.v2[j] = RNSPoly(C.kg, N);
      poly_mul(pk.v1[j], theta, pk.a_g, C.Bg);
      sample_cbd(e.data(), N, C.P.eta, rng); { RNSPoly t = C.ntt_of_signed(e.data(), C.Bg); poly_add(pk.v1[j], pk.v1[j], t, C.Bg); }
      poly_mul(pk.v2[j], theta, b1, C.Bg);
      sample_cbd(e.data(), N, C.P.eta, rng); { RNSPoly t = C.ntt_of_signed(e.data(), C.Bg); poly_add(pk.v2[j], pk.v2[j], t, C.Bg); }
      RNSPoly dn = C.ntt_of_small(sk.delta[j].data(), C.Bg);
      RNSPoly t(C.kg, N); poly_fma_scalar(t, dn, C.payload_mod_g, nullptr, C.Bg);
      poly_add(pk.v2[j], pk.v2[j], t, C.Bg);
    }
  }
  rng.fill(pk.K, 16);
}

// PKPCF.KeyGen(1): samples d wPRF keys (bit-packed), computes K~^(i) = psi(bits), runs Share1.
inline void KeyGen1(Context& C, const PublicParams& pp, Rng& rng, SecretKey1& sk, PublicKey1& pk) {
  const size_t N = C.N, n = C.P.n, sn = C.P.sqrt_n(), W = N / 64;
  sk.bits.resize(n * W);
  rng.fill(sk.bits.data(), 8 * sk.bits.size());
  RNSPoly a_p = expand_a(C, pp, C.Bp, 0);
  // powers a^{1+delta}, delta in [1, sn]  -> a^2 .. a^{sn+1}
  std::vector<RNSPoly> apow(sn + 2);
  apow[0] = RNSPoly(C.kp, N); for (size_t i = 0; i < C.kp; i++) for (size_t j = 0; j < N; j++) apow[0].limb(i)[j] = 1;
  for (size_t j = 1; j <= sn + 1; j++) { apow[j] = RNSPoly(C.kp, N); poly_mul(apow[j], apow[j - 1], a_p, C.Bp); }
  sk.u0.resize(sn); sk.u1.resize(sn); pk.v.resize(sn);
  std::vector<int64_t> e(N);
  std::vector<u64> slots(N), coeffs(N);
  RNSPoly tmp(C.kp, N), x_p(C.kp, N);
  for (size_t t = 0; t < sn; t++) {
    sample_cbd(e.data(), N, C.P.eta, rng); sk.u0[t] = C.ntt_of_signed(e.data(), C.Bp);
    sample_cbd(e.data(), N, C.P.eta, rng); sk.u1[t] = C.ntt_of_signed(e.data(), C.Bp);
    RNSPoly v = sk.u0[t];
    poly_mul_acc(v, sk.u1[t], apow[1], tmp, C.Bp);
    for (size_t d = 1; d <= sn; d++) {
      size_t i = t * sn + d - 1;
      for (size_t l = 0; l < N; l++) slots[l] = (sk.bits[i * W + l / 64] >> (l % 64)) & 1;
      C.from_slots(coeffs.data(), slots.data());            // K~^(i) = psi(bits)
      poly_from_small(x_p, coeffs.data(), C.Bp); poly_ntt(x_p, C.Bp);
      poly_mul_acc(v, x_p, apow[1 + d], tmp, C.Bp);
    }
    pk.v[t] = std::move(v);
  }
}

// =============================================================== KeyDer
inline void negate_pack_cts(Context& C, const PublicKey0& pk0, std::vector<RNSPoly>& nv1, std::vector<RNSPoly>& nv2) {
  nv1.resize(7); nv2.resize(7);
  for (size_t j = 0; j < 7; j++) {
    nv1[j] = RNSPoly(C.kg, C.N); poly_neg(nv1[j], pk0.v1[j], C.Bg);
    nv2[j] = RNSPoly(C.kg, C.N); poly_neg(nv2[j], pk0.v2[j], C.Bg);
  }
}

inline void KeyDer0(Context& C, const SecretKey0& sk, const PublicKey0& pk0, const PublicKey1& pk1, EvalKey0& ek) {
  const size_t N = C.N, n = C.P.n, sn = C.P.sqrt_n();
  ek.prf.set_key(pk0.K);
  ek.s_beta = RNSPoly(C.kg, N); poly_from_signed(ek.s_beta, sk.s.data(), C.Bg);
  RNSPoly s_g = C.ntt_of_signed(sk.s.data(), C.Bg);
  ek.nu1 = RNSPoly(C.kg, N); poly_neg(ek.nu1, pk0.u1, C.Bg);
  ek.negI1 = RNSPoly(C.kg, N); poly_mul(ek.negI1, ek.nu1, s_g, C.Bg);
  negate_pack_cts(C, pk0, ek.nv1, ek.nv2);
  // W_j = a^j s (NTT mod p), j in [0, sn)
  RNSPoly s_p = C.ntt_of_signed(sk.s.data(), C.Bp);
  ek.W.resize(sn);
  ek.W[0] = s_p;
  for (size_t j = 1; j < sn; j++) { ek.W[j] = RNSPoly(C.kp, N); poly_mul(ek.W[j], ek.W[j - 1], pk0.a_p, C.Bp); }
  ek.v = pk1.v;
  if (C.P.eager0) {
    ek.z0.resize(n);
    RNSPoly t(C.kp, N);
    for (size_t i = 0; i < n; i++) {
      size_t tt = i / sn, r = i % sn + 1;
      poly_mul(t, ek.v[tt], ek.W[sn - r], C.Bp);     // v_t a^{sn-r} s
      poly_intt(t, C.Bp);
      poly_neg(t, t, C.Bp);                           // -z0
      C.resc_p.apply(t);                              // round(-z0 / Q')
      RNSPoly z(C.kbp, N); memcpy(z.buf.p, t.buf.p, 8 * C.kbp * N);
      ek.z0[i] = std::move(z);
    }
    ek.W.clear(); ek.v.clear();
  }
}

inline void KeyDer1(Context& C, const SecretKey1& sk, const PublicKey1& /*pk1*/, const PublicKey0& pk0, EvalKey1& ek) {
  const size_t N = C.N, n = C.P.n, sn = C.P.sqrt_n(), W = N / 64;
  ek.prf.set_key(pk0.K);
  ek.bits = sk.bits;
  ek.nu1 = RNSPoly(C.kg, N); poly_neg(ek.nu1, pk0.u1, C.Bg);
  ek.nu2 = RNSPoly(C.kg, N); poly_neg(ek.nu2, pk0.u2, C.Bg);
  ek.nu3 = RNSPoly(C.kg, N); poly_neg(ek.nu3, pk0.u3, C.Bg);
  negate_pack_cts(C, pk0, ek.nv1, ek.nv2);
  // eager memory shares z1^(i) = round(-(u~ + <(u0,u1,x_t),(A_{sn-r},...,A_{2sn-r+1})>) / Q')
  ek.z1.resize(n);
  std::vector<RNSPoly> X(sn);
  std::vector<u64> slots(N), coeffs(N);
  std::vector<int64_t> e(N);
  Rng rng;
  RNSPoly acc(C.kp, N), tmp(C.kp, N);
  for (size_t t = 0; t < sn; t++) {
    for (size_t d = 1; d <= sn; d++) {
      size_t i = t * sn + d - 1;
      for (size_t l = 0; l < N; l++) slots[l] = (sk.bits[i * W + l / 64] >> (l % 64)) & 1;
      C.from_slots(coeffs.data(), slots.data());
      X[d - 1] = RNSPoly(C.kp, N); poly_from_small(X[d - 1], coeffs.data(), C.Bp); poly_ntt(X[d - 1], C.Bp);
    }
    for (size_t r = 1; r <= sn; r++) {
      size_t i = t * sn + r - 1;
      poly_mul(acc, sk.u0[t], pk0.A[sn - r], C.Bp);
      poly_mul_acc(acc, sk.u1[t], pk0.A[sn - r + 1], tmp, C.Bp);
      for (size_t d = 1; d <= sn; d++) poly_mul_acc(acc, X[d - 1], pk0.A[sn - r + 1 + d], tmp, C.Bp);
      poly_intt(acc, C.Bp);
      sample_cbd(e.data(), N, C.P.eta, rng);
      poly_from_signed(tmp, e.data(), C.Bp);
      poly_add(acc, acc, tmp, C.Bp);                    // + u~_i
      poly_neg(acc, acc, C.Bp);
      C.resc_p.apply(acc);
      RNSPoly z(C.kbp, N); memcpy(z.buf.p, acc.buf.p, 8 * C.kbp * N);
      ek.z1[i] = std::move(z);
    }
  }
}

// =============================================================== Eval
// Common: derive the 12 wPRF input indices from the 16-byte PCF input x.
inline void derive_indices(const uint8_t x[16], size_t n, size_t idx[12]) {
  AES128 a(x);
  uint32_t w[16];
  a.ctr_fill(reinterpret_cast<uint8_t*>(w), 64, 0xC0FFEEull, 0);
  for (int i = 0; i < 12; i++) idx[i] = w[i] % n;
}

// ---------------------------------------------------------------- RMS program
// XOR5-MAJ7 (both P and its complement) with 9 multiplications.
//   S_X = sum_{i<5} z[x_i],  S_M = sum_{i>=5} z[x_i]
//   X  = prod_{j in {0,2,4}} (S_X-j) (P_XOR),  Xb = prod_{j in {1,3,5}} (S_X-j) (P_notXOR)
//   M  = prod_{j=0..3} (S_M-j) (P_MAJ),        Mb = prod_{j=4..7} (S_M-j) (P_notMAJ)
//   P  = X*Mb + Xb*M (zero iff F=0),           Pbar = X*M + Xb*Mb (zero iff F=1)
// Instead of 4 chains of linear factors we use
//   A = (X+Xb)(M+Mb),  D = (X-Xb)(Mb-M),  P = (A+D)/2,  Pbar = (A-D)/2,
// where  X+Xb = 2S^3 - 15S^2 + 31S - 15,   X-Xb = 3S^2 - 15S + 15   (S = S_X)
//        M+Mb = 2T^4 - 28T^3 + 190T^2 - 644T + 840,  Mb-M = -16T^3 + 168T^2 - 632T + 840  (T = S_M).
// Multiplications: T^2,T^3,T^4 (3), then (M+Mb)*S^k and (Mb-M)*S^k for k=1..3 (6).
// Linear combinations with integer constants (and the constant 1, whose memory
// share is ((0,s),(-1,0))) are free on memory shares; the division by 2 is a
// multiplication of the shares by 2^{-1} mod beta (2P and 2Pbar are exact even
// integers, so Lemma 2 applies unchanged).
struct LinTerm { int cell; int64_t c; };
static const int64_t C_XpXb[4] = {-15, 31, -15, 2};     // coefficients of S^0..S^3
static const int64_t C_XmXb[4] = {15, -15, 3, 0};
static const int64_t C_MpMb[5] = {840, -644, 190, -28, 2};
static const int64_t C_MbmM[5] = {840, -632, 168, -16, 0};
// memory cell indices
enum { CELL_T1 = 0, CELL_T2, CELL_T3, CELL_T4, CELL_MSUM, CELL_MDIFF, CELL_A1, CELL_A2, CELL_A3, CELL_D1, CELL_D2, CELL_D3,
       CELL_A, CELL_D, CELL_P, CELL_PB, CELL_SX, CELL_COUNT };

struct Output0 { std::vector<uint8_t> r0, r1; };                  // r_i^{(0)}, r_i^{(1)}: 16 bytes each, per slot
struct Output1 { std::vector<uint8_t> b; std::vector<uint8_t> r; }; // b_i and r'_i

// scalar c (signed) times optional 2^{-1}, modulo q
inline u64 const_mod(int64_t c, u64 q, bool half) {
  u64 v = c >= 0 ? (u64)c % q : q - ((u64)(-c) % q);
  return half ? mulmod(v, invmod(2, q), q) : v;
}

// ---------------------------------------------------------------- Party 0
struct Evaluator0 {
  Context& C; EvalKey0& ek;
  size_t N, kb, kg, ka;
  Timer T;
  RNSPoly t13;
  std::vector<RNSPoly> mem;          // memory shares: z (kb valid limbs inside kg-limb buffers)
  RNSPoly IX, IM;                    // negated input shares of S_X, S_M (NTT gamma)
  AlignedBuf ys;
  RNSPoly to, uo;                    // output-stage scratch
  RNSPoly zX, zM, zi;
  std::vector<u64> dbg_ys[2], dbg_y[2];   // copies of the output slot values (tests only)
  bool keep_debug = true;
  Evaluator0(Context& c, EvalKey0& e) : C(c), ek(e), N(c.N), kb(c.kb), kg(c.kg), ka(c.ka), t13(c.kg, c.N), IX(c.kg, c.N), IM(c.kg, c.N),
      to(c.kg, c.N), uo(c.kg, c.N), zX(c.kg, c.N), zM(c.kg, c.N), zi(c.kg, c.N) {
    mem.resize(CELL_COUNT); for (auto& m : mem) m = RNSPoly(kg, N);
    ys.alloc(8 * N);
  }
  // memory share z0^(i) (integer in [0,beta')) lifted exactly to limbs [0, kbp] of dst
  void memshare(size_t i, RNSPoly& dst) {
    if (C.P.eager0) { C.lift_s.apply(ek.z0[i], dst); return; }
    size_t sn = C.P.sqrt_n(), tt = i / sn, r = i % sn + 1;
    RNSPoly t(C.kp, N);
    poly_mul(t, ek.v[tt], ek.W[sn - r], C.Bp); poly_intt(t, C.Bp); poly_neg(t, t, C.Bp); C.resc_p.apply(t);
    C.lift_s.apply(t, dst);
  }
  // From the exact (kbp+1)-limb memory share z of S: z5 = z over the kb beta-limbs
  // (memory form) and out = -(u1 * z) (negated input share, NTT mod gamma).
  void input_share_from_mem(const RNSPoly& z, RNSPoly& z5, RNSPoly& out) {
    C.lift_m.apply(z, t13);
    memcpy(z5.buf.p, t13.buf.p, 8 * kb * N);
    poly_ntt(t13, C.Bg);
    poly_mul(out, ek.nu1, t13, C.Bg);
  }
  // Mul(I_f, M_g):  z_out = round((-I * z_in + r) / (gamma/beta))
  void mul(int id, const RNSPoly& negI, const RNSPoly& min, RNSPoly& mout) {
    T.start();
    memcpy(t13.buf.p, min.buf.p, 8 * kb * N);
    C.lift_g.apply(t13, t13);                    T.stop("lift");
    poly_ntt(t13, C.Bg);                          T.stop("ntt");
    poly_mul(t13, t13, negI, C.Bg);               T.stop("pointwise");
    poly_intt(t13, C.Bg);                         T.stop("ntt");
    C.add_prf(t13, ek.prf, id);                   T.stop("prf");
    C.resc_g.apply(t13);                          T.stop("rescale");
    memcpy(mout.buf.p, t13.buf.p, 8 * kb * N);
  }
  // out = sum c_k * mem[cell_k] + c_const * 1   (times 2^{-1} if half), on memory shares mod beta
  void lincomb(RNSPoly& out, const std::vector<LinTerm>& terms, int64_t c_const, bool half) {
    for (size_t l = 0; l < kb; l++) {
      u64 q = C.Bg.q[l];
      u64* o = out.limb(l);
      bool first = true;
      for (auto& t : terms) {
        if (t.c == 0) continue;
        hx::EltwiseFMAMod(o, mem[t.cell].limb(l), const_mod(t.c, q, half), first ? nullptr : o, N, q, 1);
        first = false;
      }
      if (c_const) hx::EltwiseFMAMod(o, ek.s_beta.limb(l), const_mod(c_const, q, half), first ? nullptr : o, N, q, 1);
    }
  }
  void eval(const uint8_t x[16], Output0& out) {
    T.start();
    size_t idx[12]; derive_indices(x, C.P.n, idx);
    zX.zero(); zM.zero();
    for (int i = 0; i < 12; i++) {
      memshare(idx[i], zi);
      RNSPoly& z = (i < 5) ? zX : zM;
      for (size_t l = 0; l <= C.kbp; l++) hx::EltwiseAddMod(z.limb(l), z.limb(l), zi.limb(l), N, C.Bg.q[l]);
    }
    T.stop("memshares");
    input_share_from_mem(zX, mem[CELL_SX], IX);
    input_share_from_mem(zM, mem[CELL_T1], IM);
    T.stop("inputshares");
    // powers of S_M
    mul(0, IM, mem[CELL_T1], mem[CELL_T2]);
    mul(1, IM, mem[CELL_T2], mem[CELL_T3]);
    mul(2, IM, mem[CELL_T3], mem[CELL_T4]);
    T.start();
    lincomb(mem[CELL_MSUM], {{CELL_T1, C_MpMb[1]}, {CELL_T2, C_MpMb[2]}, {CELL_T3, C_MpMb[3]}, {CELL_T4, C_MpMb[4]}}, C_MpMb[0], false);
    lincomb(mem[CELL_MDIFF], {{CELL_T1, C_MbmM[1]}, {CELL_T2, C_MbmM[2]}, {CELL_T3, C_MbmM[3]}}, C_MbmM[0], false);
    T.stop("lincomb");
    mul(3, IX, mem[CELL_MSUM], mem[CELL_A1]);
    mul(4, IX, mem[CELL_A1], mem[CELL_A2]);
    mul(5, IX, mem[CELL_A2], mem[CELL_A3]);
    mul(6, IX, mem[CELL_MDIFF], mem[CELL_D1]);
    mul(7, IX, mem[CELL_D1], mem[CELL_D2]);
    mul(8, IX, mem[CELL_D2], mem[CELL_D3]);
    T.start();
    lincomb(mem[CELL_A], {{CELL_MSUM, C_XpXb[0]}, {CELL_A1, C_XpXb[1]}, {CELL_A2, C_XpXb[2]}, {CELL_A3, C_XpXb[3]}}, 0, false);
    lincomb(mem[CELL_D], {{CELL_MDIFF, C_XmXb[0]}, {CELL_D1, C_XmXb[1]}, {CELL_D2, C_XmXb[2]}}, 0, false);
    lincomb(mem[CELL_P], {{CELL_A, 1}, {CELL_D, 1}}, 0, true);
    lincomb(mem[CELL_PB], {{CELL_A, 1}, {CELL_D, -1}}, 0, true);
    T.stop("lincomb");
    // Output(M_f, alpha):  (M_f1 mod alpha,  Y^(j) = floor(-M_f1 * v1^(j))_{gamma_o/alpha})
    out.r0.resize(16 * N); out.r1.resize(16 * N);
    SlotHash H(x);
    for (int b = 0; b < 2; b++) {
      T.start();
      RNSPoly& M = mem[b == 0 ? CELL_P : CELL_PB];
      C.lift_o.apply(M, to);                          // M over {alpha} u beta u Q_s limbs
      poly_ntt_idx(to, C.Bg, C.idx_o);
      const u64* slots = to.limb(ka);                 // NTT_alpha(M mod alpha) = psi^{-1}(M mod alpha): main output
      T.stop("output-lift");
      for (size_t j = 0; j < 7; j++) {
        poly_mul_idx(uo, to, ek.nv1[j], C.Bg, C.idx_o);   // -M_01 * v1^(j)  (mod gamma_o)
        poly_intt_idx(uo, C.Bg, C.idx_o);
        C.resc_o.apply(uo);                               // round(. / (gamma_o/alpha)) -> alpha limb
        C.to_slots(ys.p + j * N, uo.limb(ka));
      }
      T.stop("output-round");
      if (keep_debug) { dbg_ys[b].assign(ys.p, ys.p + 7 * N); dbg_y[b].assign(slots, slots + N); }
      std::vector<uint8_t>& R = b ? out.r1 : out.r0;
      const u64* yarr[8] = {slots, ys.p, ys.p + N, ys.p + 2 * N, ys.p + 3 * N, ys.p + 4 * N, ys.p + 5 * N, ys.p + 6 * N};
      H.hash_all(yarr, N, R.data());
      T.stop("hash");
    }
  }
};

// ---------------------------------------------------------------- Party 1
struct Evaluator1 {
  Context& C; EvalKey1& ek;
  size_t N, kb, kg, ka;
  Timer T;
  struct Mem { RNSPoly z; RNSPoly g; };   // z: kb valid limbs in kg buffer; g: cleartext value, NTT mod gamma (alpha limb = slots)
  struct Fac { RNSPoly nI0, nI1, f; };    // negated input shares, value (NTT gamma)
  RNSPoly t13, tmp13;
  std::vector<Mem> mem;
  Fac FX, FM;
  AlignedBuf ys;
  std::vector<u64> coeffs, acc, sl;
  RNSPoly to, uo, wo;
  RNSPoly zX, zM, zi, YX, YM, ZX, ZM;
  std::vector<u64> dbg_ys[2], dbg_y[2];
  bool keep_debug = true;
  Evaluator1(Context& c, EvalKey1& e) : C(c), ek(e), N(c.N), kb(c.kb), kg(c.kg), ka(c.ka), t13(c.kg, c.N), tmp13(c.kg, c.N),
      to(c.kg, c.N), uo(c.kg, c.N), wo(c.kg, c.N),
      zX(c.kg, c.N), zM(c.kg, c.N), zi(c.kg, c.N), YX(c.kg, c.N), YM(c.kg, c.N), ZX(c.kg, c.N), ZM(c.kg, c.N) {
    mem.resize(CELL_COUNT); for (auto& m : mem) { m.z = RNSPoly(kg, N); m.g = RNSPoly(kg, N); }
    for (Fac* f : {&FX, &FM}) { f->nI0 = RNSPoly(kg, N); f->nI1 = RNSPoly(kg, N); f->f = RNSPoly(kg, N); }
    ys.alloc(8 * N); coeffs.resize(N); acc.resize(N); sl.resize(N);
  }
  inline u64 bit(size_t i, size_t l) const { return (ek.bits[i * (N / 64) + l / 64] >> (l % 64)) & 1; }
  // Mul(I_f, M_g):  z_out = round((-z_in*I0 - g*I1 + r)/(gamma/beta)),  g_out = f*g
  void mul(int id, const Fac& f, const Mem& min, Mem& mout) {
    T.start();
    memcpy(t13.buf.p, min.z.buf.p, 8 * kb * N);
    C.lift_g.apply(t13, t13);                     T.stop("lift");
    poly_ntt(t13, C.Bg);                          T.stop("ntt");
    poly_mul(t13, t13, f.nI0, C.Bg);              // -M1 * I0
    poly_mul(tmp13, min.g, f.nI1, C.Bg);          // -g * I1
    poly_add(t13, t13, tmp13, C.Bg);
    poly_mul(mout.g, f.f, min.g, C.Bg);           // value update g' = f g
    T.stop("pointwise");
    poly_intt(t13, C.Bg);                         T.stop("ntt");
    C.add_prf(t13, ek.prf, id);                   T.stop("prf");
    C.resc_g.apply(t13);                          T.stop("rescale");
    memcpy(mout.z.buf.p, t13.buf.p, 8 * kb * N);
  }
  // out = sum c_k mem[cell_k] + c_const (times 2^{-1} if half): shares z (mod beta) and values g (NTT gamma).
  // g_only_out: the value is only needed on the output limbs (idx_o).
  void lincomb(Mem& out, const std::vector<LinTerm>& terms, int64_t c_const, bool half, bool g_only_out = false) {
    std::vector<char> need(kg, !g_only_out);
    if (g_only_out) for (size_t i : C.idx_o) need[i] = 1;
    for (size_t l = 0; l < kg; l++) {
      u64 q = C.Bg.q[l];
      bool first = true;
      for (auto& t : terms) {
        if (t.c == 0) continue;
        u64 cq = const_mod(t.c, q, half);
        if (l < kb) hx::EltwiseFMAMod(out.z.limb(l), mem[t.cell].z.limb(l), cq, first ? nullptr : out.z.limb(l), N, q, 1);
        if (need[l]) hx::EltwiseFMAMod(out.g.limb(l), mem[t.cell].g.limb(l), cq, first ? nullptr : out.g.limb(l), N, q, 1);
        first = false;
      }
      if (c_const && need[l]) hx::EltwiseAddMod(out.g.limb(l), out.g.limb(l), const_mod(c_const, q, half), N, q);   // constant: z-share is 0
    }
  }
  void eval(const uint8_t x[16], Output1& out) {
    T.start();
    size_t idx[12]; derive_indices(x, C.P.n, idx);
    // wPRF bits b_i, evaluated directly on the packed key bits
    out.b.assign(N, 0);
    for (size_t l = 0; l < N; l++) {
      u64 xr = 0, mj = 0;
      for (int i = 0; i < 5; i++) xr ^= bit(idx[i], l);
      for (int i = 5; i < 12; i++) mj += bit(idx[i], l);
      out.b[l] = (uint8_t)(xr ^ (mj >= 4));
    }
    // memory shares z of S_X, S_M (exact integer sums, kbp+1 limbs)
    zX.zero(); zM.zero();
    for (int i = 0; i < 12; i++) {
      C.lift_s.apply(ek.z1[idx[i]], zi);
      RNSPoly& z = (i < 5) ? zX : zM;
      for (size_t l = 0; l <= C.kbp; l++) hx::EltwiseAddMod(z.limb(l), z.limb(l), zi.limb(l), N, C.Bg.q[l]);
    }
    T.stop("memshares");
    // values y_S = sum of K~^(x_i) as small integer polynomials (coefficients < 7 alpha), NTT mod gamma
    {
      std::fill(acc.begin(), acc.end(), 0);
      for (int i = 0; i < 12; i++) {
        if (i == 5) { poly_from_small(YX, acc.data(), C.Bg); std::fill(acc.begin(), acc.end(), 0); }
        for (size_t l = 0; l < N; l++) sl[l] = bit(idx[i], l);
        C.from_slots(coeffs.data(), sl.data());
        for (size_t l = 0; l < N; l++) acc[l] += coeffs[l];
      }
      poly_from_small(YM, acc.data(), C.Bg);
      poly_ntt(YX, C.Bg); poly_ntt(YM, C.Bg);
    }
    // Z = lift(z) NTT ; memory form of S_M (and S_X, unused) over beta
    C.lift_m.apply(zX, ZX); memcpy(mem[CELL_SX].z.buf.p, ZX.buf.p, 8 * kb * N); poly_ntt(ZX, C.Bg);
    C.lift_m.apply(zM, ZM); memcpy(mem[CELL_T1].z.buf.p, ZM.buf.p, 8 * kb * N); poly_ntt(ZM, C.Bg);
    // input shares: -I0(S) = nu2*Y + nu1*Z ; -I1(S) = nu3*Y + nu2*Z ; value f = Y
    auto make_fac = [&](Fac& F, const RNSPoly& Y, const RNSPoly& Z) {
      poly_mul(F.nI0, ek.nu2, Y, C.Bg); poly_mul_acc(F.nI0, ek.nu1, Z, tmp13, C.Bg);
      poly_mul(F.nI1, ek.nu3, Y, C.Bg); poly_mul_acc(F.nI1, ek.nu2, Z, tmp13, C.Bg);
      memcpy(F.f.buf.p, Y.buf.p, 8 * kg * N);
    };
    make_fac(FX, YX, ZX);
    make_fac(FM, YM, ZM);
    memcpy(mem[CELL_T1].g.buf.p, YM.buf.p, 8 * kg * N);
    memcpy(mem[CELL_SX].g.buf.p, YX.buf.p, 8 * kg * N);
    T.stop("inputshares");
    // RMS program
    mul(0, FM, mem[CELL_T1], mem[CELL_T2]);
    mul(1, FM, mem[CELL_T2], mem[CELL_T3]);
    mul(2, FM, mem[CELL_T3], mem[CELL_T4]);
    T.start();
    lincomb(mem[CELL_MSUM], {{CELL_T1, C_MpMb[1]}, {CELL_T2, C_MpMb[2]}, {CELL_T3, C_MpMb[3]}, {CELL_T4, C_MpMb[4]}}, C_MpMb[0], false);
    lincomb(mem[CELL_MDIFF], {{CELL_T1, C_MbmM[1]}, {CELL_T2, C_MbmM[2]}, {CELL_T3, C_MbmM[3]}}, C_MbmM[0], false);
    T.stop("lincomb");
    mul(3, FX, mem[CELL_MSUM], mem[CELL_A1]);
    mul(4, FX, mem[CELL_A1], mem[CELL_A2]);
    mul(5, FX, mem[CELL_A2], mem[CELL_A3]);
    mul(6, FX, mem[CELL_MDIFF], mem[CELL_D1]);
    mul(7, FX, mem[CELL_D1], mem[CELL_D2]);
    mul(8, FX, mem[CELL_D2], mem[CELL_D3]);
    T.start();
    lincomb(mem[CELL_A], {{CELL_MSUM, C_XpXb[0]}, {CELL_A1, C_XpXb[1]}, {CELL_A2, C_XpXb[2]}, {CELL_A3, C_XpXb[3]}}, 0, false, true);
    lincomb(mem[CELL_D], {{CELL_MDIFF, C_XmXb[0]}, {CELL_D1, C_XmXb[1]}, {CELL_D2, C_XmXb[2]}}, 0, false, true);
    lincomb(mem[CELL_P], {{CELL_A, 1}, {CELL_D, 1}}, 0, true, true);
    lincomb(mem[CELL_PB], {{CELL_A, 1}, {CELL_D, -1}}, 0, true, true);
    T.stop("lincomb");
    // Output(M_y, alpha):  (M_y1 mod alpha,  Y^(j) = floor(M_y0 v2^(j) - M_y1 v1^(j))_{gamma_o/alpha}),  M_y0 = -g
    out.r.resize(16 * N);
    SlotHash H(x);
    for (int b = 0; b < 2; b++) {
      T.start();
      Mem& M = mem[b == 0 ? CELL_P : CELL_PB];
      C.lift_o.apply(M.z, to);
      poly_ntt_idx(to, C.Bg, C.idx_o);
      const u64* slots = to.limb(ka);                 // main output slots
      T.stop("output-lift");
      for (size_t j = 0; j < 7; j++) {
        poly_mul_idx(uo, to, ek.nv1[j], C.Bg, C.idx_o);   // -M_11 * v1^(j)
        poly_mul_idx(wo, M.g, ek.nv2[j], C.Bg, C.idx_o);  // -g * v2^(j)  (= M_10 v2)
        for (size_t i : C.idx_o) hx::EltwiseAddMod(uo.limb(i), uo.limb(i), wo.limb(i), N, C.Bg.q[i]);
        poly_intt_idx(uo, C.Bg, C.idx_o);
        C.resc_o.apply(uo);
        C.to_slots(ys.p + j * N, uo.limb(ka));
      }
      T.stop("output-round");
      if (keep_debug) { dbg_ys[b].assign(ys.p, ys.p + 7 * N); dbg_y[b].assign(slots, slots + N); }
      const u64* yarr[8] = {slots, ys.p, ys.p + N, ys.p + 2 * N, ys.p + 3 * N, ys.p + 4 * N, ys.p + 5 * N, ys.p + 6 * N};
      H.hash_all(yarr, N, out.r.data(), out.b.data(), (uint8_t)b);
      T.stop("hash");
    }
  }
};

}  // namespace pcf
