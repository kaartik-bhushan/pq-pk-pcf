// RNS ring arithmetic over R_q = Z_q[X]/(X^N+1) with q a product of
// NTT-friendly word-sized primes, on top of Intel HEXL kernels.
#pragma once
#include <hexl/hexl.hpp>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <functional>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace pcf {

using u64 = uint64_t;

// Optional multi-threading (OpenMP). g_threads == 1 -> strictly sequential.
inline int& thread_count() { static int t = 1; return t; }
// run fn(i) for i in [0,k), in parallel over i when threads > 1
template <class F> inline void par_for(size_t k, F fn) {
#ifdef _OPENMP
  if (thread_count() > 1 && k > 1) {
#pragma omp parallel for num_threads(std::min<int>(thread_count(), (int)k)) schedule(static)
    for (size_t i = 0; i < k; i++) fn(i);
    return;
  }
#endif
  for (size_t i = 0; i < k; i++) fn(i);
}
// run fn(lo, len) over chunks of [0,N) (chunks are multiples of 64 elements)
template <class F> inline void par_chunks(size_t N, F fn) {
#ifdef _OPENMP
  int T = thread_count();
  if (T > 1 && N >= 64 * (size_t)T) {
    size_t chunk = ((N / T + 63) / 64) * 64;
#pragma omp parallel for num_threads(T) schedule(static)
    for (size_t lo = 0; lo < N; lo += chunk) fn(lo, std::min(chunk, N - lo));
    return;
  }
#endif
  fn(0, N);
}
using u128 = unsigned __int128;
using intel::hexl::NTT;

// ---------------------------------------------------------------- BigUInt
// Minimal unsigned big integer: enough for products of primes, division by
// a word, and reduction modulo a word. Only used for parameter set-up.
struct BigUInt {
  std::vector<u64> w;  // little-endian limbs
  BigUInt() : w(1, 0) {}
  explicit BigUInt(u64 v) : w(1, v) {}
  void trim() { while (w.size() > 1 && w.back() == 0) w.pop_back(); }
  BigUInt& mul_small(u64 m) {
    u128 carry = 0;
    for (auto& x : w) { u128 p = (u128)x * m + carry; x = (u64)p; carry = p >> 64; }
    if (carry) w.push_back((u64)carry);
    return *this;
  }
  BigUInt& add_small(u64 a) {
    u128 carry = a;
    for (auto& x : w) { u128 p = (u128)x + carry; x = (u64)p; carry = p >> 64; if (!carry) break; }
    if (carry) w.push_back((u64)carry);
    return *this;
  }
  // returns remainder, this /= d
  u64 divmod_small(u64 d) {
    u128 rem = 0;
    for (size_t i = w.size(); i-- > 0;) {
      u128 cur = (rem << 64) | w[i];
      w[i] = (u64)(cur / d);
      rem = cur % d;
    }
    trim();
    return (u64)rem;
  }
  u64 mod_small(u64 d) const {
    u128 rem = 0;
    for (size_t i = w.size(); i-- > 0;) rem = ((rem << 64) | w[i]) % d;
    return (u64)rem;
  }
  size_t bits() const {
    size_t b = 64 * (w.size() - 1);
    u64 top = w.back();
    while (top) { b++; top >>= 1; }
    return b;
  }
  bool is_zero() const { return w.size() == 1 && w[0] == 0; }
};

inline u64 mulmod(u64 a, u64 b, u64 q) { return (u64)(((u128)a * b) % q); }
inline u64 powmod(u64 a, u64 e, u64 q) {
  u64 r = 1; a %= q;
  while (e) { if (e & 1) r = mulmod(r, a, q); a = mulmod(a, a, q); e >>= 1; }
  return r;
}
inline u64 invmod(u64 a, u64 q) { return powmod(a % q, q - 2, q); }  // q prime

// Barrett reduction of 64-bit inputs modulo a small t (t < 2^32): one 64x64
// high multiply instead of a division.
struct FastMod {
  u64 t = 1, m = 0;
  FastMod() = default;
  explicit FastMod(u64 t_) : t(t_), m((u64)((((u128)1) << 64) / t_)) {}
  inline u64 operator()(u64 x) const {
    u64 q = (u64)(((u128)x * m) >> 64);
    u64 r = x - q * t;
    return r >= t ? r - t : r;
  }
  inline void reduce(u64* out, const u64* in, size_t N) const { for (size_t i = 0; i < N; i++) out[i] = (*this)(in[i]); }
};

// ------------------------------------------------------------ aligned buf
struct AlignedBuf {
  u64* p = nullptr;
  size_t n = 0;
  AlignedBuf() = default;
  explicit AlignedBuf(size_t n_) { alloc(n_); }
  AlignedBuf(const AlignedBuf& o) { alloc(o.n); if (n) memcpy(p, o.p, 8 * n); }
  AlignedBuf& operator=(const AlignedBuf& o) {
    if (this != &o) { if (n != o.n) { release(); alloc(o.n); } if (n) memcpy(p, o.p, 8 * n); }
    return *this;
  }
  AlignedBuf(AlignedBuf&& o) noexcept : p(o.p), n(o.n) { o.p = nullptr; o.n = 0; }
  AlignedBuf& operator=(AlignedBuf&& o) noexcept {
    if (this != &o) { release(); p = o.p; n = o.n; o.p = nullptr; o.n = 0; }
    return *this;
  }
  ~AlignedBuf() { release(); }
  void alloc(size_t n_) {
    n = n_;
    if (n) {
      size_t bytes = ((8 * n + 63) / 64) * 64;
      p = static_cast<u64*>(std::aligned_alloc(64, bytes));
      if (!p) throw std::bad_alloc();
      memset(p, 0, bytes);
    }
  }
  void release() { if (p) std::free(p); p = nullptr; n = 0; }
  u64& operator[](size_t i) { return p[i]; }
  const u64& operator[](size_t i) const { return p[i]; }
};

// ---------------------------------------------------------------- Base
// An RNS base: ordered list of primes with NTT tables and Garner constants.
struct Base {
  size_t N = 0;
  std::vector<u64> q;                          // primes
  std::vector<std::unique_ptr<NTT>> ntt;       // NTT per prime
  std::vector<u64> garner_inv;                 // (prod_{j<i} q_j)^{-1} mod q_i
  std::vector<std::vector<u64>> qmod;          // qmod[i][j] = q_j mod q_i

  Base() = default;
  Base(size_t N_, const std::vector<u64>& primes) { init(N_, primes); }
  void init(size_t N_, const std::vector<u64>& primes) {
    N = N_; q = primes;
    ntt.clear();
    for (u64 p : q) ntt.emplace_back(std::make_unique<NTT>(N, p));
    size_t k = q.size();
    garner_inv.assign(k, 1);
    qmod.assign(k, std::vector<u64>(k, 0));
    for (size_t i = 0; i < k; i++) {
      u64 prod = 1;
      for (size_t j = 0; j < i; j++) prod = mulmod(prod, q[j] % q[i], q[i]);
      garner_inv[i] = invmod(prod, q[i]);
      for (size_t j = 0; j < k; j++) qmod[i][j] = q[j] % q[i];
    }
  }
  size_t k() const { return q.size(); }
  BigUInt product() const { BigUInt P(1); for (u64 p : q) P.mul_small(p); return P; }
  // Sub-base consisting of primes [lo, hi)
  Base sub(size_t lo, size_t hi) const {
    return Base(N, std::vector<u64>(q.begin() + lo, q.begin() + hi));
  }
};

// ---------------------------------------------------------------- RNSPoly
// k limbs of N coefficients each, stored contiguously limb-major.
struct RNSPoly {
  size_t k = 0, N = 0;
  AlignedBuf buf;
  RNSPoly() = default;
  RNSPoly(size_t k_, size_t N_) : k(k_), N(N_), buf(k_ * N_) {}
  u64* limb(size_t i) { return buf.p + i * N; }
  const u64* limb(size_t i) const { return buf.p + i * N; }
  void zero() { memset(buf.p, 0, 8 * k * N); }
  bool empty() const { return k == 0; }
};

// ------------------------------------------------------ Eltwise helpers
namespace hx = intel::hexl;

inline void poly_ntt(RNSPoly& a, const Base& B) {
  par_for(a.k, [&](size_t i) { B.ntt[i]->ComputeForward(a.limb(i), a.limb(i), 2, 1); });
}
inline void poly_intt(RNSPoly& a, const Base& B) {
  par_for(a.k, [&](size_t i) { B.ntt[i]->ComputeInverse(a.limb(i), a.limb(i), 2, 1); });
}
inline void poly_mul(RNSPoly& r, const RNSPoly& a, const RNSPoly& b, const Base& B) {
  par_for(a.k, [&](size_t i) { hx::EltwiseMultMod(r.limb(i), a.limb(i), b.limb(i), a.N, B.q[i], 1); });
}
inline void poly_add(RNSPoly& r, const RNSPoly& a, const RNSPoly& b, const Base& B) {
  par_for(a.k, [&](size_t i) { hx::EltwiseAddMod(r.limb(i), a.limb(i), b.limb(i), a.N, B.q[i]); });
}
inline void poly_sub(RNSPoly& r, const RNSPoly& a, const RNSPoly& b, const Base& B) {
  par_for(a.k, [&](size_t i) { hx::EltwiseSubMod(r.limb(i), a.limb(i), b.limb(i), a.N, B.q[i]); });
}
// r = a * scalar + b   (b may be nullptr) per limb, scalar given per limb
inline void poly_fma_scalar(RNSPoly& r, const RNSPoly& a, const std::vector<u64>& sc, const RNSPoly* b, const Base& B) {
  par_for(a.k, [&](size_t i) {
    hx::EltwiseFMAMod(r.limb(i), a.limb(i), sc[i] % B.q[i], b ? b->limb(i) : nullptr, a.N, B.q[i], 1); });
}
inline void poly_add_scalar(RNSPoly& r, const RNSPoly& a, const std::vector<u64>& sc, const Base& B) {
  par_for(a.k, [&](size_t i) { hx::EltwiseAddMod(r.limb(i), a.limb(i), sc[i] % B.q[i], a.N, B.q[i]); });
}
inline void poly_neg(RNSPoly& r, const RNSPoly& a, const Base& B) {
  par_for(a.k, [&](size_t i) { hx::EltwiseFMAMod(r.limb(i), a.limb(i), B.q[i] - 1, nullptr, a.N, B.q[i], 1); });
}
// Multiply-accumulate r += a*b (pointwise, NTT domain)
inline void poly_mul_acc(RNSPoly& r, const RNSPoly& a, const RNSPoly& b, RNSPoly& tmp, const Base& B) {
  par_for(a.k, [&](size_t i) {
    hx::EltwiseMultMod(tmp.limb(i), a.limb(i), b.limb(i), a.N, B.q[i], 1);
    hx::EltwiseAddMod(r.limb(i), r.limb(i), tmp.limb(i), a.N, B.q[i]);
  });
}
// chunk-parallel scalar-FMA / sub / reduce on single limbs (used by Garner passes)
inline void fma_c(u64* r, const u64* a, u64 sc, const u64* b, size_t N, u64 q, u64 f) {
  par_chunks(N, [&](size_t lo, size_t len) { hx::EltwiseFMAMod(r + lo, a + lo, sc, b ? b + lo : nullptr, len, q, f); });
}
inline void sub_c(u64* r, const u64* a, const u64* b, size_t N, u64 q) {
  par_chunks(N, [&](size_t lo, size_t len) { hx::EltwiseSubMod(r + lo, a + lo, b + lo, len, q); });
}
inline void add_c(u64* r, const u64* a, const u64* b, size_t N, u64 q) {
  par_chunks(N, [&](size_t lo, size_t len) { hx::EltwiseAddMod(r + lo, a + lo, b + lo, len, q); });
}
inline void adds_c(u64* r, const u64* a, u64 sc, size_t N, u64 q) {
  par_chunks(N, [&](size_t lo, size_t len) { hx::EltwiseAddMod(r + lo, a + lo, sc, len, q); });
}
inline void red_c(u64* r, const u64* a, size_t N, u64 q, u64 fin, u64 fout) {
  par_chunks(N, [&](size_t lo, size_t len) { hx::EltwiseReduceMod(r + lo, a + lo, len, q, fin, fout); });
}
// Embed a small non-negative integer polynomial (coeff < 2^63) into every limb.
inline void poly_from_small(RNSPoly& r, const u64* coeffs, const Base& B) {
  for (size_t i = 0; i < r.k; i++) {
    u64 q = B.q[i];
    u64* d = r.limb(i);
    if (q > (1ull << 40)) { memcpy(d, coeffs, 8 * r.N); continue; }   // callers guarantee coeff < 2^40
    FastMod fm(q);
    fm.reduce(d, coeffs, r.N);
  }
}
// Embed a signed small polynomial (|c| small) into every limb.
inline void poly_from_signed(RNSPoly& r, const int64_t* coeffs, const Base& B) {
  for (size_t i = 0; i < r.k; i++) {
    u64 q = B.q[i];
    u64* d = r.limb(i);
    for (size_t j = 0; j < r.N; j++) { int64_t c = coeffs[j]; d[j] = c >= 0 ? (u64)c % q : q - ((u64)(-c) % q); }
  }
}

// ------------------------------------------------------------- Garner
// In-place mixed-radix (Garner) digits of the value represented by limbs
// [lo, lo+k) of `x` (base B restricted to those primes). After the call,
// limb lo+i holds digit d_i (< q_{lo+i}) with
//   value = sum_i d_i * prod_{j<i} q_{lo+j}.
// Requires the primes of the sub-base to be within a factor 2 of each other
// (all digits < 2 * any prime), which our parameter choice guarantees.
struct GarnerTables {
  // for sub-base [lo,hi) of B: inv[i], qm[i][j] = q_{lo+j} mod q_{lo+i}
  size_t lo = 0, k = 0;
  std::vector<u64> inv;
  std::vector<std::vector<u64>> qm;
  GarnerTables() = default;
  GarnerTables(const Base& B, size_t lo_, size_t hi) : lo(lo_), k(hi - lo_) {
    inv.resize(k); qm.assign(k, std::vector<u64>(k));
    for (size_t i = 0; i < k; i++) {
      u64 qi = B.q[lo + i], prod = 1;
      for (size_t j = 0; j < i; j++) prod = mulmod(prod, B.q[lo + j] % qi, qi);
      inv[i] = invmod(prod, qi);
      for (size_t j = 0; j < k; j++) qm[i][j] = B.q[lo + j] % qi;
    }
  }
};

inline void garner_inplace(RNSPoly& x, const Base& B, const GarnerTables& G, u64* tmp) {
  size_t N = x.N, lo = G.lo;
  for (size_t i = 1; i < G.k; i++) {
    u64 qi = B.q[lo + i];
    // t = Horner(d_{i-1}, ..., d_0) mod qi
    memcpy(tmp, x.limb(lo + i - 1), 8 * N);
    if (i >= 2) {
      for (size_t j = i - 1; j-- > 0;)
        fma_c(tmp, tmp, G.qm[i][j], x.limb(lo + j), N, qi, 2);
    } else {
      red_c(tmp, tmp, N, qi, 2, 1);
    }
    // d_i = (x_i - t) * inv_i
    sub_c(x.limb(lo + i), x.limb(lo + i), tmp, N, qi);
    fma_c(x.limb(lo + i), x.limb(lo + i), G.inv[i], nullptr, N, qi, 1);
  }
}

// Reconstruct value from digits (limbs [lo, lo+k) of `d`, base Bsrc) modulo
// prime t, into `out`.  Horner: v = d_{k-1}; v = v*q_j + d_j.
// If the target prime t is small (t < 2^50), digits are first reduced mod t
// with scalar division (they are ~2^58, far above 8t); `scratch` (N words)
// is then required.
inline void horner_to(u64* out, const RNSPoly& d, const Base& Bsrc, size_t lo, size_t k, u64 t, u64 in_factor = 2,
                      u64* scratch = nullptr) {
  size_t N = d.N;
  bool small = t < (1ull << 40);
  FastMod fm(small ? t : 3);
  auto red = [&](const u64* src) -> const u64* {
    if (!small) return src;
    fm.reduce(scratch, src, N);
    return scratch;
  };
  if (small) { const u64* r = red(d.limb(lo + k - 1)); memcpy(out, r, 8 * N); in_factor = 1; }
  else red_c(out, d.limb(lo + k - 1), N, t, in_factor, 1);
  for (size_t j = k - 1; j-- > 0;) {
    u64 qj = Bsrc.q[lo + j] % t;
    fma_c(out, out, qj, red(d.limb(lo + j)), N, t, in_factor);
  }
}

// ---------------------------------------------------------- Rescaler
// Given base Bfull = [B_keep (k1 primes) | B_drop (k2 primes)] and
// x in coefficient form over Bfull (integer representative in [0, prod)),
// compute h = floor((x + D/2)/D) mod prod(B_keep) = round(x / D), where
// D = prod(B_drop).  Exact.  Output over B_keep (first k1 limbs of `x`).
struct Rescaler {
  const Base* Bf = nullptr;
  size_t k1 = 0, k2 = 0;
  GarnerTables G;             // Garner over the dropped sub-base
  std::vector<u64> half_D;    // floor(D/2) mod each prime of Bfull
  std::vector<u64> Dinv;      // D^{-1} mod each keep prime
  AlignedBuf tmp, tmp2;
  Rescaler() = default;
  Rescaler(const Base& Bfull, size_t k1_) { init(Bfull, k1_); }
  void init(const Base& Bfull, size_t k1_) {
    Bf = &Bfull; k1 = k1_; k2 = Bfull.k() - k1;
    G = GarnerTables(Bfull, k1, Bfull.k());
    BigUInt D(1);
    for (size_t j = k1; j < Bfull.k(); j++) D.mul_small(Bfull.q[j]);
    BigUInt H = D; H.divmod_small(2);
    half_D.resize(Bfull.k());
    for (size_t i = 0; i < Bfull.k(); i++) half_D[i] = H.mod_small(Bfull.q[i]);
    Dinv.resize(k1);
    for (size_t i = 0; i < k1; i++) Dinv[i] = invmod(D.mod_small(Bfull.q[i]), Bfull.q[i]);
    tmp.alloc(Bfull.N); tmp2.alloc(Bfull.N);
  }
  // x: coefficient form, all k1+k2 limbs, values in [0,q). Result in limbs [0,k1).
  // `l` is scratch of N words. Destroys the dropped limbs of x.
  void apply(RNSPoly& x) {
    const Base& B = *Bf;
    size_t N = x.N;
    par_for(B.k(), [&](size_t i) { hx::EltwiseAddMod(x.limb(i), x.limb(i), half_D[i], N, B.q[i]); });
    garner_inplace(x, B, G, tmp.p);
    for (size_t m = 0; m < k1; m++) {
      u64 qm = B.q[m];
      horner_to(tmp.p, x, B, k1, k2, qm, 2, tmp2.p);
      sub_c(x.limb(m), x.limb(m), tmp.p, N, qm);
      fma_c(x.limb(m), x.limb(m), Dinv[m], nullptr, N, qm, 1);
    }
  }
};

// ---------------------------------------------------------- Lifter
// Exact base extension: value h in [0, prod(B_small)) given over the first
// k1 limbs of Bfull, extend to all limbs of Bfull (coefficient form).
struct Lifter {
  const Base* Bf = nullptr;
  size_t k1 = 0;
  GarnerTables G;
  AlignedBuf tmp, tmp2;
  RNSPoly d;                 // digit scratch (allocated once)
  Lifter() = default;
  Lifter(const Base& Bfull, size_t k1_) { init(Bfull, k1_); }
  void init(const Base& Bfull, size_t k1_) {
    Bf = &Bfull; k1 = k1_;
    G = GarnerTables(Bfull, 0, k1);
    tmp.alloc(k1 * Bfull.N); tmp2.alloc(Bfull.N);
    d = RNSPoly(k1, Bfull.N);
  }
  // src: limbs [0,k1) hold h (coefficient form). dst: all limbs of Bfull.
  // src and dst may alias (same RNSPoly) if dst has k = Bfull.k().
  void apply(const RNSPoly& src, RNSPoly& dst, size_t kout = 0) {
    const Base& B = *Bf;
    size_t N = src.N;
    if (kout == 0) kout = B.k();
    for (size_t i = 0; i < k1; i++) memcpy(d.limb(i), src.limb(i), 8 * N);
    garner_inplace(d, B, G, tmp.p);
    for (size_t t = k1; t < kout; t++) horner_to(dst.limb(t), d, B, 0, k1, B.q[t], 2, tmp2.p);
    if (&src != &dst) for (size_t i = 0; i < k1; i++) memcpy(dst.limb(i), src.limb(i), 8 * N);
  }
  // Extend h (first k1 limbs of src over Bfull) to an arbitrary target base
  // Btgt (all of its primes), writing dst over Btgt.
  void apply_to(const RNSPoly& src, RNSPoly& dst, const Base& Btgt) {
    const Base& B = *Bf;
    size_t N = src.N;
    for (size_t i = 0; i < k1; i++) memcpy(d.limb(i), src.limb(i), 8 * N);
    garner_inplace(d, B, G, tmp.p);
    for (size_t t = 0; t < Btgt.k(); t++) horner_to(dst.limb(t), d, B, 0, k1, Btgt.q[t], 2, tmp2.p);
  }
};

// Reduce the value (given as coefficient-form limbs [0,k) of x over base B)
// modulo a small prime t (e.g. alpha). Non-destructive.
inline void poly_mod_small(u64* out, const RNSPoly& x, const Base& B, size_t k, u64 t, AlignedBuf& scratch) {
  size_t N = x.N;
  RNSPoly d(k, N);
  for (size_t i = 0; i < k; i++) memcpy(d.limb(i), x.limb(i), 8 * N);
  GarnerTables G(B, 0, k);
  garner_inplace(d, B, G, scratch.p);
  // digits are ~2^58, far above t^2: Barrett-reduce them first
  FastMod fm(t);
  for (size_t i = 0; i < k; i++) fm.reduce(d.limb(i), d.limb(i), N);
  memcpy(out, d.limb(k - 1), 8 * N);
  for (size_t j = k - 1; j-- > 0;) hx::EltwiseFMAMod(out, out, B.q[j] % t, d.limb(j), N, t, 1);
}

}  // namespace pcf
