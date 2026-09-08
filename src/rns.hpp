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
// the same, restricted to a set of limb indices
inline void poly_ntt_idx(RNSPoly& a, const Base& B, const std::vector<size_t>& idx) {
  par_for(idx.size(), [&](size_t j) { size_t i = idx[j]; B.ntt[i]->ComputeForward(a.limb(i), a.limb(i), 2, 1); });
}
inline void poly_intt_idx(RNSPoly& a, const Base& B, const std::vector<size_t>& idx) {
  par_for(idx.size(), [&](size_t j) { size_t i = idx[j]; B.ntt[i]->ComputeInverse(a.limb(i), a.limb(i), 2, 1); });
}
inline void poly_mul_idx(RNSPoly& r, const RNSPoly& a, const RNSPoly& b, const Base& B, const std::vector<size_t>& idx) {
  par_for(idx.size(), [&](size_t j) { size_t i = idx[j]; hx::EltwiseMultMod(r.limb(i), a.limb(i), b.limb(i), a.N, B.q[i], 1); });
}
inline void poly_mul_acc_idx(RNSPoly& r, const RNSPoly& a, const RNSPoly& b, RNSPoly& tmp, const Base& B, const std::vector<size_t>& idx) {
  par_for(idx.size(), [&](size_t j) { size_t i = idx[j];
    hx::EltwiseMultMod(tmp.limb(i), a.limb(i), b.limb(i), a.N, B.q[i], 1);
    hx::EltwiseAddMod(r.limb(i), r.limb(i), tmp.limb(i), a.N, B.q[i]); });
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
// Mixed-radix (Garner) digits of the value represented by the limbs
// `idx[0..k)` of `x` (sub-base of B).  In place: limb idx[i] afterwards holds
// digit d_i (< q_{idx[i]}) with   value = sum_i d_i * prod_{j<i} q_{idx[j]}.
// Requirements: all primes of the sub-base except possibly the FIRST one lie
// within a factor 2 of each other (every digit is < 2 * every later prime);
// the first prime may be small (alpha).
struct GarnerTables {
  std::vector<size_t> idx;
  std::vector<u64> inv;                 // (prod_{j<i} q_j)^{-1} mod q_i
  std::vector<std::vector<u64>> qm;     // qm[i][j] = q_j mod q_i  (sub-base indices)
  GarnerTables() = default;
  GarnerTables(const Base& B, const std::vector<size_t>& idx_) : idx(idx_) {
    size_t k = idx.size();
    inv.resize(k); qm.assign(k, std::vector<u64>(k));
    for (size_t i = 0; i < k; i++) {
      u64 qi = B.q[idx[i]], prod = 1;
      for (size_t j = 0; j < i; j++) prod = mulmod(prod, B.q[idx[j]] % qi, qi);
      inv[i] = invmod(prod, qi);
      for (size_t j = 0; j < k; j++) qm[i][j] = B.q[idx[j]] % qi;
    }
  }
  size_t k() const { return idx.size(); }
};

inline void garner_inplace(RNSPoly& x, const Base& B, const GarnerTables& G, u64* tmp) {
  size_t N = x.N;
  for (size_t i = 1; i < G.k(); i++) {
    u64 qi = B.q[G.idx[i]];
    // t = Horner(d_{i-1}, ..., d_0) mod qi
    memcpy(tmp, x.limb(G.idx[i - 1]), 8 * N);
    if (i >= 2) {
      for (size_t j = i - 1; j-- > 0;)
        fma_c(tmp, tmp, G.qm[i][j], x.limb(G.idx[j]), N, qi, 2);
    } else {
      red_c(tmp, tmp, N, qi, 2, 1);
    }
    // d_i = (x_i - t) * inv_i
    sub_c(x.limb(G.idx[i]), x.limb(G.idx[i]), tmp, N, qi);
    fma_c(x.limb(G.idx[i]), x.limb(G.idx[i]), G.inv[i], nullptr, N, qi, 1);
  }
}

// Reconstruct the value from the digits in limbs G.idx of `d` modulo prime t,
// into `out` (Horner: v = d_{k-1}; v = v*q_j + d_j).  If t is small
// (t < 2^40, e.g. alpha) the digits are first Barrett-reduced mod t using
// `scratch` (N words).
inline void horner_to(u64* out, const RNSPoly& d, const Base& B, const GarnerTables& G, u64 t, u64 in_factor = 2,
                      u64* scratch = nullptr) {
  size_t N = d.N, k = G.k();
  bool small = t < (1ull << 40);
  FastMod fm(small ? t : 3);
  auto red = [&](const u64* src) -> const u64* {
    if (!small) return src;
    fm.reduce(scratch, src, N);
    return scratch;
  };
  if (small) { const u64* r = red(d.limb(G.idx[k - 1])); memcpy(out, r, 8 * N); in_factor = 1; }
  else red_c(out, d.limb(G.idx[k - 1]), N, t, in_factor, 1);
  for (size_t j = k - 1; j-- > 0;) {
    u64 qj = B.q[G.idx[j]] % t;
    fma_c(out, out, qj, red(d.limb(G.idx[j])), N, t, in_factor);
  }
}

// ---------------------------------------------------------- Rescaler
// x over the base B (coefficient form, integer representative in [0, prod)),
// D = product of the primes `drop`:  h = floor((x + D/2)/D) = round(x/D)
// modulo every prime in `keep`.  Exact.  Output in the `keep` limbs of x; the
// `drop` limbs are destroyed.
struct Rescaler {
  const Base* Bf = nullptr;
  std::vector<size_t> keep, drop;
  GarnerTables G;             // Garner over the dropped sub-base
  std::vector<u64> half_D;    // floor(D/2) mod each prime of B (by limb index)
  std::vector<u64> Dinv;      // D^{-1} mod each keep prime
  AlignedBuf tmp, tmp2;
  Rescaler() = default;
  void init(const Base& Bfull, const std::vector<size_t>& keep_, const std::vector<size_t>& drop_) {
    Bf = &Bfull; keep = keep_; drop = drop_;
    G = GarnerTables(Bfull, drop);
    BigUInt D(1);
    for (size_t j : drop) D.mul_small(Bfull.q[j]);
    BigUInt H = D; H.divmod_small(2);
    half_D.resize(Bfull.k());
    for (size_t i = 0; i < Bfull.k(); i++) half_D[i] = H.mod_small(Bfull.q[i]);
    Dinv.resize(keep.size());
    for (size_t m = 0; m < keep.size(); m++) Dinv[m] = invmod(D.mod_small(Bfull.q[keep[m]]), Bfull.q[keep[m]]);
    tmp.alloc(Bfull.N); tmp2.alloc(Bfull.N);
  }
  void apply(RNSPoly& x) {
    const Base& B = *Bf;
    size_t N = x.N;
    for (size_t i : keep) adds_c(x.limb(i), x.limb(i), half_D[i], N, B.q[i]);
    for (size_t i : drop) adds_c(x.limb(i), x.limb(i), half_D[i], N, B.q[i]);
    garner_inplace(x, B, G, tmp.p);
    for (size_t m = 0; m < keep.size(); m++) {
      u64 qm = B.q[keep[m]];
      horner_to(tmp.p, x, B, G, qm, 2, tmp2.p);
      sub_c(x.limb(keep[m]), x.limb(keep[m]), tmp.p, N, qm);
      fma_c(x.limb(keep[m]), x.limb(keep[m]), Dinv[m], nullptr, N, qm, 1);
    }
  }
};

// ---------------------------------------------------------- Lifter
// Exact base extension: the value h in [0, prod(src primes)) given in the
// `src` limbs of a poly over B is extended to the `tgt` limbs.
struct Lifter {
  const Base* Bf = nullptr;
  std::vector<size_t> src, tgt;
  GarnerTables G;
  Base srcB;                 // compact view of the source primes (no NTT tables)
  AlignedBuf tmp, tmp2;
  RNSPoly d;                 // digit scratch (limb i = source limb src[i])
  Lifter() = default;
  void init(const Base& Bfull, const std::vector<size_t>& src_, const std::vector<size_t>& tgt_) {
    Bf = &Bfull; src = src_; tgt = tgt_;
    std::vector<size_t> local(src.size()); for (size_t i = 0; i < src.size(); i++) local[i] = i;
    srcB.N = Bfull.N; srcB.q.clear(); for (size_t i : src) srcB.q.push_back(Bfull.q[i]);
    G = GarnerTables(srcB, local);
    tmp.alloc(Bfull.N); tmp2.alloc(Bfull.N);
    d = RNSPoly(src.size(), Bfull.N);
  }
  // a: poly over B holding the value in its src limbs; dst: poly over B receiving the tgt limbs (may be the same poly)
  void apply(const RNSPoly& a, RNSPoly& dst) {
    const Base& B = *Bf;
    size_t N = a.N;
    for (size_t i = 0; i < src.size(); i++) memcpy(d.limb(i), a.limb(src[i]), 8 * N);
    garner_inplace(d, srcB, G, tmp.p);
    for (size_t t : tgt) horner_to(dst.limb(t), d, srcB, G, B.q[t], 2, tmp2.p);
    if (&a != &dst) for (size_t i : src) memcpy(dst.limb(i), a.limb(i), 8 * N);
  }
};

}  // namespace pcf
