#pragma once
#include "aes.hpp"
#include "rns.hpp"

namespace pcf {

// Uniform element of R_q in RNS (each limb uniform mod its 58-bit prime).
// Primes are in (2^57, 2^58): mask to 58 bits, then reduce from [0,2q).
inline void sample_uniform(RNSPoly& r, const Base& B, Rng& rng) {
  for (size_t i = 0; i < r.k; i++) {
    u64* d = r.limb(i);
    rng.fill(d, 8 * r.N);
    if (B.q[i] < (1ull << 40)) { FastMod(B.q[i]).reduce(d, d, r.N); continue; }  // small prime (alpha)
    const u64 mask = (1ull << (64 - __builtin_clzll(B.q[i]))) - 1;                  // 2^bits - 1 with q in (2^(bits-1), 2^bits)
    for (size_t j = 0; j < r.N; j++) d[j] &= mask;
    hx::EltwiseReduceMod(d, d, r.N, B.q[i], 2, 1);
  }
}

// Centered binomial CBD(eta): sum of eta bit differences, |e| <= eta.
// eta <= 32 uses one 64-bit word per coefficient.
inline void sample_cbd(int64_t* out, size_t N, int eta, Rng& rng) {
  std::vector<u64> w(N);
  rng.fill(w.data(), 8 * N);
  const u64 m = (eta == 32) ? ~0ull : ((1ull << eta) - 1);
  for (size_t j = 0; j < N; j++)
    out[j] = (int64_t)__builtin_popcountll(w[j] & m) - (int64_t)__builtin_popcountll((w[j] >> 32) & m);
}

// Uniform ternary {-1,0,1}.
inline void sample_ternary(int64_t* out, size_t N, Rng& rng) {
  std::vector<uint8_t> w(N);
  size_t got = 0;
  while (got < N) {
    rng.fill(w.data(), N);
    for (size_t j = 0; j < N && got < N; j++) {
      uint8_t v = w[j] & 3;
      if (v == 3) continue;
      out[got++] = (int64_t)v - 1;
    }
  }
}

// Uniform in [0, t) for small t (t < 2^20); bias 2^-32-ish, irrelevant here.
inline void sample_uniform_small(u64* out, size_t N, u64 t, Rng& rng) {
  std::vector<uint32_t> w(N);
  rng.fill(w.data(), 4 * N);
  FastMod fm(t);
  for (size_t j = 0; j < N; j++) out[j] = fm(w[j]);
}

}  // namespace pcf
