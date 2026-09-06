// AES-NI based primitives: CTR PRG, PRF, and a Davies-Meyer style hash.
// Used for (i) all randomness sampling, (ii) the shared PRF(K, id) -> R_gamma
// that both parties add before rounding (Construction 4), (iii) the hash H
// of Construction 5 (random-oracle modelled; instantiated as a fixed-key
// AES Davies-Meyer compression keyed by the public input x).
#pragma once
#include <immintrin.h>
#include <wmmintrin.h>
#include <cstdint>
#include <cstring>
#include <array>
#include <stdexcept>
#include <fstream>

namespace pcf {

struct AES128 {
  __m128i rk[11];

  static inline __m128i expand_assist(__m128i t, __m128i k) {
    k = _mm_shuffle_epi32(k, _MM_SHUFFLE(3, 3, 3, 3));
    t = _mm_xor_si128(t, _mm_slli_si128(t, 4));
    t = _mm_xor_si128(t, _mm_slli_si128(t, 4));
    t = _mm_xor_si128(t, _mm_slli_si128(t, 4));
    return _mm_xor_si128(t, k);
  }
  explicit AES128(const uint8_t key[16]) { set_key(key); }
  AES128() { uint8_t z[16] = {0}; set_key(z); }
  void set_key(const uint8_t key[16]) {
    rk[0] = _mm_loadu_si128(reinterpret_cast<const __m128i*>(key));
    rk[1] = expand_assist(rk[0], _mm_aeskeygenassist_si128(rk[0], 0x01));
    rk[2] = expand_assist(rk[1], _mm_aeskeygenassist_si128(rk[1], 0x02));
    rk[3] = expand_assist(rk[2], _mm_aeskeygenassist_si128(rk[2], 0x04));
    rk[4] = expand_assist(rk[3], _mm_aeskeygenassist_si128(rk[3], 0x08));
    rk[5] = expand_assist(rk[4], _mm_aeskeygenassist_si128(rk[4], 0x10));
    rk[6] = expand_assist(rk[5], _mm_aeskeygenassist_si128(rk[5], 0x20));
    rk[7] = expand_assist(rk[6], _mm_aeskeygenassist_si128(rk[6], 0x40));
    rk[8] = expand_assist(rk[7], _mm_aeskeygenassist_si128(rk[7], 0x80));
    rk[9] = expand_assist(rk[8], _mm_aeskeygenassist_si128(rk[8], 0x1b));
    rk[10] = expand_assist(rk[9], _mm_aeskeygenassist_si128(rk[9], 0x36));
  }
  inline __m128i enc(__m128i b) const {
    b = _mm_xor_si128(b, rk[0]);
    for (int i = 1; i < 10; i++) b = _mm_aesenc_si128(b, rk[i]);
    return _mm_aesenclast_si128(b, rk[10]);
  }
  // 8-way interleaved encryption (pipelined AES units).
  inline void enc8(__m128i* b) const {
    for (int j = 0; j < 8; j++) b[j] = _mm_xor_si128(b[j], rk[0]);
    for (int i = 1; i < 10; i++)
      for (int j = 0; j < 8; j++) b[j] = _mm_aesenc_si128(b[j], rk[i]);
    for (int j = 0; j < 8; j++) b[j] = _mm_aesenclast_si128(b[j], rk[10]);
  }
  // Fill `out` (nbytes, multiple of 16 preferred) with CTR keystream for
  // nonce (hi 64 bits) starting at block counter `ctr0`.
  void ctr_fill(uint8_t* out, size_t nbytes, uint64_t nonce, uint64_t ctr0) const {
    size_t nblk = nbytes / 16;
    size_t i = 0;
    __m128i b[8];
    for (; i + 8 <= nblk; i += 8) {
      for (int j = 0; j < 8; j++) b[j] = _mm_set_epi64x((long long)nonce, (long long)(ctr0 + i + j));
      enc8(b);
      for (int j = 0; j < 8; j++) _mm_storeu_si128(reinterpret_cast<__m128i*>(out + 16 * (i + j)), b[j]);
    }
    for (; i < nblk; i++) {
      __m128i x = enc(_mm_set_epi64x((long long)nonce, (long long)(ctr0 + i)));
      _mm_storeu_si128(reinterpret_cast<__m128i*>(out + 16 * i), x);
    }
    size_t rem = nbytes - 16 * nblk;
    if (rem) {
      alignas(16) uint8_t tmp[16];
      __m128i x = enc(_mm_set_epi64x((long long)nonce, (long long)(ctr0 + nblk)));
      _mm_store_si128(reinterpret_cast<__m128i*>(tmp), x);
      memcpy(out + 16 * nblk, tmp, rem);
    }
  }
};

// Simple CSPRNG: AES-CTR with a fresh key from /dev/urandom, incrementing nonce.
struct Rng {
  AES128 aes;
  uint64_t nonce = 0;
  uint64_t ctr = 0;
  Rng() {
    uint8_t key[16];
    std::ifstream f("/dev/urandom", std::ios::binary);
    if (!f.read(reinterpret_cast<char*>(key), 16)) throw std::runtime_error("urandom");
    f.read(reinterpret_cast<char*>(&nonce), 8);
    aes.set_key(key);
  }
  explicit Rng(const uint8_t key[16], uint64_t nonce_) : aes(key), nonce(nonce_) {}
  void fill(void* out, size_t nbytes) {
    aes.ctr_fill(reinterpret_cast<uint8_t*>(out), nbytes, nonce, ctr);
    ctr += (nbytes + 15) / 16;
  }
  uint64_t u64() { uint64_t v; fill(&v, 8); return v; }
};

// H(y_1..y_8 (17-bit each), slot index, x) -> 128 bits, keyed by x (16 bytes).
// Two-block Davies-Meyer: h = E_x(B0)^B0 ; r = E_x(B1^h) ^ (B1^h).
// B0 = y1|y2|y3|y4 (68 bits) | slot (bits 96..111) | tag 0 (bits 120..127)
// B1 = y5|y6|y7|y8 (68 bits) | slot | tag 1.
struct SlotHash {
  AES128 aes;
  explicit SlotHash(const uint8_t x[16]) : aes(x) {}
  static inline __m128i pack(const uint32_t* y, uint32_t slot, uint32_t tag) {
    uint64_t lo = (uint64_t)y[0] | ((uint64_t)y[1] << 17) | ((uint64_t)y[2] << 34) | ((uint64_t)y[3] << 51);
    uint64_t hi = ((uint64_t)y[3] >> 13) | ((uint64_t)slot << 32) | ((uint64_t)tag << 56);
    return _mm_set_epi64x((long long)hi, (long long)lo);
  }
  // y: 8 values in [0, 2^17). out: 16 bytes.
  inline void hash(const uint32_t y[8], uint32_t slot, uint8_t out[16]) const {
    __m128i b0 = pack(y, slot, 0);
    __m128i b1 = pack(y + 4, slot, 1);
    __m128i h = _mm_xor_si128(aes.enc(b0), b0);
    __m128i c = _mm_xor_si128(b1, h);
    __m128i r = _mm_xor_si128(aes.enc(c), c);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out), r);
  }
  // Hash all N slots: y_j given as 8 arrays (ys[j][i]); 8 slots are processed
  // together so that the AES units are pipelined. sel (optional): only slots
  // with sel[i] == want are written.
  void hash_all(const uint64_t* const ys[8], size_t N, uint8_t* out, const uint8_t* sel = nullptr, uint8_t want = 0) const {
    uint32_t y[8];
    size_t i = 0;
    for (; i + 8 <= N; i += 8) {
      __m128i b0[8], b1[8];
      for (int k = 0; k < 8; k++) {
        for (int j = 0; j < 8; j++) y[j] = (uint32_t)ys[j][i + k];
        b0[k] = pack(y, (uint32_t)(i + k), 0);
        b1[k] = pack(y + 4, (uint32_t)(i + k), 1);
      }
      __m128i h[8];
      for (int k = 0; k < 8; k++) h[k] = b0[k];
      aes.enc8(h);
      for (int k = 0; k < 8; k++) h[k] = _mm_xor_si128(_mm_xor_si128(h[k], b0[k]), b1[k]);   // c = b1 ^ (E(b0)^b0)
      __m128i r[8];
      for (int k = 0; k < 8; k++) r[k] = h[k];
      aes.enc8(r);
      for (int k = 0; k < 8; k++) {
        if (sel && sel[i + k] != want) continue;
        _mm_storeu_si128(reinterpret_cast<__m128i*>(out + 16 * (i + k)), _mm_xor_si128(r[k], h[k]));
      }
    }
    for (; i < N; i++) {
      if (sel && sel[i] != want) continue;
      for (int j = 0; j < 8; j++) y[j] = (uint32_t)ys[j][i];
      hash(y, (uint32_t)i, out + 16 * i);
    }
  }
};

}  // namespace pcf
