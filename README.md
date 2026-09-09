# Packed public-key PCF for OT (Construction 5) — implementation

C++17 implementation of the packed public-key pseudorandom correlation function
for OT correlations of *"(Significantly) Faster Post-Quantum Public-Key PCFs for
OT"* (Construction 5, Section 7.3), instantiated with

* the succinct half-chosen VOLE with local reconstruction (Construction 2),
* the compact lattice-based packed public-key aHMAC (Construction 4, version
  of 2026-09-08: SP-RLWE public samples, KDM-Enc1, KDM-Enc-Pack = encryptions of
  the output secrets `Delta^(j) in R_alpha` under `s` (randomness `theta^(j) <- chi_s`), memory-to-input
  conversion, InpMemMult with rounding, and the rounded Output step
  `Y^(j) = floor(-M_1 v1^(j))_{gamma/alpha}` / `floor(M_0 v2^(j) - M_1 v1^(j))_{gamma/alpha}`),
* the XOR5-MAJ7 GAR-wPRF written as an RMS program (Section 8),
* the paper's parameter set: `d = 2^15`, `alpha = 65537`, `log beta >= 287`,
  `log gamma >= 725`, `n = 4900` (perfect square closest to `2^12.25`),
  `B_e = 21` (CBD(21), sigma = 3.24), ternary secrets.

All ring arithmetic is RNS/NTT on top of Intel HEXL. One evaluation produces
`d = 32768` 1-out-of-2 OTs (one per packing slot) of 128-bit strings.

## Results

Single thread per party, sandbox VM (2 vCPU Xeon @2.8 GHz, AVX-512 without
IFMA). The VM hosting this session changed between the two versions and is
~1.5x slower than the one used for the first report, so the previous version
was re-measured on the same host for reference. `--kQs 1` runs the Output
stage over the divisor `gamma_o = alpha * beta * q_1` of `gamma` (see design
notes); the default runs it over the full `gamma` exactly as written.

| configuration (n = 1024 unless noted) | party 0 / eval | party 1 / eval | OTs/s (slower party) |
|---|---|---|---|
| current construction, 58-bit primes, Output over full gamma (default) | 339 ms | 448 ms | 73 k |
| current construction, 58-bit primes, `--kQs 1` | 241 ms | 325 ms | **101 k** |
| current construction, 58-bit primes, `--kQs 1`, paper's n = 4900 (`--lazy0`) | 264 ms | 298 ms | **110 k** |
| current construction, 50-bit preset (`--ifma --kQs 1`), 2 threads/party | 192 ms | 226 ms | 145 k |
| previous version (ternary payloads under a fresh RLWE sample mod alpha*Q_s), 58-bit | 226 ms | 301 ms | 109 k |

On the earlier (faster) host the previous version ran at 160-190 k OTs/s
single-threaded; scaled by the measured host ratio the current construction
with `--kQs 1` corresponds to ~150-175 k there, and an AVX512-IFMA CPU
(e.g. the Xeon 8360Y used for the paper's estimates) should roughly double
the NTT throughput again.

Party 1 breakdown (58-bit, `--kQs 1`, n = 1024): NTT 78 ms, output-round 71,
pointwise 45, inputshares 37, rescale 31, lift 21, lincomb 15, output-lift 8,
memshares 7, hash 6, PRF 2.

One-time costs for `n = 4900` (58-bit primes): KeyGen0 1.5 s, KeyGen1 12 s,
KeyDer0 1.2 s (lazy), KeyDer1 248 s (the `n^{1.5}` inner products of Recon1
for all `n` inputs, eager). Public keys: pk0 ≈ 240 MB (141 A_j + 3 u + 14 v
ring elements), pk1 ≈ 90 MB (70 v_t).

## Build / run

```
git clone https://github.com/intel/hexl && cmake -S hexl -B hexl/build -DCMAKE_BUILD_TYPE=Release \
    -DHEXL_BENCHMARK=OFF -DHEXL_TESTING=OFF -DCMAKE_INSTALL_PREFIX=$PWD/hexl-install && cmake --build hexl/build --target install -j
cmake -S pcf -B pcf/build -DHEXL_ROOT=$PWD/hexl-install && cmake --build pcf/build -j
cd pcf/build
./rns_test && python3 ../check_rns.py rns_dump.txt          # RNS rescale/lift vs Python big integers
./pcf_bench --N 4096 --n 64 --evals 3                        # quick end-to-end test (all checks)
./pcf_bench --N 32768 --n 1024 --evals 5 --kQs 1             # full ring dimension benchmark
./pcf_bench --N 32768 --n 4900 --evals 3 --lazy0 --kQs 1     # the paper's n (needs ~3 GB; party 0 lazy)
```

Options: `--N d`, `--n key-length` (perfect square), `--evals k`, `--ifma`
(50-bit primes preset), `--prime-bits b`, `--kQ k` (limbs of Q~, gamma =
alpha*beta*Q~), `--kQs k` (Output stage over gamma_o = alpha*beta*Q_s with Q_s
the first k primes of Q~; 0 = all = the literal construction), `--rand-limbs k` (limbs of the
shared PRF value randomised before rounding), `--lazy0` (party 0 derives its
memory shares at evaluation time instead of at KeyDer), `--threads t`
(OpenMP over limbs / coefficient chunks), `--no-check`.

Checks performed by `pcf_bench` (unless `--no-check`): for every memory cell,
`M_01 - M_11 = s * g (mod beta)` and `M_10 = -g`; the RMS values of P and P-bar
vanish exactly in the slots where the wPRF is 0 / 1; the output shares satisfy
`y_0 - y_1 = s_l P_l` and `Y_0^(j) - Y_1^(j) = Delta^(j)_l P_l (mod alpha)` in
every slot for both branches; and finally party 0's `r_i^{(b_i)}` equals party
1's `r'_i` for all 32768 slots (and the other string never collides).

## Layout

* `src/rns.hpp` – mini big integer, RNS bases, HEXL wrappers, exact Garner
  based rescaling `round(x/Q)` and base extension (`Rescaler`, `Lifter`).
* `src/aes.hpp` – AES-NI CTR PRG/PRF, Davies–Meyer slot hash `H`.
* `src/sampling.hpp` – uniform / CBD / ternary samplers.
* `src/pcf.hpp` – parameters, Setup/KeyGen/KeyDer/Eval of Construction 5.
* `src/main.cpp` – tests + benchmark; `src/rns_test.cpp` + `check_rns.py`.

## Design notes (where the implementation is more specific than the paper)

**Moduli.** All limbs except alpha are NTT-friendly primes `q = 1 mod 2^16`;
alpha = 65537 is itself one limb of gamma:
`beta = 5 x 58-bit` (290 bits), `gamma = beta * alpha * Q~` with `Q~ = 8 x
58-bit` (771 bits ≥ 725). Because `alpha | gamma`, both `floor(gamma/beta) =
alpha Q~` and `floor(gamma/alpha) = beta Q~` are exact integers, and both
roundings of Construction 4 are exact RNS rescalings (Garner digits of the
dropped limbs, Horner into the kept limbs, `(x - l) D^{-1}`, with the
round-half trick `round(x/D) = floor((x + D/2)/D)`): `floor(.)_{gamma/beta}`
keeps the beta limbs (InpMemMult), `floor(.)_{gamma/alpha}` keeps the alpha
limb (Output). Lifting a beta-limb value back to all gamma limbs is the same
exact base extension. The VOLE modulus is `p = beta' * Q'` with `Q' = 3 limbs`.
The RNS layout of a gamma element is `[beta_1..beta_5 | alpha | Q~_1..Q~_8]`;
party 1's cleartext values are kept in NTT form over all limbs, so their
alpha-limb *is* the slot vector `psi^{-1}(g mod alpha)`.

**VOLE output modulus beta' (important).** The memory shares of the *inputs*
`(z_0, z_1)` with `z_0 - z_1 = s x` are produced by Construction 2 over
`R_beta'` with `beta' = 2^116` (two limbs), **not** over `R_beta`.
Lemma 5 gives the converted input share the error term `e * z_1`
(`I_{1,1} - s I_{1,0} = x s Q + e' x + e z_1`), which Lemma 7 multiplies by the
memory value `g`: `ê = g * (e' f + e z_1) + ẽ M_01`. With `z_1` uniform mod
`beta ≈ 2^290` this term is ≈ `2^{211+7+300}` (the degree-7 values have
coefficients of ≈ `2^211`, see below), far above `gamma/beta = 2^464`, and the
last multiplications of the chain fail (observed experimentally). With
`z_1 < beta'` the term is ≈ `2^{211+7+127}` (worst case `2^368`), which is
what the paper's `‖max-err‖` bound (and hence `log gamma - log beta = 438`)
actually accounts for. `beta' ≥ 2^{64} d |s x_S|` is all that Lemma 2 needs for the
inputs. Everything else (Add/Mul/Output) is unchanged; the initial memory
shares are simply small integers.

**Output stage (Delta-ciphertexts + rounded Output).** Exactly as in the
current Construction 4: `PKPCF.KeyGen(0)` samples the output secrets
`Delta^(j) <- R_alpha` (uniform coefficients in `[0, alpha)`); KDM-Enc-Pack
encrypts them under `s` with fresh randomness `theta^(j) <- chi_s`:
`v1^(j) = theta^(j) a + e1`, `v2^(j) = theta^(j) b1 + e2 + Delta^(j)
floor(gamma/alpha)`, with the SP-RLWE pair `(a, b1)` of step 2 (in the code:
`sk.delta`, `pk.v1/v2`, negated copies `ek.nv1/nv2`, scale
`delta_scale_mod_g`); the
Output step computes `Y_0^(j) = floor(-M_01 v1^(j))_{gamma/alpha}` and
`Y_1^(j) = floor(M_10 v2^(j) - M_11 v1^(j))_{gamma/alpha}` (`M_10 = -g`), all
mod gamma, which the test harness verifies to satisfy `Y_0^(j) - Y_1^(j) =
Delta^(j) P (mod alpha)` in every slot. (No PRF term is added before this
rounding, as in the paper; `-M v1 mod gamma` is pseudorandom anyway.) With
`--kQs k` the same computation is carried out over the divisor `gamma_o =
alpha * beta * Q_s` of gamma (`Q_s` = first `k` primes of `Q~`), i.e.
KDM-Enc-Pack uses `(a mod gamma_o, b1 mod gamma_o)` — still a valid RLWE
pair since `gamma_o | gamma` — with payload scale `floor(gamma_o/alpha) =
beta Q_s`, and the Output rounds by `gamma_o/alpha`. The rounding error of
this step is `g * (theta e_b + e2 - s e1) ≤ 2^250` (worst case), so
`gamma_o/alpha ≥ 2^{64+15+250}` is enough: one 58-bit prime (`--kQs 1`,
`gamma_o ≈ 2^365`) gives a failure bound of `2^{-83}`, while the full gamma
(`2^771`) costs ~2x on this stage for nothing. The tuple `(y, Y^(1..7))`
depends on the 8 independent slot secrets `(s_l, Delta^(1)_l, ...,
Delta^(7)_l)`, i.e. 128 bits.

**RMS program (9 multiplications instead of 18).** Writing `X, X̄, M, M̄` for
the four rescaled predicate polynomials, `P = X M̄ + X̄ M` and
`P̄ = X M + X̄ M̄` satisfy `P + P̄ = (X + X̄)(M + M̄)` and
`P - P̄ = (X - X̄)(M̄ - M)`, and each factor is a univariate polynomial in
`S_X = Σ_{i≤5} z[x_i]` or `S_M = Σ_{i>5} z[x_i]`:
`X+X̄ = 2S³-15S²+31S-15`, `X-X̄ = 3S²-15S+15`,
`M+M̄ = 2T⁴-28T³+190T²-644T+840`, `M̄-M = -16T³+168T²-632T+840`.
So both programs need the memory values `T², T³, T⁴` (3 multiplications by
the input share of `S_M`), then `(M+M̄)·S^k` and `(M̄-M)·S^k`, `k = 1..3`
(6 multiplications by the input share of `S_X`); linear combinations with
integer constants and with the constant-1 memory share `((0,s),(-1,0))` are
free (Remark 5), and `P = (A + D)/2`, `P̄ = (A - D)/2` are obtained by
multiplying the shares by `2^{-1} mod beta` (2P, 2P̄ are exact even integers,
so Lemma 2 applies unchanged). Only two input shares are needed per
evaluation. Value magnitudes grow by at most a factor ~4 (`log B` +2), which
the margins absorb. The re-scaling by `1/3` and `1/24` of Section 8 is not
needed for `alpha = 65537` (`max |P| = 15·840 = 12600 < alpha`) and is not
applied.

**Shared randomness r.** `PRF(K, id)` is AES-CTR. Only the first two Q-limbs
of `r` are randomised (`--rand-limbs`): a uniform offset in a subgroup of
`Z_Q` of order `q_1 q_2 ≈ 2^116` already puts at most one candidate of the
coset into the rounding-boundary interval, so the union bound of Lemma 7
holds with probability `≥ 1 - d/(q_1 q_2) ≈ 2^{-100}` per element; this
saves 80 % of the PRF bandwidth compared with a full `R_gamma` element.

**Memory-share derivation (Remark 6).** Recon1 is an inner product of `√n+2`
ring elements over `R_p` per input, so party 1 derives all `n` memory shares
once at KeyDer (eager, `n · (√n+2)` products, 2·8·d bytes per input:
2.5 GB for `n = 4900` with 58-bit primes). Party 0's Recon0 is a single
product; it can be eager (`--n 1024`: 0.5 GB) or lazy (`--lazy0`, +12 ms per
evaluation). The evaluation cost is otherwise independent of `n`.

**Hash / input.** The PCF input is a 16-byte seed `x`; the 12 wPRF indices are
`AES_x`-derived. `H(y_1..y_8, i, x)` is a two-block Davies–Meyer AES
compression keyed by `x` (random-oracle instantiation, 128-bit outputs);
party 1 hashes only the slots of its branch `b_i = F_{K_i}(x)`, evaluated
directly on the packed key bits.

**Security parameters.** RLWE dimension `d = 2^15` with moduli of 771 (gamma),
365 (gamma_o with `--kQs 1`) and 290 (p) bits, ternary secrets, CBD(21) errors – within the
HE-standard 128-bit range for `d = 2^15`. Measured magnitude of the degree-7
values (coefficients of `(M+M̄)·S_X^3`): `2^211` (the inputs `K~` have
non-centred coefficients in `[0, alpha)`, so products grow by a factor `d`,
close to the worst-case bound `2^{221.6}`). Failure probabilities per
evaluation: rounding (Lemma 7, worst-case `ê ≤ 2^368` vs `gamma/beta = 2^480`)
`≤ 2^{-96}`; Output rounding (`ê ≤ 2^250` vs `gamma_o/alpha ≥ 2^348`) `≤ 2^{-82}`;
Lemma 2 lift of the inputs `≤ 2^{-67}`; Lemma 2 lift of the final values
(`‖s·g‖ ≈ 2^218` typical vs `beta = 2^290`) `≈ 2^{-57}` – the paper's
worst-case formula `beta ≥ 2^64 d B` would need 300 bits, which the `--ifma`
preset (6 x 50-bit limbs) provides.

## Remarks on the paper found while implementing

1. `log B`: with `|Var| = 7, t = 7, alpha = 2^16, d = 2^15`, the stated
   bound `|Var|^t alpha^t d^{t-1}` is `2^{221.6}`, not `2^{208}`, and the
   measured values reach `2^211`; `beta ≥ 2^{64} d B` then asks for ~300
   bits rather than 287 (the 58-bit default here has 290, the 50-bit preset
   300).
2. The `e·z_1` term (Lemma 5) forces the VOLE output shares to live over a
   small modulus `beta' ≪ beta` (see above); with `beta' = beta` the
   construction is incorrect for these parameters.
3. Output step (version of 2026-09-08): `⌊−M v1 (mod α)⌉_{γ/α}` presumably
   means the product taken mod γ and then rounded (that is what is
   implemented; reducing mod α first would destroy the payload). For the
   rounding to be an exact RNS operation `γ/α` should be an integer, i.e.
   α should divide γ — which is how γ is built here. The modulus of the
   KDM-Enc-Pack ciphertexts only needs `γ_o/α ≥ 2^{64} d ‖g·ē‖ ≈ 2^329`,
   far below γ; using a divisor `γ_o | γ` (option `--kQs`) halves the cost
   of the Output stage. Also, party 0's `Eval` parses `k_0 = (ek_0, s,
   {Δ})` but `KeyDer(0)` only outputs `(ek_0, s)`; the Δ's are not needed
   by `Eval(0)` (they are inside the `v2`'s), so the implementation does
   not pass them.
4. Construction 1/5 evaluate `P` and `P̄` as separate programs; the `A/D`
   decomposition above halves the multiplication count and needs only two
   input shares.
5. Build note: HEXL's CMake enables its AVX512-VBMI2 kernels (`vpshrdq`,
   used by the small-modulus `EltwiseMultMod` path that the alpha limb
   exercises) whenever a `try_run` with `-march=native` succeeds, without a
   runtime check; HEXL must therefore be built on the machine it runs on (or
   with VBMI2 disabled).
