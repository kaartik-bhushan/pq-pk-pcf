# Packed public-key PCF for OT (Construction 5) — implementation

C++17 implementation of the packed public-key pseudorandom correlation function
for OT correlations of *"(Significantly) Faster Post-Quantum Public-Key PCFs for
OT"* (Construction 5, Section 7.3), instantiated with

* the succinct half-chosen VOLE with local reconstruction (Construction 2),
* the compact lattice-based packed primitive (Construction 4: SP-RLWE public
  samples, KDM-Enc1, memory-to-input conversion, InpMemMult with rounding),
* the XOR5-MAJ7 GAR-wPRF written as an RMS program (Section 8),
* the paper's parameter set: `d = 2^15`, `alpha = 65537`, `log beta >= 287`,
  `log gamma >= 725`, `n = 4900` (perfect square closest to `2^12.25`),
  `B_e = 21` (CBD(21), sigma = 3.24), ternary secrets.

All ring arithmetic is RNS/NTT on top of Intel HEXL. One evaluation produces
`d = 32768` 1-out-of-2 OTs (one per packing slot) of 128-bit strings.

## Results (single thread per party, this sandbox: 2-vCPU Xeon @2.8 GHz, AVX-512 **without** IFMA)

| configuration | party 0 / eval | party 1 / eval | OTs/s (slower party) |
|---|---|---|---|
| 58-bit primes (13 gamma limbs), 1 thread | 149 ms | 204 ms | **160 k** |
| 50-bit primes (15 gamma limbs, `--ifma` preset), 1 thread | 127 ms | 174 ms | **189 k** |
| 50-bit primes, 2 threads per party | 92 ms | 119 ms | 277 k |

The `--ifma` preset uses primes below `2^50`; HEXL then takes its faster
kernels even without IFMA, and on an AVX512-IFMA CPU (e.g. the Xeon 8360Y used
for the paper's estimates) the NTTs are roughly another 2x faster, so
> 250 k OTs/s single-threaded is the expectation there. All numbers above are
end-to-end (including memory-share look-up, input-share generation, the RMS
program, the theta outputs, slot decoding and hashing) and were produced by
`pcf_bench` with the full correctness checks enabled in separate runs.

Party 1 per-eval breakdown (50-bit preset, 1 thread): output-theta 43 ms,
NTT 29, input shares 25, pointwise 22, rescale 15, lift 12, lincomb 11,
memshares 5, hash 5, PRF 1.

One-time costs for the paper's `n = 4900` (58-bit primes): KeyGen0 0.6 s,
KeyGen1 8.3 s, KeyDer0 0.4 s (lazy) / ~10 s (eager), KeyDer1 183 s (the
`n^{1.5}` inner products of Recon1 for all `n` inputs, eager). Public keys:
pk0 ≈ 215 MB (141 A_j + 3 u + 14 theta ciphertexts), pk1 ≈ 90 MB (70 v_t).

## Build / run

```
git clone https://github.com/intel/hexl && cmake -S hexl -B hexl/build -DCMAKE_BUILD_TYPE=Release \
    -DHEXL_BENCHMARK=OFF -DHEXL_TESTING=OFF -DCMAKE_INSTALL_PREFIX=$PWD/hexl-install && cmake --build hexl/build --target install -j
cmake -S pcf -B pcf/build -DHEXL_ROOT=$PWD/hexl-install && cmake --build pcf/build -j
cd pcf/build
./rns_test && python3 ../check_rns.py rns_dump.txt          # RNS rescale/lift vs Python big integers
./pcf_bench --N 4096 --n 64 --evals 3                        # quick end-to-end test (all checks)
./pcf_bench --N 32768 --n 1024 --evals 5 --ifma              # full ring dimension benchmark
./pcf_bench --N 32768 --n 4900 --evals 3 --lazy0             # the paper's n (needs ~3 GB; party 0 lazy)
```

Options: `--N d`, `--n key-length` (perfect square), `--evals k`, `--ifma`
(50-bit primes preset), `--prime-bits b`, `--kQ k` (limbs of Q = gamma/beta),
`--kQs k` (limbs of Q_s, theta-conversion), `--rand-limbs k` (limbs of the
shared PRF value randomised before rounding), `--lazy0` (party 0 derives its
memory shares at evaluation time instead of at KeyDer), `--threads t`
(OpenMP over limbs / coefficient chunks), `--no-check`.

Checks performed by `pcf_bench` (unless `--no-check`): for every memory cell,
`M_01 - M_11 = s * g (mod beta)` and `M_10 = -g`; the RMS values of P and P-bar
vanish exactly in the slots where the wPRF is 0 / 1; the output shares satisfy
`y_0 - y_1 = s_l P_l` and `Y_0^(j) - Y_1^(j) = theta^(j)_l P_l (mod alpha)` in
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

**Moduli.** All limbs are NTT-friendly primes `q = 1 mod 2^16`.
`beta = 5 x 58-bit` (290 bits), `gamma = beta * Q` with `Q = 8 x 58-bit`
(754 bits ≥ 725), so `floor(gamma/beta) = Q` and the rounding
`⌊x⌉_{gamma/beta} = round(x/Q)` is an exact RNS rescale (Garner digits of the
Q-limbs, Horner into the beta-limbs, `(x - l) Q^{-1}`); the round-half trick
`round(x/Q) = floor((x + Q/2)/Q)` avoids any comparison. Lifting a beta-limb
value back to all gamma limbs is the same exact base extension. The VOLE
modulus is `p = beta' * Q'` with `Q' = 3 limbs`.

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

**Output stage / theta masks.** As written, the Output step of Construction 4
sets `Y^(j) = -M_1 * v1^(j) (mod alpha)`, so party 0's whole 8-tuple in slot
`l` is a function of the single `F_alpha` value of its share: party 1 (who
knows `y_{1,l}` and the public `v`'s) can predict party 0's hidden string with
probability `1/alpha = 2^{-16}` by guessing `s_l`, and the two branches
collide with probability `2^{-16}` per slot (about one slot per evaluation
– this showed up as "other-string collisions" in the first runs). The
Delta*round(gamma/alpha) term of KDM-Enc-Pack does not help because party 1
knows `Y_1^(j)` anyway. The implementation therefore realises Definition 15
literally: party 0 publishes RLWE ciphertexts `(v1^(j), v2^(j)) =
(r a + e1, r b + e2 + theta^(j) Q_s)` of `theta^(j)` under `s` (modulus
`gamma' = alpha * Q_s`, `Q_s = 6 x 58-bit` ≈ 2^348, chosen from the worst-case
bound `g * ē ≤ 2^250`), and both parties run one InpMemMult-style step per
`j` on the final memory share (`round((-M_01 v1 + r)/Q_s)` resp.
`round((-M_11 v1 - g v2 + r)/Q_s)`, all mod `alpha`), which gives
`Y_0^(j) - Y_1^(j) = theta^(j) P (mod alpha)` exactly (checked by the test
harness). The tuple now depends on the 8 independent slot secrets
`(s_l, theta^(1)_l, ..., theta^(7)_l)`, i.e. 128 bits. Cost: ~25 % of party
1's time (`output-theta`).

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

**Security parameters.** RLWE dimension `d = 2^15` with moduli of 754 (gamma),
367 (gamma') and 290 (p) bits, ternary secrets, CBD(21) errors – within the
HE-standard 128-bit range for `d = 2^15`. Measured magnitude of the degree-7
values (coefficients of `(M+M̄)·S_X^3`): `2^211` (the inputs `K~` have
non-centred coefficients in `[0, alpha)`, so products grow by a factor `d`,
close to the worst-case bound `2^{221.6}`). Failure probabilities per
evaluation: rounding (Lemma 7, worst-case `ê ≤ 2^368` vs `Q = 2^464`)
`≤ 2^{-80}`; theta conversion (`ê ≤ 2^250` vs `Q_s = 2^348`) `≤ 2^{-82}`;
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
3. The Output step of Construction 4 does not achieve Definition 15
   (`y_0^(j) - y_1^(j) = theta^(j) P`), and Construction 5 as written leaks
   party 0's hidden string with probability `2^{-16}` per slot; the
   `Delta·round(gamma/alpha)` masking in KDM-Enc-Pack has no effect. The fix
   implemented here (RLWE encryptions of `theta^(j) Q_s` under `s`, one
   rounding step per `j`) adds about 25 % to the evaluation time.
4. Construction 1/5 evaluate `P` and `P̄` as separate programs; the `A/D`
   decomposition above halves the multiplication count and needs only two
   input shares.
