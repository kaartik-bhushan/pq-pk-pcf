#!/usr/bin/env python3
"""Verify rns_dump.txt (from rns_test) against Python big integers."""
import sys
from functools import reduce

def crt(res, mods):
    M = reduce(lambda a, b: a * b, mods)
    x = 0
    for r, m in zip(res, mods):
        Mi = M // m
        x += r * Mi * pow(Mi, -1, m)
    return x % M, M

lines = open(sys.argv[1] if len(sys.argv) > 1 else "rns_dump.txt").read().split("\n")
primes = list(map(int, lines[0].split()[1:]))
_, kb, _, kg, _, alpha = lines[1].split()
kb, kg, alpha = int(kb), int(kg), int(alpha)
beta_p, Q_p = primes[:kb], primes[kb:kg]
beta = reduce(lambda a, b: a * b, beta_p)
Q = reduce(lambda a, b: a * b, Q_p)
n_ok = n_bad = 0
for ln in lines[2:]:
    if not ln.strip():
        continue
    t = ln.split()
    v = list(map(int, t[1:]))
    if t[0] == "rescale":
        x, _ = crt(v[:kg], primes[:kg])
        expect = ((x + Q // 2) // Q) % beta
        got, _ = crt(v[kg:kg + kb], beta_p)
        ok = expect == got
    elif t[0] == "lift":
        h, _ = crt(v[:kb], beta_p)
        got, _ = crt(v[kb:kb + kg], primes[:kg])
        ok = (h == got) and (v[kb + kg] == h % alpha)
    else:
        continue
    n_ok += ok; n_bad += (not ok)
print(f"rns check: {n_ok} ok, {n_bad} bad")
sys.exit(1 if n_bad else 0)
