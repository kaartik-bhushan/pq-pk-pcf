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
    return x % M

lines = open(sys.argv[1] if len(sys.argv) > 1 else "rns_dump.txt").read().split("\n")
primes = list(map(int, lines[0].split()[1:]))
t = lines[1].split(); kb, ka, kg, kQs = int(t[1]), int(t[3]), int(t[5]), int(t[7])
beta_p = primes[:kb]; alpha = primes[ka]; Q_p = primes[ka + 1:kg]
prod = lambda l: reduce(lambda a, b: a * b, l, 1)
beta, Q = prod(beta_p), prod(Q_p)
n_ok = n_bad = 0
for ln in lines[2:]:
    if not ln.strip():
        continue
    t = ln.split()
    v = list(map(int, t[1:]))
    if t[0] == "rescale_g":                       # round(x / (alpha Q~)) mod beta
        x = crt(v[:kg], primes); D = alpha * Q
        ok = ((x + D // 2) // D) % beta == crt(v[kg:kg + kb], beta_p)
    elif t[0] == "rescale_o":                     # round(x / (beta Q_s)) mod alpha
        x = crt(v[:kg], primes); D = beta * prod(Q_p[:kQs])
        ok = ((x + D // 2) // D) % alpha == v[kg]
    elif t[0] == "lift":
        h = crt(v[:kb], beta_p)
        ok = h == crt(v[kb:kb + kg], primes)
    else:
        continue
    n_ok += ok; n_bad += (not ok)
print(f"rns check: {n_ok} ok, {n_bad} bad")
sys.exit(1 if n_bad else 0)
