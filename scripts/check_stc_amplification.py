#!/usr/bin/env python3
"""
check_stc_amplification.py

QUESTION
--------
the reference audit charges StC with audited amplification "exactly 1" (sec:stc, eq. tau-e2e,
Thm main Step 5). CtS carries the 1/n of the inverse DFT (folded into S_scale);
StC does not carry a compensating 1/n. Under the reference audit's own Table-gates rule
LinTrans: B' = rho(M) B, what is rho(StC)? 1, or ~n?

This is pure numpy on IDEAL (unencrypted) values: an algebraic property of the
special FFT. No OpenFHE, so no DEV flag. Checked at N = 2^3 .. 2^LOGN_MAX and
extrapolated to N = 2^17 via rho(V0) = n (exact, unit-modulus entries).

CONVENTIONS (full packing, as in the reference audit)
  zeta = exp(i pi / N),  g_j = 5^j mod 2N,  j < n = N/2
  V  (n x N): V[j,k] = zeta^(g_j k)            decode: slots = V m / Delta
  V0 (n x n): first n columns.  Since g_j = 1 mod 4, zeta^(g_j n) = i, so
              V[:, n:] = i V0  and  slots = V0 (m_lo + i m_hi) / Delta
  CtS = (1/n) V0^*  then Re/Im split into two real channel streams
  StC = V0 applied to the recombination (y0 + i y1), then back gate 2^degb
  rho(M) = max_i sum_j |M_ij| = induced inf->inf norm (the reference audit Table-gates rule).

PARTS
  A  gates: V[:, n:] = i V0, V0 V0^* = n I, decode identity
  B  rho(CtS), rho(V0), rho([V0, iV0]), worst case for REAL channel errors
     (with an explicit adversarial vector that attains it), random-error ratio
  C  end-to-end ideal pipeline: inject a per-channel EvalMod-output error of
     size eps_ch, measure R = (final decoded slot error) / (eps_ch * 2^degb).
     the reference audit's model (recombination Add x2, StC amp 1) predicts R = 2.
  D  the reference audit headroom with ONLY the StC factor changed (arithmetic on the reference audit's
     own constants, not a measurement)

USAGE
  python3 check_stc_amplification.py            # LOGN_MAX = 12 (~1 GB peak RAM)
  python3 check_stc_amplification.py 11         # lighter
Inspect the whole output.
"""
import sys
import math
import numpy as np

LOGN_MAX = int(sys.argv[1]) if len(sys.argv) > 1 else 12
LOGN_LIST = list(range(3, LOGN_MAX + 1))
DECODE_GATE_MAX_LOGN = 10     # the n x N decode matrix gets large past this
THETA_GRID = 2048             # grid over [0, pi/2) for the real-channel sup
ROWS_REAL = 8                 # rows scanned for the real-channel sup
SEED = 20260929

# the reference audit constants (tab:inst), used only in Part D
V1 = dict(logN=17, degb=7, log_Delta=53,
          log_keps=39.548,    # kappa * eps_D
          log_dapx=41.718,    # delta'_approx
          log_Bsd=12.585, log_BStC=35.585,
          headroom=1.992)

rng = np.random.default_rng(SEED)
FAILS = []


def gate(name, ok, detail=""):
    print(f"    [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILS.append(name)


def galois(N):
    n = N // 2
    g = np.empty(n, dtype=np.int64)
    x = 1
    for j in range(n):
        g[j] = x
        x = (x * 5) % (2 * N)
    return g


def V_full(N):
    g = galois(N)
    k = np.arange(N, dtype=np.int64)
    return np.exp(1j * np.pi * (np.outer(g, k) % (2 * N)) / N)


def V0_mat(N):
    n = N // 2
    g = galois(N)
    k = np.arange(n, dtype=np.int64)
    return np.exp(1j * np.pi * (np.outer(g, k) % (2 * N)) / N)


def rho(M):
    return float(np.max(np.sum(np.abs(M), axis=1)))


def real_channel_sup(row):
    """sup over real a0, a1 in [-1,1]^n of |sum_k row_k (a0_k + i a1_k)|
       = max_theta sum_k |Re(e^{-i th} row_k)| + |Im(e^{-i th} row_k)|,
       which is pi/2-periodic in theta. Grid value = lower bound on the sup."""
    th = np.linspace(0.0, np.pi / 2, THETA_GRID, endpoint=False)
    best_v, best_t = -1.0, 0.0
    for chunk in np.array_split(th, 8):
        R = np.exp(-1j * chunk)[:, None] * row[None, :]
        vals = np.sum(np.abs(R.real) + np.abs(R.imag), axis=1)
        i = int(np.argmax(vals))
        if vals[i] > best_v:
            best_v, best_t = float(vals[i]), float(chunk[i])
    return best_v, best_t


def adversarial_real(row, theta):
    """Real sign vectors (a0, a1) attaining the grid value for this row."""
    r = np.exp(-1j * theta) * row
    return np.sign(r.real), np.sign(-r.imag)


# ---------------------------------------------------------------------------
print("=" * 78)
print("check_stc_amplification.py   N = 2^%d .. 2^%d   seed %d"
      % (LOGN_LIST[0], LOGN_LIST[-1], SEED))
print("=" * 78)

rows_B = []
rows_C = []

for logN in LOGN_LIST:
    N = 1 << logN
    n = N // 2
    print(f"\n--- N = 2^{logN}  (n = {n}) ---")
    V0 = V0_mat(N)

    # ---------------- Part A: gates ----------------
    print("  Part A (gates)")
    if logN <= DECODE_GATE_MAX_LOGN:
        V = V_full(N)
        d1 = float(np.max(np.abs(V[:, n:] - 1j * V0)))
        gate("V[:, n:] == i*V0", d1 < 1e-9, f"max dev {d1:.2e}")
        m = rng.uniform(-1, 1, N)
        d3 = float(np.max(np.abs(V @ m - V0 @ (m[:n] + 1j * m[n:]))))
        gate("decode V m == V0 (m_lo + i m_hi)", d3 < 1e-9 * n, f"max dev {d3:.2e}")
        del V
    G = V0 @ V0.conj().T
    d2 = float(np.max(np.abs(G - n * np.eye(n)))) / n
    gate("V0 V0^* == n I", d2 < 1e-9, f"rel dev {d2:.2e}")
    del G

    # ---------------- Part B: norms ----------------
    print("  Part B (norms)")
    rho_cts = rho(V0.conj().T / n)
    rho_v0 = rho(V0)
    rho_stc_cplx = rho(np.hstack([V0, 1j * V0]))
    rows = [0] + list(rng.choice(np.arange(1, n), size=min(ROWS_REAL - 1, n - 1),
                                 replace=False))
    best = (-1.0, 0, 0.0)
    for j in rows:
        v, t = real_channel_sup(V0[j])
        if v > best[0]:
            best = (v, j, t)
    c_real = best[0] / n
    a0, a1 = adversarial_real(V0[best[1]], best[2])
    attained = float(np.abs((V0 @ (a0 + 1j * a1))[best[1]]))
    # |S| >= Re(e^{-i th} S) = formula by construction, so attained >= formula;
    # the grid formula is only a lower bound on the true sup.
    gate("adversarial real vector attains at least the sup formula",
         attained >= best[0] - 1e-8 * n,
         f"|StC(a)|_slot = {attained:.6f} vs formula {best[0]:.6f}")
    ar0 = rng.uniform(-1, 1, n)
    ar1 = rng.uniform(-1, 1, n)
    rand_ratio = float(np.max(np.abs(V0 @ (ar0 + 1j * ar1))))
    print(f"    rho(CtS = V0^*/n)          = {rho_cts:.6f}")
    print(f"    rho(V0)                    = {rho_v0:.6f}   (n = {n})")
    print(f"    rho([V0, iV0]) (cplx err)  = {rho_stc_cplx:.6f}   (N = {N})")
    print(f"    real-channel sup / n       = {c_real:.6f}   (4/pi = {4/math.pi:.6f})")
    print(f"    random err |StC a|inf      = {rand_ratio:.3f}  = {rand_ratio/math.sqrt(n):.3f} sqrt(n)")
    rows_B.append((logN, n, rho_cts, rho_v0, rho_stc_cplx, c_real, rand_ratio))

    # ---------------- Part C: end-to-end ideal pipeline ----------------
    print("  Part C (ideal pipeline, per-channel error injected at EvalMod output)")
    Delta = 2.0 ** 20
    degb = 7
    q0 = Delta * 2 ** degb
    K = 4
    m = rng.uniform(-Delta, Delta, N)
    I = rng.integers(-K, K + 1, N)
    x = m + q0 * I
    z_in = V0 @ (x[:n] + 1j * x[n:]) / Delta                 # decoded input
    w = (V0.conj().T @ z_in) / n * (Delta / q0)              # CtS with 1/n, Delta/q0
    t0, t1 = w.real, w.imag
    dct = max(float(np.max(np.abs(t0 - x[:n] / q0))),
              float(np.max(np.abs(t1 - x[n:] / q0))))
    gate("CtS channels == x/q0", dct < 1e-9, f"max dev {dct:.2e}")
    y0 = t0 - np.round(t0)                                   # ideal EvalMod = g
    y1 = t1 - np.round(t1)
    dem = max(float(np.max(np.abs(y0 - m[:n] / q0))),
              float(np.max(np.abs(y1 - m[n:] / q0))))
    gate("ideal EvalMod channels == m/q0", dem < 1e-9, f"max dev {dem:.2e}")
    target = V0 @ (m[:n] + 1j * m[n:]) / Delta

    def pipeline_err(e0, e1):
        u = (y0 + e0) + 1j * (y1 + e1)        # recombination Add
        s = (V0 @ u) * (2 ** degb)            # StC, then back gate q0/Delta
        return float(np.max(np.abs(s - target)))

    E0 = pipeline_err(0.0, 0.0)
    eps_ch = 1e-6
    gate("no-injection baseline << injected error", E0 < 1e-3 * eps_ch * 2 ** degb,
         f"E0 = {E0:.2e}")
    R_adv = (pipeline_err(eps_ch * a0, eps_ch * a1) - E0) / (eps_ch * 2 ** degb)
    R_rand = (pipeline_err(eps_ch * ar0, eps_ch * ar1) - E0) / (eps_ch * 2 ** degb)
    print(f"    R_adv  = {R_adv:.4f}   R_adv/n = {R_adv/n:.4f}   (the reference audit model predicts R = 2)")
    print(f"    R_rand = {R_rand:.4f}   R_rand/sqrt(n) = {R_rand/math.sqrt(n):.4f}")
    rows_C.append((logN, n, R_adv, R_rand))
    del V0

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("SUMMARY B: rho(CtS) | rho(V0)/n | rho([V0,iV0])/N | real-sup/n | rand/sqrt(n)")
for (logN, n, rc, rv, rs, cr, rr) in rows_B:
    print(f"  2^{logN:<3d} {rc:9.6f} {rv/n:10.6f} {rs/(2*n):14.6f} {cr:11.6f} {rr/math.sqrt(n):12.4f}")
print("SUMMARY C: R_adv | R_adv/n | log2(R_adv/2) = bits the reference audit misses at this N")
for (logN, n, ra, rr) in rows_C:
    print(f"  2^{logN:<3d} {ra:12.4f} {ra/n:9.4f} {math.log2(max(ra, 1e-300)/2):9.3f}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("Part D: the reference audit headroom at N = 2^17 with ONLY the StC factor changed")
print("        (arithmetic on the reference audit constants; units of the local term as the reference audit states)")
n17 = 2 ** (V1["logN"] - 1)
loc = 2 ** V1["log_keps"] + 2 ** V1["log_dapx"]
extra = 2 ** V1["log_Bsd"] + 2 ** V1["log_BStC"]
half_delta = 2 ** (V1["log_Delta"] - 1)


def headroom(ratio_vs_ver1):
    tau = ratio_vs_ver1 * 2 ** (V1["degb"] + 1) * loc + extra
    return math.log2(half_delta) - math.log2(tau)


h0 = headroom(1.0)
gate("reproduce the reference audit headroom +1.992", abs(h0 - V1["headroom"]) < 0.01, f"got {h0:+.3f}")
c_last = rows_B[-1][5]
variants = [
    ("the reference audit as written (recomb x2, StC x1)", 1.0),
    ("the reference audit Table rule: recomb x2, then LinTrans rho(V0)=n", float(n17)),
    (f"tight, real channel errors: c_real*n / 2  (c_real={c_last:.4f} from 2^{LOGN_LIST[-1]})",
     c_last * n17 / 2),
    ("tight, complex channel errors: N / 2", float(n17)),
]
for name, r in variants:
    print(f"    {name:70s} ratio 2^{math.log2(r):6.2f}  headroom {headroom(r):+7.2f} bits")
print("    NOTE: B_StC is kept at the reference audit's value; StC's own injected constants would")
print("          also be amplified by later StC levels, so these headrooms are upper bounds.")

print("\n" + "=" * 78)
print("GATE FAILURES:", FAILS if FAILS else "none")
