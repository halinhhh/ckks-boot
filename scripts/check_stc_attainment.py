#!/usr/bin/env python3
"""
check_stc_attainment.py  --  Step 1 (direction a): can admitted remainders realise the StC factor?

QUESTION
  Lemma stc charges c(n) n per-channel error at StC; Prop. perchannel says a per-channel tag
  cannot charge less. Here we compute what ADMITTED WITNESSES can actually produce through one
  concrete mechanism: the relaxed remainders of one EvalMod rescale, chosen in both streams.

MODEL (ideal pipeline, full packing, N real coefficients, n = N/2 slots; exact linear algebra)
  A relaxed remainder of the rescale in stream b injects the polynomial
        e_b = u_{b,0} + s * u_{b,1}          (units: polynomial units at the EvalMod scale)
  where u_{b,c} in R^N are the per-coefficient remainder shifts, s = sparse ternary key (weight H).
  The admitted range is |r/p| <= 3/2, i.e. honest fraction (|x| <= 1/2) + shift in {-1,0,1}.
  We work in the UNIT BOX u in [-1,1]^dim (the common factor 3/2 cancels in every ratio).
  Channel k of a stream lives in EvalMod slot pi(k), with root zeta^{g_pi(k)}, g = 5^j mod 2N.
  After EvalMod: recombination y0 + i y1, back gate (common factor, dropped), StC = V0,
  V0[j,k] = zeta^{g_j k}. Decoded error at slot j is the real-linear functional
        M_j(u) = sum_i u_i m_{j,i}  (complex),
  and the exact maximum over the box is the dual norm
        A_j = max_u |M_j(u)| = max_theta || Re(e^{-i theta} m_j) ||_1       (computed exactly).

  Placements of the injection (mechanism):
    post  after the per-stream real-part extraction (the LAST EvalMod rescale in the current
          design: the extraction's conjugation switch sits before that rescale). Channel error
          sigma_k(e_b) is complex.
    pre   before the extraction, nothing else in between: channel error Re sigma_k(e_b).
    prod  before the last double-angle product, then extraction: channel error
          d_k Re sigma_k(e_b), d_k = derivative of the last double-angle step at the ideal
          operand, normalised to |d_k| = 1. Its sign is (-1)^{I_k} (wrap-integer parity of
          that channel), secret-dependent: modelled as uniform random signs, independent per
          channel and stream.
          prod-exist : max over u for the realised signs (existence of a bad witness)
          prod-obliv : u optimal for d = +1 (what an adversary without the parities would
                       pick), evaluated at the realised signs.
  Components:
    u0     shifts of the c0 component only (s-oblivious; per-stream tag eta = N)
    u0+u1  both components, s known (existence; eta = N (1 + H))

  Certificate at the same unit box (per-stream per-channel tag eta):
    pre, prod : c(n) n eta          (Lemma stc (ii), real channel errors)
    post      : 2 n eta             (Lemma stc (i) on the recombined vector, complex errors)
                also printed: c(n) n eta = what the port charges for this term (INFO only)
  Cauchy-Schwarz bound: A_j <= sqrt(dim) ||m_j||_2   (dim = 2N for u0, 4N for u0+u1).
  Random-shift level  : ||m_j||_2 = rms |M_j(u)| for iid uniform +-1 shifts (honest-like scale).

SLOT ORDER
  natural : channel k in slot k.   bitrev : channel k in slot bitrev(k).
  Which one the construction uses is a declaration; both are reported.

USAGE
  python3 check_stc_attainment.py               # full run, N = 2^6 .. 2^12 (a few minutes)
  python3 check_stc_attainment.py --nmax 11     # smaller
  python3 check_stc_attainment.py --gates-only  # gates only
  The summary block is printed between SUMMARY BEGIN and SUMMARY END.
Pure numpy. No homomorphic library.
"""
import argparse
import itertools
import math
import sys

import numpy as np

FAILS = []


def gate(name, ok, detail=""):
    print(f"    [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILS.append(name)


def bits(x):
    return math.log2(x) if x > 0 else float("-inf")


def c_of(n):
    return 1.0 / (n * math.sin(math.pi / (4 * n)))


# ----------------------------------------------------------------- ring / transforms
def gens(N):
    n = N // 2
    return np.array([pow(5, j, 2 * N) for j in range(n)], dtype=np.int64)


def bitrev(n):
    b = int(round(math.log2(n)))
    return np.array([int(format(k, f"0{b}b")[::-1], 2) if b > 0 else 0 for k in range(n)], dtype=np.int64)


def zpow(e, N):
    return np.exp(1j * math.pi * (np.asarray(e) % (2 * N)) / N)


def V0_rows(N, rows):
    n = N // 2
    g = gens(N)
    k = np.arange(n, dtype=np.int64)
    return zpow(np.outer(g[rows], k), N)                      # R x n


def Zmat(N, order):
    """Z[k, i] = zeta^{g_pi(k) i}: slot value of X^i in the slot that carries channel k."""
    n = N // 2
    g = gens(N)
    pi = np.arange(n) if order == "natural" else bitrev(n)
    i = np.arange(N, dtype=np.int64)
    return zpow(np.outer(g[pi], i), N)                          # n x N


def sigma_slots(poly, N, order):
    """slot values sigma_{pi(k)}(poly) for channels k."""
    return Zmat(N, order) @ poly


# ----------------------------------------------------------------- exact dual norm
def dual_max(m):
    """max over u in [-1,1]^dim of |sum_i u_i m_i| = max_theta sum_i |Re(e^{-i theta} m_i)|.
    Exact: the maximum equals max over the sign patterns of the arcs between breakpoints of
    |sum_i s_i m_i|. Returns (value, theta*, sign vector)."""
    m = np.asarray(m, dtype=complex).ravel()
    r = np.abs(m)
    keep = r > 1e-300
    mk = m[keep]
    phi = np.angle(mk)
    th0 = 1e-7
    s0 = np.sign(np.cos(th0 - phi))
    s0[s0 == 0] = 1.0
    psi = np.mod(phi + math.pi / 2 - th0, math.pi)             # breakpoint offsets in (0, pi]
    order = np.argsort(psi, kind="stable")
    S0 = np.sum(s0 * mk)
    deltas = -2.0 * s0[order] * mk[order]
    S = np.concatenate([[S0], S0 + np.cumsum(deltas)])
    t = int(np.argmax(np.abs(S)))
    s = s0.copy()
    s[order[:t]] *= -1
    full = np.zeros(m.shape[0])
    full[keep] = s
    return float(np.abs(S[t])), float(np.angle(S[t])), full


def grid_max(m, K=20000):
    th = np.linspace(0, math.pi, K, endpoint=False)
    return float(np.max(np.abs(np.real(np.exp(-1j * th)[:, None] * m[None, :])).sum(axis=1)))


# ----------------------------------------------------------------- functionals
def functionals(N, order, rows, s_key, d0, d1):
    """Return dict mech -> comp -> list over rows of complex functional vectors m_j.
    Coordinate layout: [stream0 u0 (N), stream1 u0 (N)] for 'u0',
                       [stream0 u0, stream0 u1, stream1 u0, stream1 u1] for 'u0+u1'."""
    Z = Zmat(N, order)                                          # n x N
    ss = Z @ s_key                                              # sigma(s) per channel (n,)
    W = V0_rows(N, rows)                                        # R x n
    out = {}
    # channel responses C[k, i] for u0 and u1, per mechanism (real-linear in u)
    C0_post, C1_post = Z, ss[:, None] * Z
    C0_re, C1_re = np.real(Z), np.real(ss[:, None] * Z)
    for mech in ("post", "pre", "prod"):
        res = {"u0": [], "u0+u1": []}
        if mech == "post":
            A0 = (W @ C0_post, W @ C0_post)
            A1 = (W @ C1_post, W @ C1_post)
        elif mech == "pre":
            A0 = (W @ C0_re, W @ C0_re)
            A1 = (W @ C1_re, W @ C1_re)
        else:
            A0 = ((W * d0) @ C0_re, (W * d1) @ C0_re)
            A1 = ((W * d0) @ C1_re, (W * d1) @ C1_re)
        for r in range(len(rows)):
            s0u0, s1u0 = A0[0][r], 1j * A0[1][r]                 # stream 1 enters as i * y1
            s0u1, s1u1 = A1[0][r], 1j * A1[1][r]
            res["u0"].append(np.concatenate([s0u0, s1u0]))
            res["u0+u1"].append(np.concatenate([s0u0, s0u1, s1u0, s1u1]))
        out[mech] = res
    return out


def sparse_key(N, H, rng):
    s = np.zeros(N)
    idx = rng.choice(N, H, replace=False)
    s[idx[: H // 2]] = 1
    s[idx[H // 2:]] = -1
    return s


# ================================================================= main
ap = argparse.ArgumentParser()
ap.add_argument("--nmin", type=int, default=6)
ap.add_argument("--nmax", type=int, default=12)
ap.add_argument("--rows", type=int, default=8, help="sampled rows j when N > 2^8 (all rows for N <= 2^8)")
ap.add_argument("--seed", type=int, default=20261001)
ap.add_argument("--gates-only", action="store_true")
ap.add_argument("--smoke", action="store_true", help="run the sweep at N=2^6..2^7 but print no numbers")
a = ap.parse_args()

print("=" * 100)
print("check_stc_attainment.py   Step 1 (direction a): StC factor vs admitted remainder shifts")
print("=" * 100)

# ----------------------------------------------------------------- GATES
print("\nGATES")
rng = np.random.default_rng(a.seed)
for N in (2 ** 6, 2 ** 8, 2 ** 10):
    n = N // 2
    V = V0_rows(N, np.arange(n))
    gate(f"G1 V0 V0* = n I at N=2^{int(math.log2(N))}",
         np.allclose(V @ V.conj().T, n * np.eye(n), atol=1e-8 * n))
N = 2 ** 6
n = N // 2
mpoly = rng.standard_normal(N)
dec = zpow(np.outer(gens(N), np.arange(N)), N) @ mpoly                    # sigma_{g_j}(m)
rec = V0_rows(N, np.arange(n)) @ (mpoly[:n] + 1j * mpoly[n:])
gate("G2 decode identity  sigma(m)_j = (V0 (m_lo + i m_hi))_j", np.allclose(dec, rec, atol=1e-9))
for order in ("natural", "bitrev"):
    z = sigma_slots(mpoly, N, order)
    pi = np.arange(n) if order == "natural" else bitrev(n)
    gate(f"G2b channel slots ({order})", np.allclose(z, dec[pi], atol=1e-9))
# G3: exact dual norm vs vertex enumeration (N = 8, u0, dim 16)
N8 = 8
key8 = sparse_key(N8, 2, rng)
F8 = functionals(N8, "natural", np.arange(N8 // 2), key8, np.ones(N8 // 2), np.ones(N8 // 2))
worst = 0.0
verts = np.array(list(itertools.product((-1.0, 1.0), repeat=2 * N8)))
for mech in ("post", "pre"):
    for m in F8[mech]["u0"]:
        ex = float(np.max(np.abs(verts @ m)))
        worst = max(worst, abs(dual_max(m)[0] - ex) / ex)
gate("G3 dual_max = vertex enumeration (N=8, post/pre, u0, all rows)", worst < 1e-10, f"max rel diff {worst:.1e}")
# G4: dual_max vs dense theta grid, and the returned sign vector attains the value
N = 2 ** 6
key = sparse_key(N, 16, rng)
F = functionals(N, "bitrev", np.array([0, 3]), key, rng.choice([-1.0, 1.0], N // 2), rng.choice([-1.0, 1.0], N // 2))
ok4, ok4b = True, True
for mech in F:
    for comp in F[mech]:
        for m in F[mech][comp]:
            v, _, sgn = dual_max(m)
            gm = grid_max(m)
            ok4 &= (gm <= v * (1 + 1e-12)) and (v - gm) / v < 1e-4
            ok4b &= abs(abs(np.dot(sgn, m)) - v) / v < 1e-10
gate("G4 dual_max >= theta-grid max, gap < 1e-4 (N=2^6, all mechanisms)", ok4)
gate("G4b returned sign vector attains dual_max", ok4b)
# G5: Lemma stc reproduction: max over real a0, a1 in [-1,1]^n of |(V0 (a0 + i a1))_j| = c(n) n
ok5, det5 = True, []
for N in (2 ** 4, 2 ** 6, 2 ** 8, 2 ** 10):
    n = N // 2
    for j in (0, n // 3, n - 1):
        row = V0_rows(N, np.array([j]))[0]
        v = dual_max(np.concatenate([row, 1j * row]))[0]
        rel = abs(v / (c_of(n) * n) - 1)
        ok5 &= rel < 1e-9
    det5.append(f"2^{int(math.log2(N))}:{rel:.0e}")
gate("G5 Lemma stc (ii) reproduced: dual_max = c(n) n", ok5, " ".join(det5))
if FAILS:
    print("\n  GATE FAILED:", FAILS, " -- see the output above.")
    sys.exit(1)
print("  GATES PASSED.")
if a.gates_only:
    sys.exit(0)

# ----------------------------------------------------------------- SWEEP
NS = [2 ** 6, 2 ** 7] if a.smoke else [2 ** k for k in range(a.nmin, a.nmax + 1)]
ROWS_ALL_MAX = 2 ** 8
REC = []          # (N, order, mech, comp, H, A_max, A_min, l2_max, cert, cert_port, cs)
OBL = []          # (N, order, comp, A_exist, A_obliv)
print("\nSWEEP  (G6 bound checks run inside; any FAIL stops the script)")
for N in NS:
    n = N // 2
    H = 64 if N >= 256 else N // 4
    rng_n = np.random.default_rng(a.seed + N)
    key = sparse_key(N, H, rng_n)
    d0 = rng_n.choice([-1.0, 1.0], n)
    d1 = rng_n.choice([-1.0, 1.0], n)
    if N <= ROWS_ALL_MAX:
        rows = np.arange(n)
    else:
        extra = rng_n.choice(np.arange(2, n - 1), max(0, a.rows - 4), replace=False)
        rows = np.unique(np.concatenate([[0, 1, n // 2, n - 1], extra]))
    for order in ("natural", "bitrev"):
        F = functionals(N, order, rows, key, d0, d1)
        # oblivious adversary for prod: u optimal for d = +1 (= the 'pre' optimum), evaluated at d
        for comp in ("u0", "u0+u1"):
            eta = N * (1 + (H if comp == "u0+u1" else 0))
            ex_o, ob_o = 0.0, 0.0
            for r in range(len(rows)):
                _, _, sgn = dual_max(F["pre"][comp][r])
                ob_o = max(ob_o, abs(np.dot(sgn, F["prod"][comp][r])))
                ex_o = max(ex_o, dual_max(F["prod"][comp][r])[0])
            OBL.append((N, order, comp, ex_o, ob_o))
            for mech in ("post", "pre", "prod"):
                vals = [dual_max(m)[0] for m in F[mech][comp]]
                l2 = [float(np.linalg.norm(m)) for m in F[mech][comp]]
                dim = F[mech][comp][0].shape[0]
                cs = math.sqrt(dim) * max(l2)
                cert = (2 * n if mech == "post" else c_of(n) * n) * eta
                cert_port = c_of(n) * n * eta
                A = max(vals)
                if not (A <= cs * (1 + 1e-9) and A <= cert * (1 + 1e-9)):
                    gate(f"G6 bounds N=2^{int(math.log2(N))} {order} {mech} {comp}", False,
                         f"A={A:.4g} cs={cs:.4g} cert={cert:.4g}")
                REC.append((N, order, mech, comp, H, A, min(vals), max(l2), cert, cert_port, cs))
    print(f"    N=2^{int(math.log2(N))}: done ({len(rows)} rows)", flush=True)
if FAILS:
    print("\n  BOUND CHECK FAILED:", FAILS, " -- see the output above.")
    sys.exit(1)
print("  G6 bounds hold everywhere.")
if a.smoke:
    print("  SMOKE OK (numbers suppressed).")
    sys.exit(0)


def slope(xs, ys):
    X = np.log2(np.array(xs, dtype=float))
    Y = np.log2(np.array(ys, dtype=float))
    return float(np.polyfit(X, Y, 1)[0])


# ----------------------------------------------------------------- report
print("\n" + "=" * 100)
print("SUMMARY BEGIN")
print(f"seed={a.seed} N=2^{a.nmin}..2^{a.nmax} rows: all for N<=2^8, else {a.rows} sampled; H=64 (N>=256) else N/4")
print("columns: log2 A_max (min over rows) | log2(cert/A) | log2(cert_port/A) [post only] | "
      "log2(cs/A) | A/(N sqrt n) | log2 ||m||_2")
for order in ("natural", "bitrev"):
    for comp in ("u0", "u0+u1"):
        for mech in ("post", "pre", "prod"):
            rs = [r for r in REC if r[1] == order and r[2] == mech and r[3] == comp]
            for (N, _, _, _, H, A, Amin, l2, cert, cport, cs) in rs:
                n = N // 2
                port = f"{bits(cport / A):+6.2f}" if mech == "post" else "   -  "
                print(f"ATT {order:>7} {comp:>5} {mech:>4} N=2^{int(math.log2(N)):<2} H={H:<3}| "
                      f"A 2^{bits(A):7.3f} (min 2^{bits(Amin):7.3f}) | cert/A {bits(cert / A):6.2f} | "
                      f"port/A {port} | cs/A {bits(cs / A):5.2f} | A/(N sqrt n) {A / (N * math.sqrt(n)):6.3f} | "
                      f"l2 2^{bits(l2):6.2f}")
            Ns = [r[0] for r in rs]
            print(f"FIT {order:>7} {comp:>5} {mech:>4}: slope log2A/log2N = {slope(Ns, [r[5] for r in rs]):.3f}   "
                  f"(cert slope 2 for u0; CS slope 1.5)")
for (N, order, comp, ex_o, ob_o) in OBL:
    print(f"OBL {order:>7} {comp:>5} N=2^{int(math.log2(N)):<2}: prod-exist 2^{bits(ex_o):.3f}  "
          f"prod-obliv 2^{bits(ob_o):.3f}  (exist - obliv = {bits(ex_o) - bits(ob_o):.2f} bits)")
print("GATE FAILURES:", FAILS if FAILS else "none")
print("SUMMARY END")
print("=" * 100)
print("""
READING IT
  - A is the exact worst case over the unit box of remainder shifts for this ONE mechanism;
    admitted witnesses reach between A/2 and 3A/2 once the honest fractions are included, and
    the certificate at the same box is cert (the 3/2 cancels).
  - cert/A is the part of the StC charge that this mechanism cannot realise.
  - post: 'port/A' < 0 would mean the port's c(n) n charge on a post-extraction injection is
    below what the shifts can produce, i.e. that charge is unsound for that term.
  - prod-obliv vs prod-exist: what the wrap-parity signs cost an adversary who does not know them.
  - This is one injection mechanism, not all admitted witnesses.
""")
