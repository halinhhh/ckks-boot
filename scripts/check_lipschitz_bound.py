#!/usr/bin/env python3
"""
check_lipschitz_bound.py  --  analytic bound on sup |p'| of the EvalMod approximant vs grid maxima

WHAT IS BOUNDED
  p(t) = Q( P( f(t) ) ),  t in [-1, 1], where
    f   = the base-tier Chebyshev polynomial of degree d = 18 that the port uses (discrete
          Chebyshev coefficients on M = 2048 first-kind nodes) of
          f0(t) = A cos(b t - phi),  A = (2 pi)^(-2^-l),  b = 2 pi K_eff / 2^l,  phi = pi / 2^(l+1)
    P   = the l = 6 double-angle steps v -> 2 v^2 - A_{i-1}^2.  IDENTITY (gate A0):
          P(v) = A_0 T_m(v / A),  m = 2^l,  A_0 = 1/(2 pi)
    Q   = identity (arcsine off) or y -> y + a y^3, a = (2 pi)^2 / 6 (arcsine on)

ANALYTIC BOUND (Lemma sens-real of the paper; every step is a triangle inequality)
  Write f = f0 + r, g = (Q o P)'. Then p' = g(f) f' and p0' := g(f0) f0' is the derivative of
  Q(sin(2 pi K_eff t)/(2 pi)), so |p0'| <= K_eff (with the arcsine tier (1 + s^2/2) sqrt(1-s^2) <= 1).
    |p'| <= |p0'| + |g(f) - g(f0)| |f0'| + |g(f)| |r'|
         <= K_eff + G2 * E0 * (A b) + G1 * E1                                        (L_an)
  with, for eps = E0 / A and |x| <= 1 + eps (Chebyshev derivatives are maximal at the endpoint):
    G1 = Qp * (A_0 / A) * T_m'(1 + eps)                                  >= sup |g|
    G2 = 6 a Ymax (A_0/A)^2 T_m'(1+eps)^2 + Qp (A_0/A^2) T_m''(1+eps)     >= sup |g'|
    Ymax = A_0 T_m(1 + eps),  Qp = 1 + 3 a Ymax^2 (arcsine on) or 1 (off)
  E0 >= sup |r|, E1 >= sup |r'|, E2 >= sup |r''| from the Bernstein-ellipse bound of Lemma approx:
    |c_j| <= 2 S(rho) rho^-j,  S(rho) = A cosh(b (rho - 1/rho) / 2)
    tail:  sum_{j>d} |c_j| * {1, j^2, j^2 (j^2 - 1)/3}      (|T_j^(k)| <= T_j^(k)(1), V. A. Markov)
    alias: discrete coefficients differ from c_j by sum_{k>=1} (|c_{2kM-j}| + |c_{2kM+j}|),
           bounded by 4 S rho^(j - 2M) / (1 - rho^-2M); reported in log2 (it underflows).
  Each of E0, E1, E2 is minimised over rho in [1.5, 40] separately (every rho gives a valid bound).
  Second derivative (for the complex-disk / second-order correction of PART J):
    L2_an = G2 (A b + E1)^2 + G1 (A b^2 + E2)  >= sup |p''|   (crude: the chain is not re-grouped)

GATES (script stops on any FAIL)
  A0  P(v) = A_0 T_m(v/A) on a grid                       A1  grid max |p0'| = K_eff (<=, and attained)
  A2  grid sup|r| <= E0, sup|r'| <= E1, sup|r''| <= E2      A3  grid max |p'| <= L_an   (THE CHECK)
  A4  grid max |p''| <= L2_an                               A5  E0 at the ref cell reproduces the port's E

K_eff = K + (Delta + B_in)/q0 ~ 32 + 2^-degb: swept at 32 + 2^-d, d = 2..14, both arcsine settings,
plus the exact ref-cell value. Pure numpy, double precision; a final certificate would use outward
rounding (as in E3). The summary block is printed between SUMMARY BEGIN and SUMMARY END.
"""
import math
import sys

import numpy as np
from numpy.polynomial import chebyshev as C

TWOPI = 2 * math.pi
K, ELL, DB, MNODES = 32, 6, 18, 2048
MDA = 2 ** ELL
A0 = 1 / TWOPI
A = TWOPI ** (-(2.0 ** (-ELL)))
ARC_A = TWOPI ** 2 / 6.0
GRID = np.linspace(-1.0, 1.0, 2 ** 20 + 1)
RHO = np.linspace(1.5, 40.0, 40000)          # same rho grid as the port's lam()
FAILS = []


def gate(name, ok, detail=""):
    print(f"    [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILS.append(name)


def bits(x):
    return math.log2(x) if x > 0 else float("-inf")


def port_coeffs(Keff):
    """Exactly the port's cheb_coeffs(Keff)."""
    th = np.pi * (np.arange(MNODES) + 0.5) / MNODES
    x = np.cos(th)
    f = A * np.cos((TWOPI * Keff * x - np.pi / 2) / 2 ** ELL)
    c = np.array([2 * np.sum(f * np.cos(j * th)) / MNODES for j in range(DB + 1)])
    c[0] /= 2
    return c


def f0_derivs(Keff, t):
    b = TWOPI * Keff / 2 ** ELL
    phi = math.pi / 2 ** (ELL + 1)
    u = b * t - phi
    return A * np.cos(u), -A * b * np.sin(u), -A * b * b * np.cos(u)


def chain(v, dv, ddv, arc):
    """P then Q with first and second derivatives (chain rule)."""
    for i in range(1 - ELL, 1):
        Ai = TWOPI ** (-(2.0 ** (i - 1)))
        v, dv, ddv = 2 * v * v - Ai * Ai, 4 * v * dv, 4 * (dv * dv + v * ddv)
    if arc:
        v, dv, ddv = (v + ARC_A * v ** 3, (1 + 3 * ARC_A * v * v) * dv,
                      6 * ARC_A * v * dv * dv + (1 + 3 * ARC_A * v * v) * ddv)
    return v, dv, ddv


def Tm_derivs_at(x):
    cm = np.zeros(MDA + 1)
    cm[-1] = 1.0
    c1 = C.chebder(cm)
    c2 = C.chebder(c1)
    return float(C.chebval(x, cm)), float(C.chebval(x, c1)), float(C.chebval(x, c2))


def ellipse_bounds(Keff):
    b = TWOPI * Keff / 2 ** ELL
    S = A * np.cosh(b / 2 * (RHO - 1 / RHO))                        # sup_{E_rho} |f0|
    j = np.arange(DB + 1, DB + 1 + 700, dtype=float)
    w = RHO[:, None] ** (-j[None, :])                                 # rho^-j
    E0 = 2 * S * w.sum(axis=1)
    E1 = 2 * S * (w * j ** 2).sum(axis=1)
    E2 = 2 * S * (w * j ** 2 * (j ** 2 - 1) / 3).sum(axis=1)
    i0, i1, i2 = int(np.argmin(E0)), int(np.argmin(E1)), int(np.argmin(E2))
    # aliasing of the discrete coefficients, log2 of sum_{j<=d} j^2 * 4 S rho^(j-2M)/(1-rho^-2M) at rho*_1
    r = RHO[i1]
    jj = np.arange(DB + 1)
    log2_alias = (bits(4 * S[i1]) + (DB - 2 * MNODES) * math.log2(r)
                  + math.log2(float(np.sum((jj ** 2 + 1) * r ** (jj - DB)))))
    alias = 2.0 ** log2_alias if log2_alias > -1000 else 0.0
    return dict(E0=float(E0[i0]) + alias, E1=float(E1[i1]) + alias, E2=float(E2[i2]) + alias,
                rho0=float(RHO[i0]), rho1=float(RHO[i1]), log2_alias=log2_alias, b=b)


def analytic(Keff, arc):
    eb = ellipse_bounds(Keff)
    eps = eb["E0"] / A
    Tm, Tm1, Tm2 = Tm_derivs_at(1 + eps)
    Ymax = A0 * Tm
    Qp = 1 + 3 * ARC_A * Ymax ** 2 if arc else 1.0
    G1 = Qp * (A0 / A) * Tm1
    G2 = (6 * ARC_A * Ymax * ((A0 / A) * Tm1) ** 2 if arc else 0.0) + Qp * (A0 / A ** 2) * Tm2
    b = eb["b"]
    L_an = Keff + G2 * eb["E0"] * (A * b) + G1 * eb["E1"]
    L2_an = G2 * (A * b + eb["E1"]) ** 2 + G1 * (A * b * b + eb["E2"])
    return dict(eb, G1=G1, G2=G2, L_an=L_an, L2_an=L2_an, excess=L_an - Keff)


def grid_values(Keff, arc):
    c = port_coeffs(Keff)
    c1, c2 = C.chebder(c), C.chebder(C.chebder(c))
    f, df, ddf = C.chebval(GRID, c), C.chebval(GRID, c1), C.chebval(GRID, c2)
    g0, dg0, ddg0 = f0_derivs(Keff, GRID)
    _, dp, ddp = chain(f, df, ddf, arc)
    _, dp0, _ = chain(g0, dg0, ddg0, arc)
    return dict(p1=float(np.max(np.abs(dp))), p2=float(np.max(np.abs(ddp))), p01=float(np.max(np.abs(dp0))),
                r0=float(np.max(np.abs(f - g0))), r1=float(np.max(np.abs(df - dg0))),
                r2=float(np.max(np.abs(ddf - ddg0))))


print("=" * 100)
print("check_lipschitz_bound.py   analytic sup|p'| of the EvalMod approximant vs grid maxima")
print("=" * 100)
print("\nGATES")
# A0: double-angle identity
v = np.linspace(-A, A, 4097)
Pv, _, _ = chain(v, np.ones_like(v), np.zeros_like(v), 0)
cm = np.zeros(MDA + 1)
cm[-1] = 1.0
gate("A0 P(v) = A_0 T_m(v/A)", np.allclose(Pv, A0 * C.chebval(v / A, cm), rtol=0, atol=1e-13))

# A5: the port's E at the reference cell (Delta = 2^53, q0 = 2^60, B_in = 2^36.392)
KEFF_REF = K + (2.0 ** 53 + 2.0 ** 36.392) / 2.0 ** 60
eb_ref = ellipse_bounds(KEFF_REF)
b_ref = eb_ref["b"]
Eport = 2 * A * np.cosh(b_ref / 2 * (RHO - 1 / RHO)) * RHO ** (-(DB + 1)) / (1 - 1 / RHO)
gate("A5 E0 tail = port's closed form at the ref cell (E = 4.489e-13)",
     abs(eb_ref["E0"] / float(np.min(Eport)) - 1) < 1e-9 and abs(float(np.min(Eport)) / 4.489e-13 - 1) < 1e-3,
     f"E0 = {eb_ref['E0']:.4e}, port closed form = {float(np.min(Eport)):.4e}")

CASES = [("ref", KEFF_REF, 0)] + [(f"32+2^-{d}", K + 2.0 ** -d, arc) for d in range(2, 15) for arc in (0, 3)]
ROWS = []
okA1, okA2, okA3, okA4 = True, True, True, True
for lab, Keff, arc in CASES:
    an = analytic(Keff, arc)
    gv = grid_values(Keff, arc)
    okA1 &= gv["p01"] <= Keff * (1 + 1e-12) and gv["p01"] >= Keff * (1 - 1e-6)
    okA2 &= gv["r0"] <= an["E0"] and gv["r1"] <= an["E1"] and gv["r2"] <= an["E2"]
    okA3 &= gv["p1"] <= an["L_an"]
    okA4 &= gv["p2"] <= an["L2_an"]
    ROWS.append((lab, Keff, arc, an, gv))
gate("A1 grid max |p0'| = K_eff (never above, attained to 1e-6)", okA1)
gate("A2 grid sup |r|, |r'|, |r''| <= E0, E1, E2 (every case)", okA2)
gate("A3 grid max |p'| <= L_an (every case)  <-- the analytic bound holds", okA3)
gate("A4 grid max |p''| <= L2_an (every case)", okA4)
if FAILS:
    print("\n  GATE FAILED:", FAILS, " -- see the output above.")
    for lab, Keff, arc, an, gv in ROWS:
        print(f"    {lab} arc={arc}: grid p' {gv['p1']:.15g}  L_an {an['L_an']:.15g}  "
              f"grid r {gv['r0']:.3e}/{gv['r1']:.3e}/{gv['r2']:.3e}  E {an['E0']:.3e}/{an['E1']:.3e}/{an['E2']:.3e}")
    sys.exit(1)
print("  GATES PASSED.")

print("\n" + "=" * 100)
print("SUMMARY BEGIN")
print("columns: K_eff | grid max p' | L_an | log2(L_an/K_eff - 1) | log2((L_an - grid)/grid) | "
      "E0, E1 bounds (grid) | G1 | grid p'' vs L2_an | log2 alias")
worst_ex = 0.0
for lab, Keff, arc, an, gv in ROWS:
    ex = an["L_an"] / Keff - 1
    worst_ex = max(worst_ex, ex)
    gap = (an["L_an"] - gv["p1"]) / gv["p1"]
    print(f"LIP {lab:>9} arc={'on ' if arc else 'off'} | K_eff {Keff:.10f} | grid {gv['p1']:.10f} | "
          f"L_an {an['L_an']:.10f} | ex 2^{bits(ex):7.2f} | gap 2^{bits(gap):7.2f} | "
          f"E0 {an['E0']:.3e} ({gv['r0']:.3e}) E1 {an['E1']:.3e} ({gv['r1']:.3e}) | G1 2^{bits(an['G1']):.2f} | "
          f"p'' 2^{bits(gv['p2']):.2f} vs 2^{bits(an['L2_an']):.2f} | alias 2^{an['log2_alias']:.0f}")
print(f"SUMMARY max over cases of L_an/K_eff - 1 = 2^{bits(worst_ex):.2f}  (= {worst_ex:.3e})")
for la in (-25.0, -27.5, -32.0):
    worst_corr = max(2.0 ** la * an["L2_an"] / an["L_an"] for _, _, _, an, _ in ROWS)
    print(f"SUMMARY complex-disk proxy alpha*L2_an/L_an at alpha = 2^{la:g}: 2^{bits(worst_corr):.2f}")
print("GATE FAILURES:", FAILS if FAILS else "none")
print("SUMMARY END")
print("=" * 100)
print("""
READING IT
  - 'ex' is how far the analytic bound sits above K_eff; the model in PART J used the grid max,
    which is K_eff to 4 decimals. If ex is tiny, the port can use L = L_an with no visible change.
  - 'gap' is L_an vs the grid max (the bound must be >= the grid max; A3 checks it).
  - alpha in PART J is B_cts/Delta: about 2^-27.5 at the ref cell and 2^-32 at the optima.
  - L2_an is deliberately crude; the last lines show whether that matters.
""")
