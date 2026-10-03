#!/usr/bin/env python3
"""
check_units_sparse_l2.py  --  one run, the three questions check_stc_amplification.py left open

Pure numpy, ideal (unencrypted) values plus arithmetic on the reference audit's own constants.
No OpenFHE, no DEV flag.

Q1  UNITS.  The reference audit writes tau_e2e = 2^(degb+1) (kappa eps_D + delta'), i.e. BOTH
    terms are multiplied by the back gate 2^degb.  In the reference audit's implementation,
    its EvalMod leg starts T1 at nu = Delta, so eps_D is a noise tag at scale Delta
    (poly units = Delta x channel).  Lemma dapprox defines delta' = q0 * sup|...|,
    i.e. x-units = q0 x channel = 2^degb x (Delta x channel).  If so, the back
    gate must multiply eps_D but NOT delta' (already at the unreduced scale), and
    the reference audit over-charges delta' by 2^degb.  PART A reproduces 2^41.718 from the
    definition and prints which unit it is in.

Q2  IS THE APPROXIMATION TERM REALLY AMPLIFIED BY n?  For slot-bounded messages
    (|z| <= Z) Parseval gives ||m||_2 <= Delta Z, hence
        sum_k |f(theta_k)| <= f(eps Z),   f(t) = t - sin(2 pi t)/(2 pi),
    and |StC(err)|_inf <= sum_k |err_k|.  So the sine-vs-identity term should
    reach the output with factor <= Z^3 <= 1, NOT c*n.  The n factor needs
    coefficient-aligned messages, which have |z| ~ n.  PART B measures this on the
    ideal pipeline (CtS, sine tier, recombination, StC, back gate) and GATES the
    bound ratio <= 1.  A FAIL there means the Parseval argument is wrong.

Q3  WHAT CAN STILL OPEN THE WINDOW?  PART C recomputes the reference audit's tau at N = 2^17
    under cumulative corrections, PART D sweeps sparse packing n_s, PART E inverts
    the budget: how small kappa eps_D must be for a target decoded tolerance.

    Variants
      V1  the reference audit as written
      V2  + StC sup factor c*n_s                                   (= last run)
      V3  + units fix: 2^degb multiplies eps_D only
      V4  + StC's own injected rescale constants amplified by later StC levels
          (sup-norm tags throughout; this is the reference audit's framework made consistent)
      V5  MODEL, NOT MEASURED: l2 slot tags instead of sup tags.
          - every injection constant the reference audit uses is an l1-coefficient bound, and
            ||sigma(e)||_2 <= ||V||_op ||e||_2 <= N ||e||_inf gives the same constant
            in l2 (and ||sigma(s) . sigma(u)||_2 <= ||s||_1 ||sigma(u)||_2 for s u1);
          - slotwise products: ||a . e||_2 <= ||a||_inf ||e||_2, so the reference audit's gate
            recursion holds verbatim with l2 tags and eps_D is unchanged;
          - at StC, Cauchy-Schwarz: |V0 alpha|_inf <= sqrt(n) ||alpha||_2, two
            streams -> 2 sqrt(n_s) instead of c*n_s; StC levels amplify l2 by
            2^(k/2) (butterfly op-norm), not 2^k;
          - the approximation term uses PART B's Parseval factor (<= 1).
          This needs a written lemma before it can go in a paper.

USAGE
  python3 check_units_sparse_l2.py        # PART B up to N = 2^11
  python3 check_units_sparse_l2.py 10     # lighter
Inspect the whole output.
"""
import sys
import math
import numpy as np

LOGN_MAX_B = int(sys.argv[1]) if len(sys.argv) > 1 else 11
THETA_GRID = 2048
SEED = 20260930
rng = np.random.default_rng(SEED)
FAILS = []

# ---------------------------------------------------------------- the reference audit constants
N17 = 2 ** 17
n17 = N17 // 2
DELTA = 2.0 ** 53
DEGB = 7
Q0 = 2.0 ** 60
KEPS = 2.0 ** 39.548          # kappa * eps_D          (tab:inst)
DAPX = 2.0 ** 41.718          # delta'_approx          (tab:inst, Lemma dapprox)
BSD = 2.0 ** 12.585           # B_{s->d}               (tab:inst)
BSTC_FLAT = 2.0 ** 35.585     # B_StC as the reference audit states  (tab:inst)
XI = 2.0 ** 37.504            # Xi at D_C = 3          (tab:inst)
CRS = 3.0 * N17 * (1 + N17 // 2) / 2.0   # kappa_rs B_sc, dense key (StC runs dense)
STC_PART = [3, 4, 4, 5]       # r_StC = 4, the reference audit's partition (execution order not stated)
C_PREV = 1.273239             # real-channel sup / n at N = 2^12, previous run
HALF = DELTA / 2


def bits(x):
    return math.log2(x) if x > 0 else float("-inf")


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


def V0_mat(N):
    n = N // 2
    g = galois(N)
    k = np.arange(n, dtype=np.int64)
    return np.exp(1j * np.pi * (np.outer(g, k) % (2 * N)) / N)


def real_channel_sup(row):
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
    r = np.exp(-1j * theta) * row
    return np.sign(r.real), np.sign(-r.imag)


_C = {}


def c_of(ns):
    """real-channel sup / n_s for an n_s-slot StC (ring of dimension 2 n_s)."""
    if ns in _C:
        return _C[ns]
    if ns > 2 ** 11:
        _C[ns] = C_PREV
        return C_PREV
    V0 = V0_mat(2 * ns)
    if ns <= 16:
        rows = range(ns)
    else:
        rows = [0] + list(rng.choice(np.arange(1, ns), size=7, replace=False))
    _C[ns] = max(real_channel_sup(V0[j])[0] for j in rows) / ns
    return _C[ns]


def partition(ns):
    if ns == n17:
        return list(STC_PART)
    L = int(round(math.log2(ns)))
    if L == 0:
        return []
    r = math.ceil(L / 4)
    base, extra = divmod(L, r)
    return [base + 1] * extra + [base] * (r - extra)      # larger merges first


def amp_inj(P, half=False):
    """sum_i prod_{j>i} rho_j : each level's injected constant, amplified by later levels."""
    tot = 0.0
    for i in range(len(P)):
        e = sum(P[i + 1:])
        tot += 2.0 ** (e / 2 if half else e)
    return tot


def amp_after_first(P, half=False):
    if not P:
        return 1.0
    e = sum(P[1:])
    return 2.0 ** (e / 2 if half else e)


ZFAC = 1.0   # Parseval factor for the approximation term; confirmed (or not) by PART B


def tau_terms(ns, variant, P=None):
    """Returns dict of named terms (poly units at Delta after the back gate) and their sum."""
    P = partition(ns) if P is None else P
    A_sup = c_of(ns) * ns
    A_l2 = 2.0 * math.sqrt(ns)
    b = 2.0 ** DEGB
    if variant == "V1":
        t = {"noise": 2 * b * KEPS, "approx": 2 * b * DAPX, "StC inj": BSTC_FLAT, "B_sd": BSD}
    elif variant == "V2":
        t = {"noise": A_sup * b * KEPS, "approx": A_sup * b * DAPX, "StC inj": BSTC_FLAT, "B_sd": BSD}
    elif variant == "V3":
        t = {"noise": A_sup * b * KEPS, "approx": A_sup * DAPX, "StC inj": BSTC_FLAT, "B_sd": BSD}
    elif variant == "V4":
        t = {"noise": A_sup * b * KEPS, "approx": A_sup * DAPX,
             "StC inj": CRS * amp_inj(P), "B_sd": BSD * amp_after_first(P)}
    elif variant == "V5":
        t = {"noise": A_l2 * b * KEPS, "approx": ZFAC * DAPX,
             "StC inj": CRS * amp_inj(P, True), "B_sd": BSD * amp_after_first(P, True)}
    else:
        raise ValueError(variant)
    return t, sum(t.values())


def dominant(t):
    k = max(t, key=t.get)
    return f"{k} 2^{bits(t[k]):.2f}"


# =============================================================================
print("=" * 86)
print("check_units_sparse_l2.py   seed %d   PART B up to N = 2^%d" % (SEED, LOGN_MAX_B))
print("=" * 86)

# ---------------------------------------------------------------- PART 0
print("\nPART 0  gates against the reference audit and the previous run")
_, tau1 = tau_terms(n17, "V1")
gate("V1 reproduces the reference audit headroom +1.992", abs(bits(HALF) - bits(tau1) - 1.992) < 0.01,
     f"got {bits(HALF) - bits(tau1):+.3f}")
gate("4 * kappa_rs B_sc (dense) == the reference audit B_StC 2^35.585", abs(bits(4 * CRS) - 35.585) < 0.01,
     f"got 2^{bits(4 * CRS):.3f}")
_, tau2 = tau_terms(n17, "V2")
gate("V2 reproduces last run's -13.36", abs(bits(HALF) - bits(tau2) + 13.36) < 0.02,
     f"got {bits(HALF) - bits(tau2):+.3f}")

# ---------------------------------------------------------------- PART A
print("\nPART A  units of delta' (Lemma dapprox, sine-vs-identity term)")
theta_max = (DELTA + XI) / Q0
th = np.linspace(0.0, theta_max, 200001)
f = th - np.sin(2 * np.pi * th) / (2 * np.pi)
supf = float(np.max(f))
d_x = Q0 * supf
d_D = DELTA * supf
print(f"    sup |theta - sin(2 pi theta)/2pi| on |theta| <= (Delta+Xi)/q0 : 2^{bits(supf):.3f}  (channel units)")
print(f"    x q0     (x-units, the Lemma's definition)                   : 2^{bits(d_x):.3f}")
print(f"    x Delta  (noise-tag units at the EvalMod scale)               : 2^{bits(d_D):.3f}")
gate("the reference audit's 2^41.718 is the x-unit value", abs(bits(d_x) - 41.718) < 0.01)
print(f"    ratio x-units / Delta-units = 2^{bits(d_x / d_D):.3f} = 2^degb")
print("    eps_D: the reference EvalMod leg seeds T1 with nu = Delta -> tag at scale Delta.")
print("    the reference branch run: t = KAPPA*eD + d_ap ; t = 2*t ; t = 2**deg * t  -> both terms get 2^degb.")

# ---------------------------------------------------------------- PART B
print("\nPART B  approximation term through the ideal pipeline, sine tier, slot-bounded messages")
Dl = 2.0 ** 20
q0s = Dl * 2 ** DEGB
eps = Dl / q0s
dx_small = q0s * (eps - math.sin(2 * math.pi * eps) / (2 * math.pi))


def run_m(V0, n, m, exact=False, K=4):
    N = 2 * n
    I = rng.integers(-K, K + 1, N)
    x = m + q0s * I
    z_in = V0 @ (x[:n] + 1j * x[n:]) / Dl
    wv = (V0.conj().T @ z_in) / n * (Dl / q0s)
    t = np.concatenate([wv.real, wv.imag])
    y = (t - np.round(t)) if exact else np.sin(2 * np.pi * t) / (2 * np.pi)
    s = (V0 @ (y[:n] + 1j * y[n:])) * (q0s / Dl)
    target = V0 @ (m[:n] + 1j * m[n:]) / Dl
    return float(np.max(np.abs(s - target))) * Dl


def m_from_z(V0, n, z):
    w = Dl * (V0.conj().T @ z) / n
    return np.concatenate([w.real, w.imag])


rows_B = []
worst_bounded = 0.0
for logN in range(6, LOGN_MAX_B + 1):
    N = 1 << logN
    n = N // 2
    V0 = V0_mat(N)
    fams = {
        "ones": np.ones(n, dtype=complex),
        "single": np.eye(n, dtype=complex)[0],
        "rand-real": rng.uniform(-1, 1, n).astype(complex),
        "rand-phase": np.exp(1j * rng.uniform(0, 2 * np.pi, n)),
    }
    res = {}
    for name, z in fams.items():
        res[name] = (run_m(V0, n, m_from_z(V0, n, z)) / dx_small, float(np.max(np.abs(z))))
    best = 0.0
    for _ in range(64):
        z = np.exp(1j * rng.uniform(0, 2 * np.pi, n))
        best = max(best, run_m(V0, n, m_from_z(V0, n, z)) / dx_small)
    res["search64"] = (best, 1.0)
    v, t0 = real_channel_sup(V0[0])
    a0, a1 = adversarial_real(V0[0], t0)
    m_al = Dl * np.concatenate([a0, a1])
    z_al = V0 @ (m_al[:n] + 1j * m_al[n:]) / Dl
    res["coef-aligned"] = (run_m(V0, n, m_al) / dx_small, float(np.max(np.abs(z_al))))
    E0 = run_m(V0, n, m_from_z(V0, n, fams["ones"]), exact=True)
    print(f"  N = 2^{logN} (n = {n})")
    gate("exact-g baseline << delta'", E0 < 1e-4 * dx_small, f"E0/delta' = {E0 / dx_small:.2e}")
    for name, (r, zmax) in res.items():
        tag = "(slot-bounded)" if zmax <= 1 + 1e-9 else f"(|z|max = {zmax:.1f})"
        extra = f"  ratio/n = {r / n:.4f}" if name == "coef-aligned" else ""
        print(f"      {name:>12}: E_poly / delta'_x = {r:10.6f}  {tag}{extra}")
        if zmax <= 1 + 1e-9:
            worst_bounded = max(worst_bounded, r)
    bmax = max(r for (r, zmax) in res.values() if zmax <= 1 + 1e-9)
    gate("Parseval: slot-bounded ratio <= 1", bmax <= 1 + 1e-4, f"max = {bmax:.6f}")
    rows_B.append((logN, bmax, res["coef-aligned"][0] / n))
    del V0
ZFAC = max(1.0, worst_bounded)
print(f"    ZFAC used in V5 = {ZFAC:.6f}")

# ---------------------------------------------------------------- PART C
print("\nPART C  the reference audit's tau at N = 2^17, full packing, cumulative corrections")
print(f"    {'var':>4} | {'StC order':>10} | {'tau':>9} | {'headroom':>9} | {'decoded err':>11} | dominant")
print("    " + "-" * 78)
for var in ("V1", "V2", "V3", "V4", "V5"):
    orders = [STC_PART, STC_PART[::-1]] if var in ("V4", "V5") else [STC_PART]
    for P in orders:
        t, tau = tau_terms(n17, var, P)
        print(f"    {var:>4} | {str(P):>10} | 2^{bits(tau):6.2f} | {bits(HALF) - bits(tau):+9.2f} | "
              f"2^{bits(tau / DELTA):8.2f} | {dominant(t)}")

# ---------------------------------------------------------------- PART D
print("\nPART D  sparse packing sweep (eps_D, delta', c_rs held at the reference audit's full-packing values)")
print(f"    {'n_s':>7} | {'c(n_s)':>8} | {'StC P':>13} | {'V4 head':>8} | {'V4 dec':>8} | "
      f"{'V5 head':>8} | {'V5 dec':>8}")
print("    " + "-" * 78)
first = {"V4": None, "V5": None}
for L in range(0, 17):
    ns = 2 ** L
    P = partition(ns)
    _, t4 = tau_terms(ns, "V4", P)
    _, t5 = tau_terms(ns, "V5", P)
    h4, h5 = bits(HALF) - bits(t4), bits(HALF) - bits(t5)
    for k, h in (("V4", h4), ("V5", h5)):
        if h >= 0:
            first[k] = ns
    print(f"    {'2^%d' % L:>7} | {c_of(ns):8.5f} | {str(P):>13} | {h4:+8.2f} | 2^{bits(t4 / DELTA):6.2f} | "
          f"{h5:+8.2f} | 2^{bits(t5 / DELTA):6.2f}")
print(f"    largest n_s with headroom >= 0:  V4 {first['V4']}   V5 {first['V5']}")

# ---------------------------------------------------------------- PART E
print("\nPART E  budget inversion: largest kappa*eps_D (Delta units) meeting a decoded tolerance T")
print(f"        the reference audit has kappa*eps_D = 2^{bits(KEPS):.3f}; 'gap' = bits it must shrink by")
print(f"    {'T':>6} | {'n_s':>6} | {'V4 max keps':>12} | {'V4 gap':>7} | {'V5 max keps':>12} | {'V5 gap':>7}")
print("    " + "-" * 66)
for logT in (-1, -10, -20):
    for L in (4, 8, 12, 16):
        ns = 2 ** L
        out = []
        for var in ("V4", "V5"):
            t, _ = tau_terms(ns, var)
            rest = t["approx"] + t["StC inj"] + t["B_sd"]
            coef = t["noise"] / KEPS
            room = (2.0 ** logT) * DELTA - rest
            if room <= 0:
                out.append(("infeasible", "  -  "))
            else:
                kmax = room / coef
                out.append((f"2^{bits(kmax):.2f}", f"{bits(KEPS) - bits(kmax):+.2f}"))
        print(f"    {'2^%d' % logT:>6} | {'2^%d' % L:>6} | {out[0][0]:>12} | {out[0][1]:>7} | "
              f"{out[1][0]:>12} | {out[1][1]:>7}")

# ---------------------------------------------------------------- summary block
print("\n" + "=" * 86)
print("SUMMARY BEGIN")
for (logN, bmax, al) in rows_B:
    print(f"B N=2^{logN}: slot-bounded max ratio {bmax:.6f}, coef-aligned ratio/n {al:.4f}")
for var in ("V1", "V2", "V3", "V4", "V5"):
    t, tau = tau_terms(n17, var)
    print(f"C {var}: tau 2^{bits(tau):.2f} head {bits(HALF) - bits(tau):+.2f} "
          f"dec 2^{bits(tau / DELTA):.2f} dom {dominant(t)}")
print(f"D first-open n_s: V4 {first['V4']}  V5 {first['V5']}")
print("GATE FAILURES:", FAILS if FAILS else "none")
print("SUMMARY END")
print("=" * 86)
