#!/usr/bin/env python3
"""
sweep_degb_delta_arcsine.py  --  where is the best certified decoded precision?

STEP-1e PATCH (1 Oct 2026): PART L. Under the K bound (kappa = 1): ablation without Lemma
parseval (sup-norm sine term) for every packing, and full term tables at the ref cell and the
optima of 1, 2^3, 2^8 slots and full packing. Earlier parts untouched.

STEP-1d PATCH (1 Oct 2026): PART K = PART J in lemma form: analytic derivative bounds (no grid
maxima), imaginary part of the input error from the conjugation key switch only, and message
bounds on the enlarged domain. Gates K0-K2. Earlier parts untouched.

STEP-1c PATCH (1 Oct 2026): PART J (MODEL, NOT A RESULT). Sensitivity annotation for the
EvalMod input error: instead of pushing B_cts through the gate-by-gate tag recursion, charge
  eD_J = eD_int + L_eff * B_cts + D * L2 * alpha^2 / 2,   alpha = B_cts / D,
where eD_int = EvalMod recursion with zero input error (internal rescale + key-switch seeds,
unchanged), L = max |p'(t)| and L2 = max |p''(t)| of the composite approximant p (base
interpolant, double-angle tier, arcsine tier if on) on a dense grid of [-1, 1], and
L_eff = L + alpha * L2 (proxy for a complex disk of radius alpha). The model takes
min(eD, eD_J) since both would be valid bounds. Grid maxima are NOT certified: a lemma would
need certified derivative bounds (E3-style) and a soundness argument for lowering an error tag.
Only the noise term changes; every other term and every earlier part is untouched.

STEP-1b PATCH (1 Oct 2026): PART I. Decomposes the EvalMod noise tag eD at every optimum into
the parts seeded by (a) EvalMod-internal rescale remainders c_rs(H), (b) EvalMod-internal key
switches bks/D, (c) the input tag B_cts (CtS + front gate + extraction), by re-running the
EvalMod recursion with the other two switched off. Gate I0: all three on reproduces eD exactly.
Nothing else changes; all earlier parts and gates are untouched.

STEP-1 PATCH (1 Oct 2026): POSTFIX + PART H. The last EvalMod rescale follows the per-stream
real-part extraction, so its injection c_rs(H) is a COMPLEX channel error: Lemma stc (i)
charges 2 n_s on it, not c(n_s) n_s. POSTFIX adds the difference (2 n_s - c n_s) 2^degb c_rs(H)
to V4c / V4+P / V4+Pg. PART 0 (legacy V4/V5) is unaffected. PART H compares off vs on.

FINAL-RUN PATCH (1 Oct 2026): KEFF_FIX, PART F (all packings), SUBSUM + PART G.
With SUBSUM = True every sparse row (PART C/E/F) carries the SubSum level; full-packing
rows and all gates are unchanged. See the SUBSUM comment block for the model.

PATCH 30 Sep 2026  --  what is new, read this first
-------------------------------------------------------------------
Everything from the 29 Sep version is unchanged, including PART 0 (HARD GATE),
which still gates the LEGACY variants V1/V4/V5 at kappa = 2 with no conjugation
term. Three new things are layered on top:

  V4c   V4 + the two conjugation key switches of the per-stream real-part
        extraction (paper Sec 5.5, Lemma stc discussion). Modelling choice:
        the extraction is applied to the last EvalMod output BEFORE its rescale,
        so each switch has bks(stc_top + 1)/D after that rescale. (v + conj v)/2
        halves it per stream; the two streams recombine to <= bks/D per channel,
        and the error is COMPLEX, so StC charges n_s (Lemma stc (i)), not
        c(n_s) n_s. Term: n_s * 2^degb * bks(stc_top + 1)/D.
  V4+P  V4c with the sine term replaced by Lemma parseval AT THE IDEAL INPUT:
          approx = A_sup * q0*Lambda(E)  +  q0 * parseval_factor(eps_in, arc)
          eps_in = (nu + B_in)/q0 = (D/q0) * Z',  Z' = 1 + B_in/D,
        B_in = upstream error tag entering the refresh (end of C^dagger incl.
        KS_ds). NOT the enlarged domain Xi. The interpolation term keeps A_sup.
        parseval_factor uses max_theta |f|/theta^2 * eps_in^2, which equals
        f(eps_in) with the arcsine tier off and is the valid generalisation with
        it on (the paper's Lemma parseval is stated for the sine only).
  KAPPA V4c and V4+P are evaluated at kappa in {2, 1}. kappa multiplies eps_D
        only. Legacy V1/V4/V5 and the gates stay at kappa = 2.

New gate (PART 0b): tau(V4+P) <= tau(V4c) at every cell of the PART C domain,
every packing, arcsine on/off, both kappa; and eps_in <= 1/4 (Parseval
precondition). If it fails the script stops.
New outputs: reference-cell term table (V4, V4c, V4+P), PART C with 5 terms per
best cell, PART D (the 15.85-bit check).

-------------------------------------------------------------------
ORIGINAL HEADER (29 Sep)
-------------------------------------------------------------------
QUESTION
Last run: at the reference audit's cell (Delta = 2^53, q0 = 2^60, degb = 7, arcsine off) the
corrected tau gives a decoded error of 2^9.95 (V4, sup tags) or 2^2.55 (V5, l2
model). The binding term is the back-gated noise 2^degb * kappa eps_D. Lowering
degb shrinks the back gate but grows delta' ~ (2 pi)^2 Delta^3 / (6 q0^2); the
arcsine tier cuts delta' at the price of 2 chain primes. Sweep (Delta, degb, arcsine)
and report the best decoded error each variant can certify.

MODEL
A port of the reference audit's gate-level noise model (gate-level structure, the reference audit's locked
parameters: N = 2^17, H = 64, K = 32, ell = 6, d_base = 18, (k,m) = (18,1),
CtS [4,3,3,3,3] rot [12,8,8,8,8], StC [3,4,4,5], t = 20, E_ksk = 2^6, E_0 = 1,
kappa = 2, kappa_rs = 3, D_C = 3). PART 0 is a HARD GATE: it must reproduce the reference audit's
recorded B_CtS, eps_D, Xi, delta', E, rho*, all ten cells of the Delta x D_C
headroom sweep in sec:tau-honest, and last run's V4 / V5. If any gate fails the
script stops: the port has drifted from the reference audit and nothing below means anything.

VARIANTS
  V4  sup tags, made consistent: StC factor c*n_s, 2^degb on eps_D only,
      StC injections amplified by later StC levels.
  V5  MODEL, NOT PROVED: l2 slot tags. Noise through StC with 2 sqrt(n_s);
      the sine-vs-identity term through Parseval with factor
      max_theta |f(theta)|/theta^2 * eps^2; evaluated at ideal inputs.
      The Chebyshev interpolation term q0*Lambda(E) takes the full sup factor.

ARCSINE TIER (MODEL): degree-3 Taylor y -> y + (2 pi)^2 y^3 / 6 after the
double-angle tier, depth 2. Noise rule: y^2 and y^3 gates plus 2 level-matching
rescales for the linear term.

SECURITY / SOUNDNESS COLUMNS ARE INDICATORS, NOT RESULTS
  sec~  linear interpolation through the reference audit's two estimator rows at N = 2^17, H = 64
        (168.0 at log Q = 1332, 165.4 at 1385). Outside that range it is an
        extrapolation. Any cell you want to use needs its own estimator run.
  pmin  smallest chain prime, bits.

Pure python + numpy. No OpenFHE, no DEV flag.
USAGE:  python3 sweep_degb_delta_arcsine.py      The summary block is printed between
        SUMMARY BEGIN / SUMMARY END.
"""
import sys
import math
import numpy as np

# ------------------------------------------------------------------- locked (the reference audit)
N = 2 ** 17
n_full = N // 2
H = 64
H_DENSE = N // 2
K = 32
ELL = 6
DB = 18
E0 = 1.0
EKSK = 2.0 ** 6
TG = 20
KRS = 3.0
KAPPA = 2.0            # legacy variants and all PART 0 gates
KAPPAS = (2.0, 1.0)    # new variants V4c, V4+P
DC = 3
CTS_KS = [4, 3, 3, 3, 3]
CTS_ROT = [12, 8, 8, 8, 8]
STC_PART = [3, 4, 4, 5]
R_STC = 4
TWOPI = 2 * math.pi

# measured (check_stc_amplification / check_units_sparse_l2): real-channel sup / n_s
C_OF = {1: 1.41421, 2 ** 8: 1.27324, 2 ** 16: 1.273239}
PART_OF = {1: [], 2 ** 8: [4, 4], 2 ** 16: STC_PART}

# NEW (final-run patch): intermediate packings for PART F. c(n) from the closed form of
# Lemma stc; the three measured entries above are kept so every earlier number is unchanged.
# StC level split for 2^k slots: one level if k <= 5, else ceil(k/5) near-equal levels
# (reproduces [4, 4] at 2^8). The chain length (R_STC = 4 levels) is kept for every packing,
# as in the existing sparse rows, so logQ is the full-packing chain.
KEFF_FIX = True   # K_eff = K + (nu + B_in)/q0, as Lemma cts (ii) requires (was K + nu/q0)
# SubSum for sparse packing (n_s < n), modelled as its OWN level at the top of the chain:
#   the raised ciphertext (phase w + q0 I, no scale) is multiplied by a chain prime p ~ 2^lD
#   exactly (integer scalar), then log2(k) steps ct <- ct + tau(ct) with k = N/(2 n_s), each
#   an automorphism plus a key switch at top+1 towers, then one rescale by p.
#   Without the scalar/rescale the raw key-switch noise (~2^67) would exceed the phase
#   (~2^65): proof-friendly key switching has no auxiliary modulus to divide it (F3).
#   Error after the level: E_ss = (k-1) * bks(top+1)/D + c_rs(H)   [doubling per step, then /p]
#   The message becomes k * (stride projection); the sparse CtS normalisation carries 1/k,
#   while the port keeps the full-packing CtS (product n = k n_s), so E_ss enters as E_ss/k.
#   Cost: one extra chain prime, logQ += lD, for every sparse row.
SUBSUM = True
# POSTFIX (Step 1): charge the last EvalMod rescale injection (after the real-part extraction)
# with Lemma stc (i): 2 n_s instead of c(n_s) n_s. The extra term is kept separately as
# post_term, so PART H can report the landscape with and without it from one run.
POSTFIX = True


def c_of(ns):
    return C_OF[ns] if ns in C_OF else 1.0 / (ns * math.sin(math.pi / (4 * ns)))


def part_of(ns):
    if ns in PART_OF:
        return PART_OF[ns]
    k = int(round(math.log2(ns)))
    r = max(1, math.ceil(k / 5))
    base, extra = divmod(k, r)
    return [base] * (r - extra) + [base + 1] * extra

FAILS = []


def bits(x):
    return math.log2(x) if x > 0 else float("-inf")


def gate(name, ok, detail=""):
    print(f"    [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILS.append(name)


def crs(h):
    return KRS * N * (1 + h) / 2.0


C_DENSE, C_SPARSE = crs(H_DENSE), crs(H)


def bks(towers, lq0, lD):
    m = math.ceil(lq0 / TG) + (towers - 1) * math.ceil(lD / TG)
    return m * (2 * N * 2.0 ** TG) * (N * EKSK)


def depth(j):
    return 0 if j == 1 else math.ceil(math.log2(j))


_CHEB = {}


def cheb_coeffs(Keff, r=ELL, n=DB, M=2048):
    key = (round(Keff, 12), r, n, M)
    if key not in _CHEB:
        _CHEB[key] = _cheb_coeffs(Keff, r, n, M)
    return _CHEB[key]


def _cheb_coeffs(Keff, r=ELL, n=DB, M=2048):
    A = TWOPI ** (-(2.0 ** (-r)))
    th = np.pi * (np.arange(M) + 0.5) / M
    x = np.cos(th)
    f = A * np.cos((TWOPI * Keff * x - np.pi / 2) / 2 ** r)
    c = np.array([2 * np.sum(f * np.cos(j * th)) / M for j in range(n + 1)])
    c[0] /= 2
    return c


def lam(Keff, n=DB, ell=ELL):
    """Lemma dapprox: Chebyshev tail bound E (optimised rho) pushed through the DA recursion."""
    A = TWOPI ** (-(2.0 ** (-ell)))
    b = TWOPI * Keff / 2 ** ell
    rho = np.linspace(1.5, 40.0, 40000)
    r = 1.0 / rho
    E = 2 * A * np.cosh(b / 2 * (rho - r)) * r ** (n + 1) / (1 - r)
    i = int(np.argmin(E))
    d = float(E[i])
    for k in range(1 - ell, 1):
        Ak = TWOPI ** (-(2.0 ** (k - 1)))
        d = 4 * Ak * d + 6 * d * d
    return float(E[i]), float(rho[i]), d


def f_sine(th):
    """theta - sin(2 pi theta)/(2 pi) by its alternating series (no cancellation)."""
    th = np.asarray(th, dtype=float)
    out = np.zeros_like(th)
    term = th.copy()
    for k in range(1, 25):
        term = term * (TWOPI * th) ** 2 / ((2 * k) * (2 * k + 1))
        out += term if k % 2 == 1 else -term
    return out


def f_approx(th, arc):
    d = f_sine(th)
    if arc == 0:
        return d
    y = th - d
    return d - (TWOPI ** 2 / 6.0) * y ** 3          # theta - (y + a y^3)


def sup_f(tmax, arc):
    th = np.linspace(0.0, tmax, 20001)
    return float(np.max(np.abs(f_approx(th, arc))))


def parseval_factor(tmax, arc):
    th = np.linspace(tmax / 20000, tmax, 20000)
    return float(np.max(np.abs(f_approx(th, arc)) / th ** 2)) * tmax ** 2


def towers_layout(arc, dc=DC):
    dem = 12 + (2 if arc else 0)
    top = 1 + dc + R_STC + dem + len(CTS_KS)
    return top, dem


_MO = {}


def model(lD, lq0, dc=DC, arc=0, ns=n_full, ss=None):
    ss = SUBSUM if ss is None else ss
    ss_ns = ns if (ss and ns < n_full) else None
    key = (lD, lq0, dc, arc, ss_ns)
    if key not in _MO:
        _MO[key] = _model(lD, lq0, dc, arc, ss_ns)
    return _MO[key]


def _model(lD, lq0, dc, arc, ss_ns=None):
    D = 2.0 ** lD
    q0 = 2.0 ** lq0
    eps = D / q0
    Keff = K + eps
    top, dem = towers_layout(arc, dc)
    # upstream C^dagger, dense key
    B = N * E0 + N / 2.0
    for i in range(dc):
        B = (2 + B / D) * B + C_DENSE + bks(dc + 1 - i, lq0, lD) / D
    B_term = bks(2, lq0, lD) / D
    B += B_term
    B_in = B                                  # NEW: upstream tag entering the refresh
    if KEFF_FIX:
        Keff = K + (D + B_in) / q0
    E_ss, extra_lvl = 0.0, 0
    if ss_ns is not None:                     # SubSum level (sparse packing), see SUBSUM
        kss = N // (2 * ss_ns)
        E_ss = (kss - 1) * bks(top + 1, lq0, lD) / D + C_SPARSE
        B += E_ss / kss
        extra_lvl = 1
    # CtS, sparse key, S end-folded
    for i, k in enumerate(CTS_KS):
        B = 2.0 ** k * B + C_SPARSE + CTS_ROT[i] * bks(top - i, lq0, lD) / D
    S = D / (q0 * Keff * 2.0 ** 16 * 2)
    B = S * B + C_SPARSE
    em_top = top - len(CTS_KS)
    B_conj = bks(em_top, lq0, lD) / D
    B_cts = 2 * B + B_conj
    # EvalMod base tier, (k,m) = (18,1): Chebyshev basis then additive plaintext combination
    st = {1: B_cts}
    for j in range(2, DB + 1):
        tw = em_top - (depth(j) - 1)
        if j % 2 == 0:
            a = j // 2
            st[j] = 4 * st[a] + 2 * st[a] ** 2 / D + C_SPARSE + bks(tw, lq0, lD) / D
        else:
            a, b = j // 2, j // 2 + 1
            st[j] = (2 * (st[a] + st[b] + st[a] * st[b] / D) + C_SPARSE
                     + bks(tw, lq0, lD) / D + st[1])
    c = cheb_coeffs(Keff)
    B = C_SPARSE + sum(abs(c[j]) * (st[j] + (5 - depth(j)) * C_SPARSE) for j in range(1, DB + 1))
    # double-angle tier
    for idx, i in enumerate(range(1 - ELL, 1)):
        A = TWOPI ** (-(2.0 ** (i - 1)))
        tw = em_top - 6 - idx
        B = 4 * A * B + 2 * B * B / D + C_SPARSE + bks(tw, lq0, lD) / D
    # arcsine tier (model)
    if arc:
        tw = em_top - 12
        nu_y = D * eps * 1.01
        B2 = (2 * nu_y * B + B * B) / D + C_SPARSE + bks(tw, lq0, lD) / D
        nu_y2 = D * (eps * 1.01) ** 2
        a = TWOPI ** 2 / 6.0
        B3 = a * (nu_y2 * B + nu_y * B2 + B * B2) / D + C_SPARSE + bks(tw - 1, lq0, lD) / D
        B = B + 2 * C_SPARSE + B3
    eD = B
    B_sig = B_cts + B_conj + B_term
    Xi = B_sig / D * q0 * Keff
    stc_top = 1 + dc + R_STC
    B_sd = bks(stc_top, lq0, lD) / D
    # NEW: one conjugation key switch per stream, on the last EvalMod output before
    # its rescale (stc_top + 1 towers), divided by that rescale.
    B_cj = bks(stc_top + 1, lq0, lD) / D
    logQ = lq0 + (top - 1 + extra_lvl) * lD
    return dict(B_cts=B_cts, eD=eD, Xi=Xi, B_sd=B_sd, logQ=logQ, top=top, Keff=Keff,
                B_in=B_in, B_cj=B_cj, E_ss=E_ss)


_LAM = {}


def q0lam(lq0, Keff):
    key = round(Keff, 9)
    if key not in _LAM:
        _LAM[key] = lam(Keff)[2]
    return 2.0 ** lq0 * _LAM[key]


def amp_inj(P, half=False):
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


_EV = {}


def evaluate(lD, degb, arc=0, ns=n_full, dc=DC, ss=None):
    ss = SUBSUM if ss is None else ss
    key = (lD, degb, arc, ns, dc, ss)
    if key not in _EV:
        _EV[key] = _evaluate(lD, degb, arc, ns, dc, ss)
    return _EV[key]


def _evaluate(lD, degb, arc, ns, dc, ss):
    lq0 = lD + degb
    D, q0 = 2.0 ** lD, 2.0 ** lq0
    eps = D / q0
    mo = model(lD, lq0, dc, arc, ns, ss)
    keps = KAPPA * mo["eD"]
    QL = q0lam(lq0, mo["Keff"])
    P = part_of(ns)
    A_sup = c_of(ns) * ns
    back = 2.0 ** degb
    # the reference audit as written (only meaningful at full packing, arc off)
    dsup = QL + q0 * sup_f((D + mo["Xi"]) / q0, arc)
    tau_v1 = 2 * back * (keps + dsup) + mo["B_sd"] + R_STC * C_DENSE
    # V4 (legacy, gated)
    t4 = {"noise": A_sup * back * keps, "approx": A_sup * dsup,
          "StC inj": C_DENSE * amp_inj(P), "B_sd": mo["B_sd"] * amp_after_first(P)}
    # V5 (legacy model, gated)
    t5 = {"noise": 2 * math.sqrt(ns) * back * keps,
          "approx": q0 * parseval_factor(eps, arc) + A_sup * QL,
          "StC inj": C_DENSE * amp_inj(P, True), "B_sd": mo["B_sd"] * amp_after_first(P, True)}
    # NEW: V4c and V4+P, per kappa
    eps_in = (D + mo["B_in"]) / q0
    sine_p = q0 * parseval_factor(eps_in, arc)
    conj = ns * back * mo["B_cj"]
    post_term = (2 * ns - A_sup) * back * C_SPARSE      # POSTFIX, see header
    new = {}
    for kap in KAPPAS:
        noise_k = A_sup * back * kap * mo["eD"]
        t4c = {"noise": noise_k, "approx": A_sup * dsup, "StC inj": t4["StC inj"],
               "B_sd": t4["B_sd"], "conj": conj}
        if POSTFIX:
            t4c["post"] = post_term
        # Lemma parseval is stated for eps*Z' <= 1/4; outside it V4+P falls back to V4c
        t4p = dict(t4c, approx=A_sup * QL + sine_p) if eps_in <= 0.25 else dict(t4c)
        # V4+Pg: Lemma parseval in the max|f|/theta^2 form, valid for eps_in < 1/2
        # (|theta_k| <= ||theta||_2 <= eps_in and no-wrap); also a = eps_in < 1/2 keeps K = H/2
        t4pg = dict(t4c, approx=A_sup * QL + sine_p) if eps_in < 0.5 else dict(t4c)
        new[kap] = dict(t4c=t4c, tau4c=sum(t4c.values()), t4p=t4p, tau4p=sum(t4p.values()),
                        t4pg=t4pg, tau4pg=sum(t4pg.values()))
    sec = 168.0 - (mo["logQ"] - 1332) * (2.6 / 53.0)
    return dict(lD=lD, degb=degb, lq0=lq0, arc=arc, ns=ns, mo=mo, keps=keps, dsup=dsup,
                tau_v1=tau_v1, t4=t4, tau4=sum(t4.values()), t5=t5, tau5=sum(t5.values()),
                logQ=mo["logQ"], sec=sec, pmin=min(lD, lq0), new=new, eps_in=eps_in, post_term=post_term,
                interp=A_sup * QL, sine_p=sine_p, sine_sup=A_sup * q0 * sup_f((D + mo["Xi"]) / q0, arc))


def dec(tau, lD):
    return bits(tau) - lD


def dom(t):
    k = max(t, key=t.get)
    return f"{k} 2^{bits(t[k]):.1f}"


# variant accessors: name -> (terms, tau)
def V(e, var, kap=KAPPA):
    if var == "V4":
        return e["t4"], e["tau4"]
    if var == "V5":
        return e["t5"], e["tau5"]
    if var == "V4c":
        return e["new"][kap]["t4c"], e["new"][kap]["tau4c"]
    if var == "V4+P":
        return e["new"][kap]["t4p"], e["new"][kap]["tau4p"]
    if var == "V4+Pg":
        return e["new"][kap]["t4pg"], e["new"][kap]["tau4pg"]
    raise KeyError(var)


def terms_str(t):
    keys = ["noise", "approx", "StC inj", "B_sd", "conj"] + (["post"] if POSTFIX else [])
    return " / ".join(f"2^{bits(t[k]):.1f}" if k in t else "  -  " for k in keys)


def cells():
    for lD in range(36, 63):
        for degb in range(3, 15):
            if lD + degb <= 64:
                yield lD, degb


# =============================================================================
print("=" * 100)
print("sweep_degb_delta_arcsine.py   N = 2^17, H = 64, D_C = %d   (patched 30 Sep: V4c, V4+P, kappa)" % DC)
print("=" * 100)

# ---------------------------------------------------------------- PART 0 (HARD GATE, unchanged)
print("\nPART 0  HARD GATE: the port must reproduce the reference audit's recorded numbers")
m53 = model(53, 60)
gate("B_CtS(2^53, D_C=3) = 2^25.504", abs(bits(m53["B_cts"]) - 25.504) < 0.01, f"2^{bits(m53['B_cts']):.3f}")
gate("eps_D = 2^38.548", abs(bits(m53["eD"]) - 38.548) < 0.02, f"2^{bits(m53['eD']):.3f}")
gate("Xi = 2^37.504", abs(bits(m53["Xi"]) - 37.504) < 0.01, f"2^{bits(m53['Xi']):.3f}")
E, rho, _ = lam(m53["Keff"])
gate("Lemma dapprox E = 4.51e-13, rho* = 12.07", abs(E / 4.51e-13 - 1) < 0.02 and abs(rho - 12.07) < 0.05,
     f"E = {E:.3e}, rho* = {rho:.2f}")
e53 = evaluate(53, 7)
gate("delta' = 2^41.718", abs(bits(e53["dsup"]) - 41.718) < 0.01, f"2^{bits(e53['dsup']):.3f}")
want = {3: [0.79, 1.99, 1.99, 1.23, 0.27], 0: [0.97, 2.20, 2.09, 1.26, 0.28]}
for dc_, row in want.items():
    got = [bits(2.0 ** l / 2) - bits(evaluate(l, 60 - l, 0, n_full, dc_)["tau_v1"]) for l in (51, 52, 53, 54, 55)]
    ok = all(abs(g - w) < 0.02 for g, w in zip(got, row))
    gate(f"sec:tau-honest sweep, D_C={dc_}", ok,
         "got " + ",".join(f"{g:+.2f}" for g in got) + "  want " + ",".join(f"{w:+.2f}" for w in row))
h4 = bits(2.0 ** 52) - bits(e53["tau4"])
h5 = bits(2.0 ** 52) - bits(e53["tau5"])
gate("last run V4 = -10.95", abs(h4 + 10.95) < 0.03, f"{h4:+.3f}")
gate("last run V5 = -3.55 (after the q0*Lambda correction)", abs(h5 + 3.55) < 0.03, f"{h5:+.3f}")
if FAILS:
    print("\n  HARD GATE FAILED:", FAILS)
    print("  The port does not reproduce the reference audit. Do not read anything below.")
    sys.exit(1)
print("  GATE PASSED.")

# ---------------------------------------------------------------- PART 0b (NEW GATE)
print("\nPART 0b  NEW GATE: tau(V4+P) <= tau(V4c) on every cell (V4+P = V4c where eps_in > 1/4)")
n_chk, viol, worst_eps, no_p = 0, [], 0.0, set()
for arc in (0, 3):
    for ns in (n_full, 2 ** 8, 1):
        for lD, degb in cells():
            e = evaluate(lD, degb, arc, ns)
            worst_eps = max(worst_eps, e["eps_in"])
            if e["eps_in"] > 0.25:
                no_p.add((lD, degb, arc))
            for kap in KAPPAS:
                n_chk += 1
                if e["new"][kap]["tau4p"] > e["new"][kap]["tau4c"] * (1 + 1e-12):
                    viol.append((lD, degb, arc, ns, kap))
gate(f"tau4p <= tau4c on {n_chk} (cell, packing, arc, kappa) combinations", not viol,
     f"violations: {viol[:5]}" if viol else "")
print(f"    [INFO] Lemma parseval not applicable (eps_in > 1/4) at {len(no_p)} (logD, degb, arc) cells; "
      f"V4+P = V4c there. max eps_in = 2^{bits(worst_eps):.2f}")
if no_p:
    print(f"           e.g. {sorted(no_p)[:6]}")
if FAILS:
    print("\n  NEW GATE FAILED:", FAILS, " -- see the output above.")
    sys.exit(1)
print("  GATE PASSED.")

# ---------------------------------------------------------------- REFERENCE CELL
print("\nREFERENCE CELL  Delta = 2^53, q0 = 2^60 (degb = 7), full packing, arcsine off")
print(f"    B_in = 2^{bits(e53['mo']['B_in']):.3f}  Z' - 1 = 2^{bits(e53['mo']['B_in'] / 2.0 ** 53):.2f}   "
      f"eps_in = 2^{bits(e53['eps_in']):.3f}   conj bks/D = 2^{bits(e53['mo']['B_cj']):.3f}")
print(f"    approx split: interp A*q0*Lambda = 2^{bits(e53['interp']):.2f}   "
      f"sine sup-norm A*dsin(D+Xi) = 2^{bits(e53['sine_sup']):.2f}   sine Parseval = 2^{bits(e53['sine_p']):.2f}")
print(f"    {'variant':>12} | {'dec err':>8} | noise / approx / StC inj / B_sd / conj")
REF = []
for var, kap in (("V4", 2.0), ("V4c", 2.0), ("V4+P", 2.0), ("V4c", 1.0), ("V4+P", 1.0)):
    t, tau = V(e53, var, kap)
    lab = f"{var} k={kap:g}"
    REF.append((lab, dec(tau, 53), t))
    print(f"    {lab:>12} | {dec(tau, 53):+8.2f} | {terms_str(t)}")

# ---------------------------------------------------------------- PART A
print("\nPART A  the trade-off at fixed Delta = 2^53, full packing, arcsine off")
print(f"    {'degb':>4} | {'eps_D':>8} | {'back*keps':>9} | {'delta_sup':>9} | {'sineP':>8} | "
      f"{'V4 dec':>7} | {'V4+P k2':>7} | {'V4+P k1':>7} | V4+P k2 dominant")
print("    " + "-" * 96)
for degb in range(3, 13):
    e = evaluate(53, degb)
    print(f"    {degb:>4} | 2^{bits(e['mo']['eD']):6.2f} | 2^{bits(2 ** degb * e['keps']):6.2f} | "
          f"2^{bits(e['dsup']):6.2f} | 2^{bits(e['sine_p']):5.1f} | "
          f"{dec(e['tau4'], 53):+7.2f} | {dec(V(e, 'V4+P', 2.0)[1], 53):+7.2f} | "
          f"{dec(V(e, 'V4+P', 1.0)[1], 53):+7.2f} | {dom(V(e, 'V4+P', 2.0)[0])}")

# ---------------------------------------------------------------- PART B
LD_RANGE = list(range(40, 62, 2))
DEGB_RANGE = list(range(3, 13))
print("\nPART B  decoded error bits log2(tau/Delta), full packing  (lower is better; '  x' = q0 > 2^64)")
for var in ("V4", "V4+P"):
    for arc in (0, 3):
        print(f"\n  {var}{' (kappa=2, conj on)' if var == 'V4+P' else ' (legacy)'}, "
              f"arcsine {'ON (deg 3, +2 primes)' if arc else 'off'}")
        print("    logD\\degb " + "".join(f"{d:>7}" for d in DEGB_RANGE))
        for lD in LD_RANGE:
            row = []
            for degb in DEGB_RANGE:
                if lD + degb > 64:
                    row.append(f"{'x':>7}")
                    continue
                e = evaluate(lD, degb, arc)
                row.append(f"{dec(V(e, var, 2.0)[1], lD):7.2f}")
            print(f"    {lD:>9} " + "".join(row))

# ---------------------------------------------------------------- PART C
print("\nPART C  best cell per (variant, kappa, packing, arcsine) over logD in 36..62, degb in 3..14, q0 <= 2^64")
print(f"    {'variant':>10} {'n_s':>5} {'arc':>4} | {'logD':>4} {'degb':>4} | {'dec err':>8} | {'logQ':>5} "
      f"{'sec~':>6} {'pmin':>4} | noise / approx / StC inj / B_sd / conj")
print("    " + "-" * 118)
VARS = [("V4", 2.0), ("V4c", 2.0), ("V4c", 1.0), ("V4+P", 2.0), ("V4+P", 1.0), ("V5", 2.0)]
best_rows = []
BEST = {}
for var, kap in VARS:
    for ns in (n_full, 2 ** 8, 1):
        for arc in (0, 3):
            best = None
            for lD, degb in cells():
                e = evaluate(lD, degb, arc, ns)
                d_ = dec(V(e, var, kap)[1], lD)
                if best is None or d_ < best[0]:
                    best = (d_, e)
            d_, e = best
            t = V(e, var, kap)[0]
            nsl = "full" if ns == n_full else f"2^{int(math.log2(ns))}"
            lab = f"{var} k={kap:g}" if var in ("V4c", "V4+P") else var
            BEST[(var, kap, ns, arc)] = (d_, e)
            print(f"    {lab:>10} {nsl:>5} {('on' if arc else 'off'):>4} | {e['lD']:>4} {e['degb']:>4} | "
                  f"{d_:+8.2f} | {e['logQ']:>5} {e['sec']:6.1f} {e['pmin']:>4} | {terms_str(t)}"
                  + ("" if var != "V4+P" else (" [P]" if e["eps_in"] <= 0.25 else " [P n/a]")))
            pf = "" if var != "V4+P" else (" [P]" if e["eps_in"] <= 0.25 else " [P n/a]")
            best_rows.append((lab + pf, nsl, arc, e["lD"], e["degb"], d_, e["logQ"], e["sec"], e["pmin"], t))
    print()

# ---------------------------------------------------------------- PART D
L_FULL1 = bits(C_OF[n_full] * n_full / C_OF[1])
L_FULL8 = bits(C_OF[n_full] * n_full / (C_OF[2 ** 8] * 2 ** 8))
print("PART D  packing gaps of the best cells vs the StC factor")
print(f"    reference: log2(c(n)n/c(1)) = {L_FULL1:.2f}   log2(c(n)n/(c(2^8)2^8)) = {L_FULL8:.2f}")
print(f"    {'variant':>10} {'arc':>4} | {'full-1':>7} {'vs':>6} | {'full-2^8':>8} {'vs':>6} | "
      f"approx/noise at full optimum")
D_ROWS = []
for var, kap in VARS[:-1]:
    for arc in (0, 3):
        df, ef = BEST[(var, kap, n_full, arc)]
        d8, _ = BEST[(var, kap, 2 ** 8, arc)]
        d1, _ = BEST[(var, kap, 1, arc)]
        tf = V(ef, var, kap)[0]
        lab = f"{var} k={kap:g}" if var in ("V4c", "V4+P") else var
        g1, g8 = df - d1, df - d8
        ratio = bits(tf["approx"]) - bits(tf["noise"])
        D_ROWS.append((lab, arc, g1, g1 - L_FULL1, g8, g8 - L_FULL8, ratio))
        print(f"    {lab:>10} {('on' if arc else 'off'):>4} | {g1:7.2f} {g1 - L_FULL1:+6.2f} | "
              f"{g8:8.2f} {g8 - L_FULL8:+6.2f} | 2^{ratio:+.1f}")

# ---------------------------------------------------------------- PART E (NEW, 30 Sep follow-up)
print("\nPART E  V4+P full-packing optimum sat on degb = 3, the lower edge of the PART C domain.")
print("        Extend to degb = 2 with Lemma parseval in its max|f|/theta^2 form (V4+Pg, needs eps_in < 1/2).")
print("        degb = 1 stays excluded: eps = 1/2 breaks the small-phase condition a < 1/2 of Lemma wrap.")


INADM = set()


def cells_ext():
    """degb 2..14; drops cells with eps_in >= 1/2 (a < 1/2 of Lemma wrap fails: inadmissible design)."""
    for lD in range(36, 63):
        for degb in range(2, 15):
            if lD + degb <= 64:
                if evaluate(lD, degb, 0, n_full)["eps_in"] >= 0.5:
                    INADM.add((lD, degb))
                    continue
                yield lD, degb


n_e, viol_e = 0, []
for arc in (0, 3):
    for ns in (n_full, 2 ** 8, 1):
        for lD, degb in cells_ext():
            e = evaluate(lD, degb, arc, ns)
            for kap in KAPPAS:
                n_e += 1
                if e["new"][kap]["tau4pg"] > e["new"][kap]["tau4c"] * (1 + 1e-12):
                    viol_e.append((lD, degb, arc, ns, kap))
gate(f"tau4pg <= tau4c on {n_e} extended combinations", not viol_e,
     f"violations: {viol_e[:5]}" if viol_e else "")
print(f"    [INFO] inadmissible (eps_in >= 1/2) and dropped: {sorted(INADM)}")
if FAILS:
    print("\n  PART E GATE FAILED:", FAILS, " -- see the output above.")
    sys.exit(1)
print("\n    profile at q0 = 2^64, full packing, arcsine off (dec err):")
print(f"    {'degb':>4} {'logD':>4} | {'V4c k2':>7} {'V4+Pg k2':>9} {'V4+Pg k1':>9} | V4+Pg k2 noise / approx")
E_PROF = []
for degb in range(2, 9):
    lD = 64 - degb
    e = evaluate(lD, degb, 0, n_full)
    t = V(e, "V4+Pg", 2.0)[0]
    row = (degb, lD, dec(V(e, "V4c", 2.0)[1], lD), dec(V(e, "V4+Pg", 2.0)[1], lD), dec(V(e, "V4+Pg", 1.0)[1], lD))
    E_PROF.append(row)
    print(f"    {degb:>4} {lD:>4} | {row[2]:+7.2f} {row[3]:+9.2f} {row[4]:+9.2f} | "
          f"2^{bits(t['noise']):.1f} / 2^{bits(t['approx']):.1f}")
print("\n    best V4+Pg over logD 36..62, degb 2..14:")
E_ROWS = []
for kap in KAPPAS:
    for ns in (n_full, 2 ** 8, 1):
        for arc in (0, 3):
            best = None
            for lD, degb in cells_ext():
                e = evaluate(lD, degb, arc, ns)
                d_ = dec(V(e, "V4+Pg", kap)[1], lD)
                if best is None or d_ < best[0]:
                    best = (d_, e)
            d_, e = best
            t = V(e, "V4+Pg", kap)[0]
            nsl = "full" if ns == n_full else f"2^{int(math.log2(ns))}"
            flag = ("EDGE degb=2" if e["degb"] == 2 else "") + (" needs-gen" if e["eps_in"] > 0.25 else "")
            E_ROWS.append((f"V4+Pg k={kap:g}", nsl, arc, e["lD"], e["degb"], d_, e["logQ"], e["sec"], e["pmin"], t, flag))
            print(f"    V4+Pg k={kap:g} {nsl:>5} {('on' if arc else 'off'):>4} | {e['lD']:>4} {e['degb']:>4} | "
                  f"{d_:+8.2f} | {e['logQ']:>5} {e['sec']:6.1f} {e['pmin']:>4} | {terms_str(t)} {flag}")

# ---------------------------------------------------------------- PART F (NEW, final run)
print("\nPART F  V4+Pg best cell for every packing 2^0..2^16 (degb 2..14, q0 <= 2^64); thresholds")
print(f"    {'n_s':>5} | {'k2 off':>7} {'k2 on':>7} {'k1 off':>7} {'k1 on':>7} | best k2 (logD/degb/arc)  best k1 (logD/degb/arc)")
F_ROWS = []
for k in range(0, 17):
    ns = 2 ** k
    res = {}
    for kap in KAPPAS:
        for arc in (0, 3):
            best = None
            for lD, degb in cells_ext():
                e = evaluate(lD, degb, arc, ns)
                d_ = dec(V(e, "V4+Pg", kap)[1], lD)
                if best is None or d_ < best[0]:
                    best = (d_, e)
            res[(kap, arc)] = best
    b2 = min((res[(2.0, a)] for a in (0, 3)), key=lambda x: x[0])
    b1 = min((res[(1.0, a)] for a in (0, 3)), key=lambda x: x[0])
    F_ROWS.append((k, {key: v[0] for key, v in res.items()}, b2, b1))
    cell = lambda b: f"{b[1]['lD']}/{b[1]['degb']}/{'on' if b[1]['arc'] else 'off'}"
    print(f"    2^{k:<3} | {res[(2.0, 0)][0]:+7.2f} {res[(2.0, 3)][0]:+7.2f} {res[(1.0, 0)][0]:+7.2f} "
          f"{res[(1.0, 3)][0]:+7.2f} | {cell(b2):>22}  {cell(b1):>22}")
for kap, idx in ((2.0, 2), (1.0, 3)):
    for thr in (0.0, -10.0):
        ok = [r[0] for r in F_ROWS if r[idx][0] < thr]
        print(f"    kappa={kap:g}: largest n_s with T < 2^{thr:g}: " + (f"2^{max(ok)}" if ok else "none"))

# ---------------------------------------------------------------- PART G (NEW: SubSum)
print("\nPART G  SubSum off vs on, V4+Pg, kappa = 1 (the chosen kappa); best over degb 2..14, arcsine on/off")
print(f"    {'n_s':>5} | {'off: dec':>8} {'cell':>9} {'logQ':>5} | {'on: dec':>8} {'cell':>9} {'logQ':>5} | "
      f"E_ss/k  B_cts(off->on) at the 'on' cell")
G_ROWS = []
for k in (0, 2, 3, 4, 8):
    ns = 2 ** k
    out = {}
    for ssv in (False, True):
        best = None
        for arc in (0, 3):
            for lD, degb in cells_ext():
                e = evaluate(lD, degb, arc, ns, DC, ssv)
                d_ = dec(V(e, "V4+Pg", 1.0)[1], lD)
                if best is None or d_ < best[0]:
                    best = (d_, e)
        out[ssv] = best
    d0, e0 = out[False]
    d1, e1 = out[True]
    kss = N // (2 * ns)
    m_off = model(e1["lD"], e1["lq0"], DC, e1["arc"], ns, False)
    cell = lambda e: f"{e['lD']}/{e['degb']}/{'on' if e['arc'] else 'off'}"
    G_ROWS.append((k, d0, cell(e0), e0["logQ"], d1, cell(e1), e1["logQ"],
                   bits(e1["mo"]["E_ss"] / kss), bits(m_off["B_cts"]), bits(e1["mo"]["B_cts"])))
    r = G_ROWS[-1]
    print(f"    2^{k:<3} | {d0:+8.2f} {r[2]:>9} {r[3]:>5} | {d1:+8.2f} {r[5]:>9} {r[6]:>5} | "
          f"2^{r[7]:.1f}  2^{r[8]:.3f} -> 2^{r[9]:.3f}")

# ---------------------------------------------------------------- PART H (NEW: POSTFIX)
print("\nPART H  POSTFIX off vs on, V4+Pg, kappa = 1, best over degb 2..14, arcsine on/off, every packing")
print(f"    {'n_s':>5} | {'off: dec':>8} {'cell':>9} | {'on: dec':>8} {'cell':>9} | {'delta':>7} | post term vs noise at 'on' cell")
H_ROWS = []
for k in range(0, 17):
    ns = 2 ** k
    best = {False: None, True: None}
    for arc in (0, 3):
        for lD, degb in cells_ext():
            e = evaluate(lD, degb, arc, ns)
            t, tau = V(e, "V4+Pg", 1.0)
            tau_on = tau if POSTFIX else tau + e["post_term"]
            tau_off = tau - e["post_term"] if POSTFIX else tau
            for flag, tv in ((False, tau_off), (True, tau_on)):
                d_ = dec(tv, lD)
                if best[flag] is None or d_ < best[flag][0]:
                    best[flag] = (d_, e)
    (d0, e0), (d1, e1) = best[False], best[True]
    t1 = V(e1, "V4+Pg", 1.0)[0]
    cell = lambda e: f"{e['lD']}/{e['degb']}/{'on' if e['arc'] else 'off'}"
    gap_bits = bits(t1["noise"]) - bits(e1["post_term"])
    H_ROWS.append((k, d0, cell(e0), d1, cell(e1), d1 - d0, bits(e1["post_term"]), gap_bits))
    print(f"    2^{k:<3} | {d0:+8.3f} {cell(e0):>9} | {d1:+8.3f} {cell(e1):>9} | {d1 - d0:+7.4f} | "
          f"post 2^{bits(e1['post_term']):.1f}, {gap_bits:.1f} bits below noise")

# ---------------------------------------------------------------- PART I (NEW: what seeds eD)
def evalmod_split(lD, lq0, arc, ns, rs=1.0, ks=1.0, inp=1.0, dc=DC):
    """The EvalMod recursion of _model, verbatim, with the three seeds scaled:
    rs  -> every c_rs(H) injected inside EvalMod, ks -> every bks/D inside EvalMod,
    inp -> the input tag B_cts. (1,1,1) must reproduce model()['eD'] (gate I0)."""
    mo = model(lD, lq0, dc, arc, ns)
    D, q0 = 2.0 ** lD, 2.0 ** lq0
    eps = D / q0
    top, _ = towers_layout(arc, dc)
    em_top = top - len(CTS_KS)
    CS = rs * C_SPARSE
    KS_ = lambda tw: ks * bks(tw, lq0, lD) / D
    st = {1: inp * mo["B_cts"]}
    for j in range(2, DB + 1):
        tw = em_top - (depth(j) - 1)
        if j % 2 == 0:
            a_ = j // 2
            st[j] = 4 * st[a_] + 2 * st[a_] ** 2 / D + CS + KS_(tw)
        else:
            a_, b_ = j // 2, j // 2 + 1
            st[j] = 2 * (st[a_] + st[b_] + st[a_] * st[b_] / D) + CS + KS_(tw) + st[1]
    c = cheb_coeffs(mo["Keff"])
    B = CS + sum(abs(c[j]) * (st[j] + (5 - depth(j)) * CS) for j in range(1, DB + 1))
    for idx, i in enumerate(range(1 - ELL, 1)):
        A = TWOPI ** (-(2.0 ** (i - 1)))
        tw = em_top - 6 - idx
        B = 4 * A * B + 2 * B * B / D + CS + KS_(tw)
    if arc:
        tw = em_top - 12
        nu_y = D * eps * 1.01
        B2 = (2 * nu_y * B + B * B) / D + CS + KS_(tw)
        nu_y2 = D * (eps * 1.01) ** 2
        a_ = TWOPI ** 2 / 6.0
        B3 = a_ * (nu_y2 * B + nu_y * B2 + B * B2) / D + CS + KS_(tw - 1)
        B = B + 2 * CS + B3
    return B, em_top


print("\nPART I  what seeds the EvalMod noise tag eD (V4+Pg, kappa = 1, optimum cell of every packing + ref cell)")
print("        per-injection margin = log2 c_rs(H) - log2 max_EvalMod bks/D;  shares = log2(part / eD)")
print(f"    {'n_s':>5} {'cell':>9} | {'eD':>8} | {'inj margin':>10} | {'rescale':>8} {'keysw':>8} {'input':>8} | {'cross':>8}")
I_ROWS = []
i0_worst = 0.0
targets = [("ref", 53, 7, 0, n_full)]
for k in range(0, 17):
    ns = 2 ** k
    best = None
    for arc in (0, 3):
        for lD, degb in cells_ext():
            e = evaluate(lD, degb, arc, ns)
            d_ = dec(V(e, "V4+Pg", 1.0)[1], lD)
            if best is None or d_ < best[0]:
                best = (d_, e)
    e = best[1]
    targets.append((f"2^{k}", e["lD"], e["degb"], e["arc"], ns))
for lab, lD, degb, arc, ns in targets:
    lq0 = lD + degb
    eD = model(lD, lq0, DC, arc, ns)["eD"]
    full, em_top = evalmod_split(lD, lq0, arc, ns)
    i0_worst = max(i0_worst, abs(full / eD - 1))
    rs_only = evalmod_split(lD, lq0, arc, ns, 1.0, 0.0, 0.0)[0]
    ks_only = evalmod_split(lD, lq0, arc, ns, 0.0, 1.0, 0.0)[0]
    in_only = evalmod_split(lD, lq0, arc, ns, 0.0, 0.0, 1.0)[0]
    cross = full - rs_only - ks_only - in_only
    margin = bits(C_SPARSE) - bits(bks(em_top, lq0, lD) / 2.0 ** lD)
    cell = f"{lD}/{degb}/{'on' if arc else 'off'}"
    row = (lab, cell, bits(eD), margin, bits(rs_only) - bits(eD), bits(ks_only) - bits(eD),
           bits(in_only) - bits(eD), (bits(cross) - bits(eD)) if cross > 0 else float("-inf"))
    I_ROWS.append(row)
    print(f"    {lab:>5} {cell:>9} | 2^{row[2]:6.2f} | {row[3]:10.2f} | {row[4]:+8.3f} {row[5]:+8.2f} "
          f"{row[6]:+8.3f} | {row[7]:+8.2f}")
gate("I0 split(1,1,1) reproduces eD at every target", i0_worst < 1e-12, f"max rel diff {i0_worst:.1e}")
if FAILS:
    print("\n  PART I GATE FAILED:", FAILS, " -- see the output above.")
    sys.exit(1)

# ---------------------------------------------------------------- PART J (NEW, MODEL: sensitivity annotation)
from numpy.polynomial import chebyshev as _C

_LIP = {}
GRID_J = np.linspace(-1.0, 1.0, 2 ** 17 + 1)


def composite_derivs(Keff, arc, t):
    """p, p', p'' of the port's composite approximant at points t (chain rule)."""
    c = cheb_coeffs(Keff)
    c1 = _C.chebder(c)
    c2 = _C.chebder(c1)
    v, dv, ddv = _C.chebval(t, c), _C.chebval(t, c1), _C.chebval(t, c2)
    for i in range(1 - ELL, 1):
        A = TWOPI ** (-(2.0 ** (i - 1)))
        v, dv, ddv = 2 * v * v - A * A, 4 * v * dv, 4 * (dv * dv + v * ddv)
    if arc:
        a_ = TWOPI ** 2 / 6.0
        v, dv, ddv = (v + a_ * v ** 3, (1 + 3 * a_ * v * v) * dv,
                      6 * a_ * v * dv * dv + (1 + 3 * a_ * v * v) * ddv)
    return v, dv, ddv


def lip(Keff, arc):
    key = (round(Keff, 12), arc)
    if key not in _LIP:
        _, d1, d2 = composite_derivs(Keff, arc, GRID_J)
        _LIP[key] = (float(np.max(np.abs(d1))), float(np.max(np.abs(d2))))
    return _LIP[key]


def eval_J(lD, degb, arc, ns):
    """(tau_J, tau_cur, details) for V4+Pg, kappa = 1."""
    e = evaluate(lD, degb, arc, ns)
    lq0 = lD + degb
    D = 2.0 ** lD
    mo = e["mo"]
    t, tau = V(e, "V4+Pg", 1.0)
    L, L2 = lip(mo["Keff"], arc)
    alpha = mo["B_cts"] / D
    eD_int = evalmod_split(lD, lq0, arc, ns, 1.0, 1.0, 0.0)[0]
    eD_J = eD_int + (L + alpha * L2) * mo["B_cts"] + D * L2 * alpha ** 2 / 2
    eD_use = min(mo["eD"], eD_J)
    noise_J = c_of(ns) * ns * 2.0 ** degb * 1.0 * eD_use
    tau_J = tau - t["noise"] + noise_J
    return tau_J, tau, dict(e=e, L=L, L2=L2, eD=mo["eD"], eD_int=eD_int, eD_J=eD_J,
                            used_old=eD_J >= mo["eD"], noise_J=noise_J, t=t,
                            amp_tag=(mo["eD"] - eD_int) / mo["B_cts"])


print("\nPART J  MODEL (not a result): sensitivity annotation for the EvalMod input error, V4+Pg, kappa = 1")
# J0: chain-rule derivatives vs central finite differences
rngJ = np.random.default_rng(7)
tt = rngJ.uniform(-0.95, 0.95, 64)
okJ0 = True
for Keff_, arc_ in ((32.0 + 2 ** -7, 0), (32.0 + 2 ** -4, 3)):
    h = 1e-6
    p0, d1, d2 = composite_derivs(Keff_, arc_, tt)
    pp, _, _ = composite_derivs(Keff_, arc_, tt + h)
    pm, _, _ = composite_derivs(Keff_, arc_, tt - h)
    fd = (pp - pm) / (2 * h)
    okJ0 &= float(np.max(np.abs(fd - d1)) / np.max(np.abs(d1))) < 1e-6
gate("J0 chain-rule p' matches finite differences", okJ0)
# J1: sanity: composite derivative max is K_eff (sin(2 pi K t)/(2 pi) has derivative K cos)
Lr, _ = lip(e53["mo"]["Keff"], 0)
gate("J1 max|p'| / K_eff within 2% of 1 at the ref cell", abs(Lr / e53["mo"]["Keff"] - 1) < 0.02,
     f"ratio {Lr / e53['mo']['Keff']:.5f}")
if FAILS:
    print("\n  PART J GATE FAILED:", FAILS, " -- see the output above.")
    sys.exit(1)

print(f"    {'n_s':>5} | {'cur dec':>8} {'cell':>9} | {'J dec':>8} {'cell':>9} {'gain':>6} | "
      f"{'logQ':>5} | tag amp  L/Keff | eD -> eD_J at J cell | noise / approx at J cell")
J_ROWS = []
tJ_ref, tc_ref, dref = eval_J(53, 7, 0, n_full)
J_REF = (dec(tc_ref, 53), dec(tJ_ref, 53), dref)
print(f"    {'ref':>5} | {dec(tc_ref, 53):+8.2f} {'53/7/off':>9} | {dec(tJ_ref, 53):+8.2f} {'53/7/off':>9} "
      f"{dec(tc_ref, 53) - dec(tJ_ref, 53):6.2f} | {dref['e']['logQ']:>5} | 2^{bits(dref['amp_tag']):5.2f}  "
      f"{dref['L'] / dref['e']['mo']['Keff']:.4f} | 2^{bits(dref['eD']):.2f} -> 2^{bits(min(dref['eD'], dref['eD_J'])):.2f} | "
      f"2^{bits(dref['noise_J']):.1f} / 2^{bits(dref['t']['approx']):.1f}")
n_old = 0
for k in range(0, 17):
    ns = 2 ** k
    bc, bj = None, None
    for arc in (0, 3):
        for lD, degb in cells_ext():
            tJ, tc, dd = eval_J(lD, degb, arc, ns)
            n_old += dd["used_old"]
            if bc is None or dec(tc, lD) < bc[0]:
                bc = (dec(tc, lD), lD, degb, arc)
            if bj is None or dec(tJ, lD) < bj[0]:
                bj = (dec(tJ, lD), lD, degb, arc, dd)
    dj = bj[4]
    cellc = f"{bc[1]}/{bc[2]}/{'on' if bc[3] else 'off'}"
    cellj = f"{bj[1]}/{bj[2]}/{'on' if bj[3] else 'off'}"
    row = (k, bc[0], cellc, bj[0], cellj, bc[0] - bj[0], dj["e"]["logQ"], bits(dj["amp_tag"]),
           dj["L"] / dj["e"]["mo"]["Keff"], bits(dj["eD"]), bits(min(dj["eD"], dj["eD_J"])),
           bits(dj["noise_J"]), bits(dj["t"]["approx"]))
    J_ROWS.append(row)
    print(f"    2^{k:<3} | {row[1]:+8.2f} {row[2]:>9} | {row[3]:+8.2f} {row[4]:>9} {row[5]:6.2f} | {row[6]:>5} | "
          f"2^{row[7]:5.2f}  {row[8]:.4f} | 2^{row[9]:.2f} -> 2^{row[10]:.2f} | 2^{row[11]:.1f} / 2^{row[12]:.1f}")
print(f"    [INFO] cells where eD_J >= eD (model keeps the old tag): {n_old}")
for thr in (0.0, -5.0, -10.0):
    ok = [r[0] for r in J_ROWS if r[3] < thr]
    print(f"    J threshold T < 2^{thr:g}: " + (f"n_s <= 2^{max(ok)}" if ok else "none"))

# ---------------------------------------------------------------- PART K (NEW: lemma form of PART J)
# eD_K = eD_int(mu) + L_R * B_cts + L_I * B_conj, where
#   B_conj = bks(em_top)/D is the conjugation key switch of the CtS real-part extraction (the only
#            complex part of the EvalMod input error: v + conj(v) turns the slot error into 2 Re e);
#   delta = B_cts/D bounds |Re alpha|, eta = B_conj/D bounds |Im alpha|;
#   L_R  = analytic bound on |p'| over real |t| <= 1 + delta  (lemma sens-real);
#   L_I  = analytic bound on |p'| over the strip |Re z| <= 1 + delta, |Im z| <= eta (lemma sens-strip);
#   eD_int(mu) = EvalMod tag recursion with zero input error and message bounds on the enlarged
#            domain: |T_j(z)| <= cosh(j s_z), double-angle operands <= A_{i-1} cosh(2^idx s_u),
#            arcsine operand |y| <= eps*1.01 + delta*L_R + eta*L_I.
# No grid maxima enter PART K. Every earlier part is untouched.
RHO_K = np.geomspace(1.5, 40.0, 4000)
JJ_K = np.arange(DB + 1, DB + 1 + 400, dtype=float)
_TAILK = {}


def tails_K(Keff, s):
    """E0, E1 >= sup |r|, |r'| on {|x| <= cosh s} (real) and on the Bernstein ellipse E_{e^s}:
    |c_j| <= 2 S(rho) rho^-j and |T_j^(k)| <= j^{2k} e^{js} there (k = 0, 1); plus aliasing."""
    key = (round(Keff, 12), round(s, 15))
    if key not in _TAILK:
        b = TWOPI * Keff / 2 ** ELL
        A = TWOPI ** (-(2.0 ** (-ELL)))
        ok = RHO_K > math.exp(s) * 1.01
        rho = RHO_K[ok]
        S = A * np.cosh(b / 2 * (rho - 1 / rho))
        x = math.exp(s) / rho                              # < 1
        m_ = DB + 1.0
        E0 = 2 * S * x ** m_ / (1 - x)                     # sum_{j>=m} x^j
        E1 = 2 * S * x ** m_ * (m_ ** 2 - (2 * m_ ** 2 - 2 * m_ - 1) * x + (m_ - 1) ** 2 * x ** 2) / (1 - x) ** 3
        i0, i1 = int(np.argmin(E0)), int(np.argmin(E1))
        r = rho[i1]
        jj = np.arange(DB + 1)
        l2a = (math.log2(4 * S[i1]) + (DB - 2 * 2048) * math.log2(r) + DB * s / math.log(2)
               + math.log2(float(np.sum((jj ** 2 + 1) * r ** (jj - DB)))))
        alias = 2.0 ** l2a if l2a > -1000 else 0.0
        _TAILK[key] = (float(E0[i0]) + alias, float(E1[i1]) + alias)
    return _TAILK[key]


_TM = {}


def Tm_at(x):
    if x not in _TM:
        cm = np.zeros(2 ** ELL + 1)
        cm[-1] = 1.0
        c1 = np.polynomial.chebyshev.chebder(cm)
        c2 = np.polynomial.chebyshev.chebder(c1)
        C_ = np.polynomial.chebyshev.chebval
        _TM[x] = (float(C_(x, cm)), float(C_(x, c1)), float(C_(x, c2)))
    return _TM[x]


def s_for_rectangle(xmax, ymax):
    """smallest s (bisection) with the rectangle |x| <= xmax, |y| <= ymax inside E_{e^s}."""
    lo, hi = 1e-18, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if (xmax / math.cosh(mid)) ** 2 + (ymax / math.sinh(mid)) ** 2 <= 1:
            hi = mid
        else:
            lo = mid
    return hi


def sens_bounds(Keff, arc, delta, eta):
    A = TWOPI ** (-(2.0 ** (-ELL)))
    A0 = 1 / TWOPI
    M = 2 ** ELL
    a3 = TWOPI ** 2 / 6.0
    b = TWOPI * Keff / 2 ** ELL
    # real segment |t| <= 1 + delta
    sR = math.acosh(1 + delta)
    E0R, E1R = tails_K(Keff, sR)
    Tm, Tm1, Tm2 = Tm_at(1 + E0R / A)
    Ymax = A0 * Tm
    Qp = 1 + 3 * a3 * Ymax ** 2 if arc else 1.0
    G1 = Qp * (A0 / A) * Tm1
    G2 = (6 * a3 * Ymax * ((A0 / A) * Tm1) ** 2 if arc else 0.0) + Qp * (A0 / A ** 2) * Tm2
    L_R = Keff + G2 * E0R * (A * b) + G1 * E1R
    # strip |Re z| <= 1 + delta, |Im z| <= eta
    sz = s_for_rectangle(1 + delta, max(eta, 1e-300))
    E0z, E1z = tails_K(Keff, sz)
    eps_u = E0z / A
    su = max(math.acosh(math.cosh(b * eta) + eps_u), math.asinh(math.sinh(b * eta) + eps_u))
    Ymax_c = A0 * math.cosh(M * su)
    Qp_c = 1 + 3 * a3 * Ymax_c ** 2 if arc else 1.0
    L_I = Qp_c * (A0 / A) * M * M * math.exp((M - 1) * su) * (A * b * math.cosh(b * eta) + E1z)
    return dict(L_R=L_R, L_I=L_I, sR=sR, sz=sz, su=su, E0R=E0R, E1R=E1R)


def evalmod_split_mu(lD, lq0, arc, ns, sz, su, dy, mu_on=True, dc=DC):
    """evalmod_split(rs=1, ks=1, inp=0) with message bounds on the enlarged domain (mu).
    dy >= |y(actual input) - y(ideal input)| for the double-angle output y (<= delta L_R + eta L_I)."""
    mo = model(lD, lq0, dc, arc, ns)
    D, q0 = 2.0 ** lD, 2.0 ** lq0
    eps = D / q0
    top, _ = towers_layout(arc, dc)
    em_top = top - len(CTS_KS)
    mu = (lambda j: math.cosh(j * sz)) if mu_on else (lambda j: 1.0)
    muda = (lambda idx: math.cosh(2 ** idx * su)) if mu_on else (lambda idx: 1.0)
    KS_ = lambda tw: bks(tw, lq0, lD) / D
    st = {1: 0.0}
    for j in range(2, DB + 1):
        tw = em_top - (depth(j) - 1)
        if j % 2 == 0:
            a_ = j // 2
            st[j] = 4 * mu(a_) * st[a_] + 2 * st[a_] ** 2 / D + C_SPARSE + KS_(tw)
        else:
            a_, b_ = j // 2, j // 2 + 1
            st[j] = (2 * (mu(b_) * st[a_] + mu(a_) * st[b_] + st[a_] * st[b_] / D) + C_SPARSE
                     + KS_(tw) + st[1])
    c = cheb_coeffs(mo["Keff"])
    B = C_SPARSE + sum(abs(c[j]) * (st[j] + (5 - depth(j)) * C_SPARSE) for j in range(1, DB + 1))
    for idx, i in enumerate(range(1 - ELL, 1)):
        A = TWOPI ** (-(2.0 ** (i - 1)))
        tw = em_top - 6 - idx
        B = 4 * A * muda(idx) * B + 2 * B * B / D + C_SPARSE + KS_(tw)
    if arc:
        tw = em_top - 12
        yb = eps * 1.01 + (dy if mu_on else 0.0)
        nu_y = D * yb
        B2 = (2 * nu_y * B + B * B) / D + C_SPARSE + KS_(tw)
        nu_y2 = D * yb ** 2
        a_ = TWOPI ** 2 / 6.0
        B3 = a_ * (nu_y2 * B + nu_y * B2 + B * B2) / D + C_SPARSE + KS_(tw - 1)
        B = B + 2 * C_SPARSE + B3
    return B, em_top


def eval_K(lD, degb, arc, ns):
    e = evaluate(lD, degb, arc, ns)
    lq0 = lD + degb
    D = 2.0 ** lD
    mo = e["mo"]
    t, tau = V(e, "V4+Pg", 1.0)
    top, _ = towers_layout(arc, DC)
    B_conj = bks(top - len(CTS_KS), lq0, lD) / D
    delta, eta = mo["B_cts"] / D, B_conj / D
    sb = sens_bounds(mo["Keff"], arc, delta, eta)
    eD_int, _ = evalmod_split_mu(lD, lq0, arc, ns, sb["sz"], sb["su"], delta * sb["L_R"] + eta * sb["L_I"])
    eD_K = eD_int + sb["L_R"] * mo["B_cts"] + sb["L_I"] * B_conj
    eD_use = min(mo["eD"], eD_K)
    noise_K = c_of(ns) * ns * 2.0 ** degb * eD_use
    return tau - t["noise"] + noise_K, tau, dict(e=e, sb=sb, B_conj=B_conj, eD=mo["eD"], eD_int=eD_int,
                                                  eD_K=eD_K, used_old=eD_K >= mo["eD"], noise_K=noise_K, t=t)


print("\nPART K  V4+PgJ in lemma form (analytic L_R, strip L_I, message bounds on the enlarged domain), kappa = 1")
# K0: with mu off and zero input error, the mu-recursion reproduces PART I's internal part
k0 = 0.0
for (lD, degb, arc, ns) in ((53, 7, 0, n_full), (59, 5, 3, 1), (62, 2, 0, n_full), (60, 4, 3, 2 ** 8)):
    ref_ = evalmod_split(lD, lD + degb, arc, ns, 1.0, 1.0, 0.0)[0]
    got_ = evalmod_split_mu(lD, lD + degb, arc, ns, 0.0, 0.0, 0.0, mu_on=False)[0]
    k0 = max(k0, abs(got_ / ref_ - 1))
gate("K0 mu-recursion with mu = 1 reproduces eD_int of PART I", k0 < 1e-12, f"max rel diff {k0:.1e}")
# K2: ellipse + disk containment used for s_u (sampled)
rngK = np.random.default_rng(11)
okK2 = True
for _ in range(2000):
    h, epsd = 10 ** rngK.uniform(-12, -1), 10 ** rngK.uniform(-14, -2)
    su_ = max(math.acosh(math.cosh(h) + epsd), math.asinh(math.sinh(h) + epsd))
    tq, ph = rngK.uniform(0, 2 * math.pi, 2)
    x = math.cosh(h) * math.cos(tq) + epsd * math.cos(ph)
    y = math.sinh(h) * math.sin(tq) + epsd * math.sin(ph)
    okK2 &= (x / math.cosh(su_)) ** 2 + (y / math.sinh(su_)) ** 2 <= 1 + 1e-12
gate("K2 E_{e^h} + disk(eps) inside E_{e^{s_u}} (2000 samples)", okK2)
_x = np.array([0.05, 0.3, 0.7, 0.93])
_j = np.arange(DB + 1, DB + 1 + 4000, dtype=float)
_m = DB + 1.0
_cf0 = _x ** _m / (1 - _x)
_cf1 = _x ** _m * (_m ** 2 - (2 * _m ** 2 - 2 * _m - 1) * _x + (_m - 1) ** 2 * _x ** 2) / (1 - _x) ** 3
_br0 = (_x[:, None] ** _j).sum(axis=1)
_br1 = (_x[:, None] ** _j * _j ** 2).sum(axis=1)
gate("K3 closed-form tail sums = brute-force sums", np.allclose(_cf0, _br0, rtol=1e-10) and np.allclose(_cf1, _br1, rtol=1e-10))
if FAILS:
    print("\n  PART K GATE FAILED:", FAILS, " -- see the output above.")
    sys.exit(1)

print(f"    {'n_s':>5} | {'cur dec':>8} {'cell':>9} | {'K dec':>8} {'cell':>9} {'gain':>6} | {'logQ':>5} | "
      f"L_R/Keff-1  L_I     conj/cts | eD -> eD_K | noise / approx at K cell")
K_ROWS = []
okK1, n_oldK = True, 0


def k_row(lab, bc, bk):
    dk = bk[4]
    sb = dk["sb"]
    Keff = dk["e"]["mo"]["Keff"]
    return (lab, bc[0], f"{bc[1]}/{bc[2]}/{'on' if bc[3] else 'off'}", bk[0],
            f"{bk[1]}/{bk[2]}/{'on' if bk[3] else 'off'}", bc[0] - bk[0], dk["e"]["logQ"],
            bits(sb["L_R"] / Keff - 1), bits(sb["L_I"]), bits(dk["B_conj"]) - bits(dk["e"]["mo"]["B_cts"]),
            bits(dk["eD"]), bits(min(dk["eD"], dk["eD_K"])), bits(dk["noise_K"]), bits(dk["t"]["approx"]),
            bits(dk["eD_K"]) - bits(dk["eD_int"]))


tK, tc, dK = eval_K(53, 7, 0, n_full)
okK1 &= dK["sb"]["L_R"] >= lip(dK["e"]["mo"]["Keff"], 0)[0]
K_REF = k_row("ref", (dec(tc, 53), 53, 7, 0), (dec(tK, 53), 53, 7, 0, dK))
for k in range(0, 17):
    ns = 2 ** k
    bc, bk = None, None
    for arc in (0, 3):
        for lD, degb in cells_ext():
            tk_, tc_, dd = eval_K(lD, degb, arc, ns)
            n_oldK += dd["used_old"]
            if bc is None or dec(tc_, lD) < bc[0]:
                bc = (dec(tc_, lD), lD, degb, arc)
            if bk is None or dec(tk_, lD) < bk[0]:
                bk = (dec(tk_, lD), lD, degb, arc, dd)
    okK1 &= bk[4]["sb"]["L_R"] >= lip(bk[4]["e"]["mo"]["Keff"], bk[3])[0]
    K_ROWS.append(k_row(f"2^{k}", bc, bk))
gate("K1 analytic L_R >= grid max |p'| at every optimum", okK1)
if FAILS:
    print("\n  PART K GATE FAILED:", FAILS, " -- see the output above.")
    sys.exit(1)
for r in [K_REF] + K_ROWS:
    print(f"    {r[0]:>5} | {r[1]:+8.2f} {r[2]:>9} | {r[3]:+8.2f} {r[4]:>9} {r[5]:6.2f} | {r[6]:>5} | "
          f"2^{r[7]:7.2f}  2^{r[8]:5.2f} {r[9]:+7.2f} | 2^{r[10]:.2f} -> 2^{r[11]:.2f} | 2^{r[12]:.1f} / 2^{r[13]:.1f}")
print(f"    [INFO] cells where eD_K >= eD (old tag kept): {n_oldK}")
JK = {r[0]: r for r in J_ROWS}
dmax = max(abs(r[3] - JK[int(r[0][2:])][3]) for r in K_ROWS)
print(f"    [INFO] max |K dec - J dec| over packings: {dmax:.4f} bits")
for thr in (0.0, -5.0, -10.0):
    ok = [int(r[0][2:]) for r in K_ROWS if r[3] < thr]
    print(f"    K threshold T < 2^{thr:g}: " + (f"n_s <= 2^{max(ok)}" if ok else "none"))

# ---------------------------------------------------------------- PART L (NEW: ablation + term tables under K)
print("\nPART L  under the K bound (kappa = 1): (a) ablation without Lemma parseval, (b) term tables")


def eval_K_noP(lD, degb, arc, ns):
    """K noise with the sup-norm sine term (V4c) instead of Lemma parseval."""
    tK, tc, dd = eval_K(lD, degb, arc, ns)
    t4c, tau4c = V(dd["e"], "V4c", 1.0)
    return tau4c - t4c["noise"] + dd["noise_K"], dd


L_ROWS = []
print(f"    {'n_s':>5} | {'K+Pars':>8} {'cell':>9} | {'K noPars':>8} {'cell':>9} | {'Pars gain':>9}")
for k in range(0, 17):
    ns = 2 ** k
    bp, bn = None, None
    for arc in (0, 3):
        for lD, degb in cells_ext():
            tk_, _, _ = eval_K(lD, degb, arc, ns)
            tn_, _ = eval_K_noP(lD, degb, arc, ns)
            if bp is None or dec(tk_, lD) < bp[0]:
                bp = (dec(tk_, lD), f"{lD}/{degb}/{'on' if arc else 'off'}")
            if bn is None or dec(tn_, lD) < bn[0]:
                bn = (dec(tn_, lD), f"{lD}/{degb}/{'on' if arc else 'off'}")
    L_ROWS.append((k, bp[0], bp[1], bn[0], bn[1], bn[0] - bp[0]))
    print(f"    2^{k:<3} | {bp[0]:+8.2f} {bp[1]:>9} | {bn[0]:+8.2f} {bn[1]:>9} | {bn[0] - bp[0]:9.2f}")


def term_table(lab, lD, degb, arc, ns):
    tK, _, dd = eval_K(lD, degb, arc, ns)
    e, t = dd["e"], dd["t"]
    D = 2.0 ** lD
    sb = dd["sb"]
    row = dict(lab=lab, cell=f"{lD}/{degb}/{'on' if arc else 'off'}", ns=ns, dec=dec(tK, lD), logQ=e["logQ"],
               noise=dd["noise_K"], interp=e["interp"], sine=e["sine_p"], stc=t["StC inj"], bsd=t["B_sd"],
               conj=t["conj"], post=t.get("post", 0.0), eD_old=dd["eD"], eD_int=dd["eD_int"],
               lip_R=sb["L_R"] * e["mo"]["B_cts"], lip_I=sb["L_I"] * dd["B_conj"], eD_K=min(dd["eD"], dd["eD_K"]),
               B_cts=e["mo"]["B_cts"], B_conj=dd["B_conj"], L_R=sb["L_R"], L_I=sb["L_I"],
               Keff=e["mo"]["Keff"], B_in_over_D=e["mo"]["B_in"] / D)
    return row


TT = [term_table("ref", 53, 7, 0, n_full)]
for k, lab in ((0, "1 slot"), (3, "2^3"), (8, "2^8"), (16, "full")):
    rowk = [r for r in K_ROWS if r[0] == f"2^{k}"][0]
    lD_, degb_, arc_ = rowk[4].split("/")
    TT.append(term_table(lab, int(lD_), int(degb_), 3 if arc_ == "on" else 0, 2 ** k))
print("\n    term tables (polynomial units at scale Delta; log2):")
for r in TT:
    print(f"    {r['lab']:>6} {r['cell']:>9} logQ {r['logQ']} dec {r['dec']:+.2f} | noise 2^{bits(r['noise']):.2f} "
          f"interp 2^{bits(r['interp']):.2f} sine 2^{bits(r['sine']):.2f} StCinj 2^{bits(r['stc']):.2f} "
          f"B_sd 2^{bits(r['bsd']):.2f} conj 2^{bits(r['conj']):.2f} post 2^{bits(r['post']):.2f}")
    print(f"    {'':>6} {'':>9} eD_old 2^{bits(r['eD_old']):.3f} -> eD_K 2^{bits(r['eD_K']):.3f} = eD_int 2^{bits(r['eD_int']):.3f}"
          f" + L_R B_cts 2^{bits(r['lip_R']):.2f} + L_I B_conj 2^{bits(r['lip_I']):.2f} | B_cts 2^{bits(r['B_cts']):.3f}"
          f" B_conj 2^{bits(r['B_conj']):.2f} L_R {r['L_R']:.7f} (K_eff {r['Keff']:.7f}) L_I 2^{bits(r['L_I']):.2f}")

# ---------------------------------------------------------------- summary block
print("\n" + "=" * 100)
print("SUMMARY BEGIN")
print(f"gate: B_CtS 2^{bits(m53['B_cts']):.3f} eps_D 2^{bits(m53['eD']):.3f} V4 {h4:+.2f} V5 {h5:+.2f}; "
      f"0b checked {n_chk}, Parseval n/a at {len(no_p)} cells, max eps_in 2^{bits(worst_eps):.2f}")
print(f"ref: B_in 2^{bits(e53['mo']['B_in']):.3f} eps_in 2^{bits(e53['eps_in']):.3f} "
      f"interp 2^{bits(e53['interp']):.2f} sine_sup 2^{bits(e53['sine_sup']):.2f} "
      f"sine_P 2^{bits(e53['sine_p']):.2f} conj 2^{bits(e53['mo']['B_cj']):.2f}")
for lab, d_, t in REF:
    print(f"ref {lab}: dec={d_:+.2f} terms {terms_str(t)}")
for r in best_rows:
    print(f"best {r[0]} n_s={r[1]} arc={'on' if r[2] else 'off'}: logD={r[3]} degb={r[4]} "
          f"dec={r[5]:+.2f} logQ={r[6]} sec~{r[7]:.1f} pmin={r[8]} terms {terms_str(r[9])}")
for lab, arc, g1, x1, g8, x8, ratio in D_ROWS:
    print(f"gap {lab} arc={'on' if arc else 'off'}: full-1={g1:.2f} ({x1:+.2f} vs {L_FULL1:.2f}) "
          f"full-2^8={g8:.2f} ({x8:+.2f} vs {L_FULL8:.2f}) approx/noise@full=2^{ratio:+.1f}")
for r in E_PROF:
    print(f"E profile degb={r[0]} logD={r[1]}: V4c k2 {r[2]:+.2f}  V4+Pg k2 {r[3]:+.2f}  V4+Pg k1 {r[4]:+.2f}")
for r in E_ROWS:
    print(f"E best {r[0]} n_s={r[1]} arc={'on' if r[2] else 'off'}: logD={r[3]} degb={r[4]} "
          f"dec={r[5]:+.2f} logQ={r[6]} sec~{r[7]:.1f} pmin={r[8]} terms {terms_str(r[9])} {r[10]}")
for k, r, b2, b1 in F_ROWS:
    print(f"F n_s=2^{k}: k2 off {r[(2.0, 0)]:+.2f} on {r[(2.0, 3)]:+.2f} | k1 off {r[(1.0, 0)]:+.2f} on {r[(1.0, 3)]:+.2f} "
          f"| best k2 {b2[1]['lD']}/{b2[1]['degb']}/{'on' if b2[1]['arc'] else 'off'} logQ={b2[1]['logQ']} "
          f"| best k1 {b1[1]['lD']}/{b1[1]['degb']}/{'on' if b1[1]['arc'] else 'off'} logQ={b1[1]['logQ']}")
for kap, idx in ((2.0, 2), (1.0, 3)):
    for thr in (0.0, -10.0):
        ok = [r[0] for r in F_ROWS if r[idx][0] < thr]
        print(f"F threshold kappa={kap:g} T<2^{thr:g}: " + (f"n_s<=2^{max(ok)}" if ok else "none"))
for r in G_ROWS:
    print(f"G n_s=2^{r[0]}: off dec={r[1]:+.2f} {r[2]} logQ={r[3]} | on dec={r[4]:+.2f} {r[5]} logQ={r[6]} "
          f"| E_ss/k=2^{r[7]:.1f} B_cts 2^{r[8]:.3f}->2^{r[9]:.3f}")
for r in H_ROWS:
    print(f"H n_s=2^{r[0]}: off dec={r[1]:+.3f} {r[2]} | on dec={r[3]:+.3f} {r[4]} | delta={r[5]:+.4f} "
          f"| post 2^{r[6]:.1f} ({r[7]:.1f} bits below noise)")
for r in I_ROWS:
    print(f"I {r[0]} {r[1]}: eD 2^{r[2]:.2f} | inj margin {r[3]:.2f} bits | share rescale {r[4]:+.3f} "
          f"keysw {r[5]:+.2f} input {r[6]:+.3f} cross {r[7]:+.2f}  (log2 part/eD)")
print(f"J ref 53/7/off: cur {J_REF[0]:+.2f} -> J {J_REF[1]:+.2f} | tag amp 2^{bits(J_REF[2]['amp_tag']):.2f} "
      f"L/Keff {J_REF[2]['L'] / J_REF[2]['e']['mo']['Keff']:.4f} L2 2^{bits(J_REF[2]['L2']):.2f} "
      f"eD 2^{bits(J_REF[2]['eD']):.2f} eD_int 2^{bits(J_REF[2]['eD_int']):.2f} eD_J 2^{bits(J_REF[2]['eD_J']):.2f}")
for r in J_ROWS:
    print(f"J n_s=2^{r[0]}: cur {r[1]:+.2f} {r[2]} -> J {r[3]:+.2f} {r[4]} gain {r[5]:.2f} logQ={r[6]} | "
          f"tag amp 2^{r[7]:.2f} L/Keff {r[8]:.4f} | eD 2^{r[9]:.2f} -> 2^{r[10]:.2f} | noise 2^{r[11]:.1f} approx 2^{r[12]:.1f}")
for thr in (0.0, -5.0, -10.0):
    ok = [r[0] for r in J_ROWS if r[3] < thr]
    print(f"J threshold T<2^{thr:g}: " + (f"n_s<=2^{max(ok)}" if ok else "none"))
print("J is a MODEL: grid maxima of |p'|, |p''|, not certified; no lemma yet")
for r in [K_REF] + K_ROWS:
    print(f"K {r[0]}: cur {r[1]:+.2f} {r[2]} -> K {r[3]:+.2f} {r[4]} gain {r[5]:.2f} logQ={r[6]} | "
          f"L_R/Keff-1 2^{r[7]:.2f} L_I 2^{r[8]:.2f} conj-cts {r[9]:+.2f} | eD 2^{r[10]:.2f} -> 2^{r[11]:.2f} "
          f"(eD_K - eD_int {r[14]:+.3f}) | noise 2^{r[12]:.1f} approx 2^{r[13]:.1f}")
print(f"K max |K dec - J dec| = {dmax:.4f} bits; cells keeping the old tag: {n_oldK}")
for thr in (0.0, -5.0, -10.0):
    ok = [int(r[0][2:]) for r in K_ROWS if r[3] < thr]
    print(f"K threshold T<2^{thr:g}: " + (f"n_s<=2^{max(ok)}" if ok else "none"))
for r in L_ROWS:
    print(f"L n_s=2^{r[0]}: K+Parseval {r[1]:+.2f} {r[2]} | K noParseval {r[3]:+.2f} {r[4]} | Parseval gain {r[5]:.2f}")
for r in TT:
    print(f"LT {r['lab']} {r['cell']} logQ={r['logQ']} dec={r['dec']:+.2f} | noise 2^{bits(r['noise']):.2f} "
          f"interp 2^{bits(r['interp']):.2f} sine 2^{bits(r['sine']):.2f} StCinj 2^{bits(r['stc']):.2f} "
          f"B_sd 2^{bits(r['bsd']):.2f} conj 2^{bits(r['conj']):.2f} post 2^{bits(r['post']):.2f} | "
          f"eD_old 2^{bits(r['eD_old']):.3f} eD_K 2^{bits(r['eD_K']):.3f} eD_int 2^{bits(r['eD_int']):.3f} "
          f"LRB 2^{bits(r['lip_R']):.2f} LIB 2^{bits(r['lip_I']):.2f} B_cts 2^{bits(r['B_cts']):.3f} "
          f"B_conj 2^{bits(r['B_conj']):.2f} L_R {r['L_R']:.7f} Keff {r['Keff']:.7f} L_I 2^{bits(r['L_I']):.2f}")
print(f"KEFF_FIX={KEFF_FIX} SUBSUM={SUBSUM} POSTFIX={POSTFIX}")
print("GATE FAILURES:", FAILS if FAILS else "none")
print("SUMMARY END")
print("=" * 100)
print("""
READING IT
  - 'dec err' is the certified decoded slot error, log2.
  - V4 is the legacy sup-norm port (gated).
    V4c adds the two conjugation key switches. V4+P is V4c with Lemma parseval
    applied to the sine term at the ideal input: every term is covered by a proved
    lemma, provided the paper states Lemma parseval with max |f|/theta^2 (arcsine).
  - kappa = 1 vs 2 is a sensitivity column; which one is right is a proof question
    (does kappa double-count the admitted remainder range already inside c_rs?).
  - V5 needs an l2-tag lemma that is not written yet: model only.
  - sparse rows keep the full-packing CtS structure and chain, plus the SubSum level
    when SUBSUM = True (logQ += lD); PART G shows what SubSum changes.
  - sec~ and pmin are flags, not results.
""")
