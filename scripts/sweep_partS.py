#!/usr/bin/env python3
"""
sweep_partS.py  --  PART S (1 Oct 2026): sequential composition of certified refreshes, and the
projection level of sparse packing at the certified optima.

HOW IT RUNS
  It runs sweep_partQ.py silently (Q runs P, O and the sweep; ~8 min on a laptop) and reuses its
  certified residues and optima. Then one extra certification pass (~85 encodings, ~1-3 min) and
  the chains (~2-4 min).
      python3 /path/to/sweep_partS.py   (next to sweep_partQ.py, sweep_partP.py, sweep_partO.py
                                         and sweep_degb_delta_arcsine.py)
  NaN/overflow warnings from chebyshev.py come from rejected alternatives inside P/Q, as before.

MODEL DECISIONS (1 Oct)
  M1 Routing of the inherited error (Lemma refresh as stated). The ideal EvalMod input contains the
     phase w = m + e (Lemma cts (ii), exact ModRaise), so the CtS tag starts at 0 (+ E_ss/k for
     sparse packing) and the input tag is charged once, as the term B/Delta of (eq:T) ("inh").
     The port instead seeds the CtS tag with B_in and has no B/Delta term; that is a valid upper
     bound (it charges about A_StC (L_R/K_eff) B_in/Delta >= B_in/Delta) and is kept as mode "port",
     both for gate S0a and as a comparison column of the chains.
  M2 Arcsine seed. The port seeds the arcsine operand with 1.01 eps. PART S uses the bound
     at the ideal input, eps Z' + Lambda(E) = (Delta + B)/q0 + Lambda(E) ("honest"), because B/Delta
     = T_{k-1} can exceed 0.01. The EvalMod recursions are copied verbatim from PART P with the seed
     as a parameter (gate S0b).
  M3 One declared K_eff per chain. Refresh k needs K_eff >= K + a_k, a_k = (Delta + B_k)/q0
     (Lemma cts, Lemma wrap with nu = Delta). The chain declares K_eff = K + eps (1 + beta) once,
     beta = g_c(tau)/Delta * (1 + 2^-10), where g_c is the intermediate circuit's tag rule below
     evaluated at the largest admissible input Delta tau (for the identity g_c(tau)/Delta ~ tau; for
     2x^2-1 it is 4 tau + 2 tau^2 + ..., which is why beta is not literally L_f tau). The Chebyshev
     fit, Lambda(E), L_R, L_I and the CtS front scale S = Delta/(q0 K_eff 2^17) all use that K_eff;
     S changes the encoded CtS diagonals, so their R_level is RE-CERTIFIED at every declared K_eff
     (PART O's Certifier, same primes, same margin).
  M4 Intermediate circuit between refreshes, under the dense key, one level at 2 towers, with KS_ds
     before the last rescale as in the port's client circuit (c_rs(h) kept, as the port does):
       id : multiply by the chain prime exactly, KS_ds, rescale
            g(B) = B + c_rs(h) + bks(2)/Delta
       sq : x -> 2x^2 - 1 (|u| <= 1 stays, |f'| <= 4 on [-1, 1]), relin + KS_ds at 2 towers
            g(B) = (4 + 2B/Delta) B + c_rs(h) + 2 bks(2)/Delta
            (the constant 1 at scale Delta is exact in the port's model, where p = Delta).
  M5 Chain. Refresh 1 takes the port's fresh input tag (client circuit + KS_ds). Refresh k >= 2 takes
     B_k = g_c(Delta T_{k-1}). Refresh k is admissible iff B_k/Delta <= beta (K_eff), a_k < 1/2
     (Lemma wrap K = H/2 and Lemma parseval, the same condition at nu = Delta), and passes iff
     T_k <= tau. Every packing is evaluated at its certified single-refresh optimum (tab:packing).

GATES (any FAIL stops the run)
  S0a mode port (port routing, seed 1.01 eps, K_eff = K + (Delta + B_in)/q0, Q's R_level) = Q's
      eval_KQ at the 17 optima + 40 certified random instances (1e-12), and the 17 optimum decs = Q/P.
  S0b the seeded EvalMod copies with seed = 1.01 eps = PART P's evalmod_old / evalmod_mu (40 cases).
  S0c re-certification through PART S's job builder at the locked K_eff = RQ at the 17 optima (1e-12).
  S0d at k = 1 with the locked K_eff, Q's R_level and seed 1.01 eps: lemma routing <= port routing at
      the 17 optima (the argument of M1).
  S0e every rounding residue of the new certification |r_k| <= 1/2 + 1e-9.

WHAT IT PRINTS
  S1  k = 1 at every optimum: locked dec; lemma routing; + honest seed; + declared K_eff (id, 2^-10)
      with re-certified R_level (the chain's first refresh).
  S2  projection level at the certified optimum of every sparse packing: B_cts off / on (4 decimals),
      the difference in bits, E_ss/k, dec off / on.
  S3  chains: circuit x tau x packing: refreshes that pass, T_1, T_2, (T_2 - T_1)/T_1, last passing T,
      what stops the chain and the margins at the last passing refresh, the same count with port
      routing, and max over the chain of honest seed / (1.01 eps).
"""
import contextlib
import io
import math
import os
import runpy
import sys
import time

import numpy as np

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
PART_Q = os.path.join(HERE, "sweep_partQ.py")
if not os.path.isfile(PART_Q):
    print(f"Cannot find {PART_Q}. Put sweep_partS.py next to sweep_partQ.py, P, O and the sweep.")
    sys.exit(1)
print("=" * 100)
print("sweep_partS.py   PART S: sequential composition (Lemma refresh routing) and the projection level")
print("=" * 100)
print("  running sweep_partQ.py silently (Q, P, O and the sweep; ~8 min) ...", flush=True)
_buf = io.StringIO()
try:
    with contextlib.redirect_stdout(_buf):
        Q = runpy.run_path(PART_Q, run_name="partQ")
except SystemExit:
    print(_buf.getvalue())
    print("\nsweep_partQ.py stopped at one of its gates (output above). Fix that first.")
    sys.exit(1)
if Q["FAILS"]:
    print(_buf.getvalue())
    print("\nsweep_partQ.py reports gate failures:", Q["FAILS"])
    sys.exit(1)
print(f"  sweep_partQ.py: all gates passed ({time.time() - T0:.0f} s)", flush=True)

P, O = Q["P"], Q["O"]
N, n_full, K, H, DC = O["N"], O["n_full"], O["K"], O["H"], O["DC"]
ELL, DB, E0, TWOPI = O["ELL"], O["DB"], O["E0"], O["TWOPI"]
CTS_KS, CTS_ROT, R_STC, SUBSUM = O["CTS_KS"], O["CTS_ROT"], O["R_STC"], O["SUBSUM"]
C_DENSE, C_SPARSE = O["C_DENSE"], O["C_SPARSE"]
bks, depth, cheb_coeffs, q0lam = O["bks"], O["depth"], O["cheb_coeffs"], O["q0lam"]
parseval_factor, c_of, part_of = O["parseval_factor"], O["c_of"], O["part_of"]
amp_inj, amp_after_first, towers_layout = O["amp_inj"], O["amp_after_first"], O["towers_layout"]
sens_bounds, bits, dec = O["sens_bounds"], O["bits"], O["dec"]
model_P, stc_residue, BASE, OPT_P = P["model_P"], P["stc_residue"], P["BASE"], P["OPT"]
evalmod_old_P, evalmod_mu_P = P["evalmod_old"], P["evalmod_mu"]
RQ, rlev_q, eval_KQ, BEST_Q, BN = Q["RQ"], Q["rlev_q"], Q["eval_KQ"], Q["BEST"], Q["BN"]
FAILS = []
np.seterr(over="ignore", invalid="ignore")
TAUS = (2.0 ** -5, 2.0 ** -10)
CIRCS = ("id", "sq")
KMAX = 4096
BETA_SLACK = 1.0 + 2.0 ** -10


def gate(name, ok, detail=""):
    print(f"    [{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def stop_if_failed(part):
    if FAILS:
        print(f"\n  {part} GATE FAILED:", FAILS, " -- see the output above.")
        sys.exit(1)


# =============================================================================================
# seeded copies of PART P's EvalMod recursions (only the arcsine seed eps * 1.01 -> seed)
# =============================================================================================
def scal_res(nu_over_D, B, D):
    return 0.5 * (nu_over_D * D + B) / D


def evalmod_old_s(lD, lq0, arc, Keff, B_cts, CS, em_top, scal, seed):
    D, q0 = 2.0 ** lD, 2.0 ** lq0
    st = {1: B_cts}
    for j in range(2, DB + 1):
        tw = em_top - (depth(j) - 1)
        if j % 2 == 0:
            a = j // 2
            st[j] = 4 * st[a] + 2 * st[a] ** 2 / D + CS + bks(tw, lq0, lD) / D
        else:
            a, b = j // 2, j // 2 + 1
            st[j] = 2 * (st[a] + st[b] + st[a] * st[b] / D) + CS + bks(tw, lq0, lD) / D + st[1]
    c = cheb_coeffs(Keff)
    add = sum(scal_res(1.0, st[j], D) for j in range(1, DB + 1)) if scal else 0.0
    B = CS + sum(abs(c[j]) * (st[j] + (5 - depth(j)) * CS) for j in range(1, DB + 1)) + add
    for idx, i in enumerate(range(1 - ELL, 1)):
        A = TWOPI ** (-(2.0 ** (i - 1)))
        tw = em_top - 6 - idx
        B = 4 * A * B + 2 * B * B / D + CS + bks(tw, lq0, lD) / D + (0.5 if scal else 0.0)
    if arc:
        tw = em_top - 12
        nu_y = D * seed
        B2 = (2 * nu_y * B + B * B) / D + CS + bks(tw, lq0, lD) / D
        nu_y2 = D * seed ** 2
        a = TWOPI ** 2 / 6.0
        r_a = scal_res(seed * seed, B2, D) if scal else 0.0
        B3 = a * (nu_y2 * B + nu_y * B2 + B * B2) / D + CS + bks(tw - 1, lq0, lD) / D + r_a
        B = B + 2 * CS + B3
    return B


def evalmod_mu_s(lD, lq0, arc, Keff, CS, em_top, sz, su, dy, scal, seed):
    D, q0 = 2.0 ** lD, 2.0 ** lq0
    mu = lambda j: math.cosh(j * sz)
    muda = lambda idx: math.cosh(2 ** idx * su)
    KS_ = lambda tw: bks(tw, lq0, lD) / D
    st = {1: 0.0}
    for j in range(2, DB + 1):
        tw = em_top - (depth(j) - 1)
        if j % 2 == 0:
            a_ = j // 2
            st[j] = 4 * mu(a_) * st[a_] + 2 * st[a_] ** 2 / D + CS + KS_(tw)
        else:
            a_, b_ = j // 2, j // 2 + 1
            st[j] = 2 * (mu(b_) * st[a_] + mu(a_) * st[b_] + st[a_] * st[b_] / D) + CS + KS_(tw) + st[1]
    c = cheb_coeffs(Keff)
    add = sum(scal_res(mu(j), st[j], D) for j in range(1, DB + 1)) if scal else 0.0
    B = CS + sum(abs(c[j]) * (st[j] + (5 - depth(j)) * CS) for j in range(1, DB + 1)) + add
    for idx, i in enumerate(range(1 - ELL, 1)):
        A = TWOPI ** (-(2.0 ** (i - 1)))
        tw = em_top - 6 - idx
        B = 4 * A * muda(idx) * B + 2 * B * B / D + CS + KS_(tw) + (0.5 if scal else 0.0)
    if arc:
        tw = em_top - 12
        yb = seed + dy
        nu_y = D * yb
        B2 = (2 * nu_y * B + B * B) / D + CS + KS_(tw)
        nu_y2 = D * yb ** 2
        a_ = TWOPI ** 2 / 6.0
        r_a = scal_res(yb * yb, B2, D) if scal else 0.0
        B3 = a_ * (nu_y2 * B + nu_y * B2 + B * B2) / D + CS + KS_(tw - 1) + r_a
        B = B + 2 * CS + B3
    return B


# =============================================================================================
# the refresh with an external input tag (PART P's model_P + Q's eval_KQ, base options)
# =============================================================================================
def fresh_B_in(lD, lq0):
    """the port's input tag: client circuit of depth DC (squarings, |z| <= 1) and KS_ds."""
    D = 2.0 ** lD
    B = N * E0 + N / 2.0
    for i in range(DC):
        B = (2 + B / D) * B + C_DENSE + bks(DC + 1 - i, lq0, lD) / D
    return B + bks(2, lq0, lD) / D


def keff_port(lD, lq0):
    return K + (2.0 ** lD + fresh_B_in(lD, lq0)) / 2.0 ** lq0


def model_S(lD, lq0, arc, ns, Rlev, B_in, Keff, routing, seed, scal=True, ss=None):
    ss = SUBSUM if ss is None else ss
    ss_ns = ns if (ss and ns < n_full) else None
    Rlev = [0.0] * len(CTS_KS) if Rlev is None else Rlev
    D, q0 = 2.0 ** lD, 2.0 ** lq0
    p = D
    top, dem = towers_layout(arc, DC)
    B = B_in if routing == "port" else 0.0        # M1
    E_ss, extra_lvl = 0.0, 0
    if ss_ns is not None:
        kss = N // (2 * ss_ns)
        E_ss = (kss - 1) * bks(top + 1, lq0, lD) / D + C_SPARSE
        B += E_ss / kss
        extra_lvl = 1
    nu = N * q0 * (1 + H) / 2.0 + D + B_in
    add_cts = 0.0
    S = D / (q0 * Keff * 2.0 ** 16 * 2)
    s1 = S ** (1.0 / len(CTS_KS))                 # spread fold (BASE)
    for i, k in enumerate(CTS_KS):
        res = Rlev[i] * (nu + B) / p
        B = 2.0 ** k * s1 * B + C_SPARSE + CTS_ROT[i] * bks(top - i, lq0, lD) / D + res
        add_cts = 2.0 ** k * s1 * add_cts + res
        nu = 2.0 ** k * s1 * nu
    B = B + C_SPARSE
    em_top = top - len(CTS_KS)
    B_conj = bks(em_top, lq0, lD) / D
    B_sdb = 0.0
    B_cts = 2 * B + B_conj + B_sdb
    CS = C_SPARSE
    eD = evalmod_old_s(lD, lq0, arc, Keff, B_cts, CS, em_top, scal, seed)
    stc_top = 1 + DC + R_STC
    return dict(B_cts=B_cts, eD=eD, B_sd=bks(stc_top, lq0, lD) / D, logQ=lq0 + (top - 1 + extra_lvl) * lD,
                top=top, Keff=Keff, B_in=B_in, B_cj=bks(stc_top + 1, lq0, lD) / D, E_ss=E_ss, B_conj=B_conj,
                B_sdb=B_sdb, CS=CS, em_top=em_top, add_cts=2 * add_cts, S=S, s1=s1)


def seed_of(lD, lq0, B_in, Keff, mode):
    eps = 2.0 ** lD / 2.0 ** lq0
    if mode == "port":
        return eps * 1.01
    return (2.0 ** lD + B_in) / 2.0 ** lq0 + q0lam(lq0, Keff) / 2.0 ** lq0      # eps Z' + Lambda(E)


def eval_S(lD, degb, arc, ns, Rlev, B_in, Keff, routing, seed_mode, ss=None):
    lq0 = lD + degb
    D, q0 = 2.0 ** lD, 2.0 ** lq0
    eps_in = (D + B_in) / q0
    if eps_in >= 0.5:
        return None
    seed = seed_of(lD, lq0, B_in, Keff, seed_mode)
    mo = model_S(lD, lq0, arc, ns, Rlev, B_in, Keff, routing, seed, True, ss)
    A_sup = c_of(ns) * ns
    back = 2.0 ** degb
    Pp = part_of(ns)
    QL = q0lam(lq0, Keff)
    t = dict(approx=A_sup * QL + q0 * parseval_factor(eps_in, arc),
             stc=C_DENSE * amp_inj(Pp), bsd=mo["B_sd"] * amp_after_first(Pp),
             conj=ns * back * mo["B_cj"], post=(2 * ns - A_sup) * back * mo["CS"], mism=0.0)
    if routing == "lemma":
        t["inh"] = B_in                                   # B/Delta of (eq:T), polynomial units
    B_im = mo["B_conj"] + mo["B_sdb"]
    delta, eta = mo["B_cts"] / D, B_im / D
    try:
        if math.exp(math.acosh(1 + delta)) * 1.01 >= 40.0:
            raise ValueError("rho grid")
        sb = sens_bounds(Keff, arc, delta, eta)
        eD_int = evalmod_mu_s(lD, lq0, arc, Keff, mo["CS"], mo["em_top"], sb["sz"], sb["su"],
                              delta * sb["L_R"] + eta * sb["L_I"], True, seed)
        eD_K = eD_int + sb["L_R"] * mo["B_cts"] + sb["L_I"] * B_im
    except (ValueError, OverflowError):
        eD_int = eD_K = math.inf
    t["noise"] = A_sup * back * min(mo["eD"], eD_K)
    t["stc_res"] = stc_residue(Pp, D, B_in, sum(t.values()))
    tot = sum(t.values())
    if not math.isfinite(tot):
        tot = math.inf
    return tot, dict(t=t, mo=mo, seed=seed, eps=D / q0, eps_in=eps_in)


def circuit_tag(c, B, lq0, lD):
    D = 2.0 ** lD
    if c == "id":
        return B + C_DENSE + bks(2, lq0, lD) / D
    if c == "sq":
        return (4 + 2 * B / D) * B + C_DENSE + 2 * bks(2, lq0, lD) / D
    raise ValueError(c)


cellf = lambda b: f"{b[1]}/{b[2]}/{'on' if b[3] else 'off'}"
OPT = {j: (BEST_Q[(BN, 2 ** j)] if j < 16 else OPT_P[16]) for j in range(17)}

# =============================================================================================
print("\nPART S0  GATES (model)", flush=True)
rng = np.random.default_rng(20261005)
keys_rq = sorted(RQ)
pick = [keys_rq[i] for i in rng.choice(len(keys_rq), min(40, len(keys_rq)), replace=False)]
cases = [(OPT[j][1], OPT[j][2], OPT[j][3], 2 ** j) for j in range(17)] + \
        [(k_[0], k_[1], int(rng.choice([0, 3])), k_[2]) for k_ in pick]
wa = 0.0
for lD, degb, arc, ns in cases:
    lq0 = lD + degb
    Rl = rlev_q(lD, degb, ns, BASE, True)
    r1 = eval_KQ(lD, degb, arc, ns, BASE, Rl)
    r2 = eval_S(lD, degb, arc, ns, Rl, fresh_B_in(lD, lq0), keff_port(lD, lq0), "port", "port")
    if r1 is None or r2 is None:
        if (r1 is None) != (r2 is None):
            wa = math.inf
        continue
    if math.isfinite(r1[0]) or math.isfinite(r2[0]):
        wa = max(wa, abs(r2[0] / r1[0] - 1))
wo = max(abs(dec(eval_S(OPT[j][1], OPT[j][2], OPT[j][3], 2 ** j, rlev_q(OPT[j][1], OPT[j][2], 2 ** j, BASE, True),
                        fresh_B_in(OPT[j][1], OPT[j][1] + OPT[j][2]), keff_port(OPT[j][1], OPT[j][1] + OPT[j][2]),
                        "port", "port")[0], OPT[j][1]) - OPT[j][0]) for j in range(17))
gate("S0a mode port = Q's eval_KQ (17 optima + 40 certified instances) and = the 17 optimum decs",
     wa < 1e-12 and wo < 1e-12, f"rel {wa:.1e}, dec dev {wo:.1e}")
wb = 0.0
for _ in range(40):
    lD = int(rng.integers(44, 63))
    degb = int(rng.integers(2, 65 - lD)) if lD < 62 else 2
    lq0 = lD + degb
    arc = int(rng.choice([0, 3]))
    em_top = towers_layout(arc, DC)[0] - len(CTS_KS)
    Keff = K + float(rng.uniform(0, 0.3))
    Bc = 2.0 ** float(rng.uniform(20, 40))
    CS = C_SPARSE if rng.random() < 0.5 else C_DENSE
    sz, su, dy = 10 ** float(rng.uniform(-9, -5)), 10 ** float(rng.uniform(-6, -3)), 10 ** float(rng.uniform(-12, -6))
    sp = 2.0 ** lD / 2.0 ** lq0 * 1.01
    a1, b1 = evalmod_old_P(lD, lq0, arc, Keff, Bc, CS, em_top, True), evalmod_old_s(lD, lq0, arc, Keff, Bc, CS, em_top, True, sp)
    a2 = evalmod_mu_P(lD, lq0, arc, Keff, CS, em_top, sz, su, dy, True)
    b2 = evalmod_mu_s(lD, lq0, arc, Keff, CS, em_top, sz, su, dy, True, sp)
    wb = max(wb, abs(b1 / a1 - 1), abs(b2 / a2 - 1))
gate("S0b seeded EvalMod copies with seed 1.01 eps = PART P's evalmod_old / evalmod_mu (40 cases)", wb < 1e-12,
     f"rel {wb:.1e}")
stop_if_failed("PART S0")


# =============================================================================================
# declared K_eff per chain and the certification pass
# =============================================================================================
def beta_of(c, tau, lD, lq0):
    return circuit_tag(c, 2.0 ** lD * tau, lq0, lD) / 2.0 ** lD * BETA_SLACK


def keff_decl(c, tau, lD, lq0):
    return K + 2.0 ** lD / 2.0 ** lq0 * (1 + beta_of(c, tau, lD, lq0))


def s1_for(lD, lq0, ns, Keff):
    S = 2.0 ** lD / (2.0 ** lq0 * Keff * 2.0 ** 16 * 2)
    kss = N // (2 * ns) if ns < n_full else 1
    return (S / kss) ** (1.0 / len(CTS_KS))


jobs = {}
for j in range(17):
    _, lD, degb, arc, _ = OPT[j]
    lq0 = lD + degb
    ns = 2 ** j
    jobs[("lock", j)] = (lD, s1_for(lD, lq0, ns, keff_port(lD, lq0)), O["ntt_primes"](lD, N, 5))
    for c in CIRCS:
        for tau in TAUS:
            jobs[(c, tau, j)] = (lD, s1_for(lD, lq0, ns, keff_decl(c, tau, lD, lq0)), O["ntt_primes"](lD, N, 5))
print(f"\nPART S0  certification pass: {len(jobs)} encodings per diagonal (17 locked + 17 x 2 circuits x 2 tau)",
      flush=True)
cert = O["Certifier"](N, jobs)
O["certify"](N, CTS_KS, O["E"], O["Gg"], O["BRg"], cert, progress=False)
RS = {k_: cert.R[k_] for k_ in jobs}
wc = max(max(abs(x / y - 1) for x, y in zip(RS[("lock", j)], RQ[(OPT[j][1], OPT[j][2], 2 ** j)])) for j in range(17))
gate("S0c PART S job builder at the locked K_eff reproduces RQ at the 17 optima", wc < 1e-12,
     f"rel {wc:.1e} ({time.time() - T0:.0f} s)")
gate("S0e every rounding residue |r_k| <= 1/2 + 1e-9", cert.max_abs_r <= 0.5 + 1e-9, f"max {cert.max_abs_r:.12f}")
stop_if_failed("PART S0")

# =============================================================================================
print("\nPART S1  k = 1 at every optimum (dec = log2 T)", flush=True)
S1, wd = [], -math.inf
for j in range(17):
    d0, lD, degb, arc, _ = OPT[j]
    lq0, ns = lD + degb, 2 ** j
    Bf, Kp = fresh_B_in(lD, lq0), keff_port(lD, lq0)
    Rl = RQ[(lD, degb, ns)]
    dp = dec(eval_S(lD, degb, arc, ns, Rl, Bf, Kp, "port", "port")[0], lD)
    dl = dec(eval_S(lD, degb, arc, ns, Rl, Bf, Kp, "lemma", "port")[0], lD)
    dh = dec(eval_S(lD, degb, arc, ns, Rl, Bf, Kp, "lemma", "honest")[0], lD)
    Kd = keff_decl("id", 2.0 ** -10, lD, lq0)
    dc_ = dec(eval_S(lD, degb, arc, ns, RS[("id", 2.0 ** -10, j)], Bf, Kd, "lemma", "honest")[0], lD)
    wd = max(wd, dl - dp)
    S1.append((j, cellf(OPT[j]), d0, dp, dl, dh, dc_))
    print(f"    2^{j:<3} {cellf(OPT[j]):>9} | locked {d0:+.4f} | lemma routing {dl:+.4f} ({dl - d0:+.4f}) | "
          f"+ honest seed {dh:+.4f} ({dh - d0:+.4f}) | + declared K_eff, recert {dc_:+.4f} ({dc_ - d0:+.4f})")
gate("S0d k = 1: lemma routing <= port routing at the 17 optima (same K_eff, R_level, seed)", wd <= 1e-12,
     f"max (lemma - port) {wd:+.2e} bits")
stop_if_failed("PART S1")

# =============================================================================================
print("\nPART S2  projection level at the certified optimum of every sparse packing (Q's R_level)", flush=True)
S2 = []
for j in range(16):
    _, lD, degb, arc, _ = OPT[j]
    lq0, ns = lD + degb, 2 ** j
    Rl = RQ[(lD, degb, ns)]
    m0 = model_P(lD, lq0, arc, ns, BASE, Rl, True, ss=False)
    m1 = model_P(lD, lq0, arc, ns, BASE, Rl, True, ss=True)
    d0 = dec(eval_KQ(lD, degb, arc, ns, BASE, Rl, True, ss=False)[0], lD)
    d1 = dec(eval_KQ(lD, degb, arc, ns, BASE, Rl, True, ss=True)[0], lD)
    b0, b1 = bits(m0["B_cts"]), bits(m1["B_cts"])
    kss = N // (2 * ns)
    S2.append((j, cellf(OPT[j]), b0, b1, b1 - b0, bits(m1["E_ss"] / kss), d0, d1))
    print(f"    2^{j:<3} {cellf(OPT[j]):>9} | B_cts 2^{b0:.6f} -> 2^{b1:.6f} (diff {b1 - b0:.2e} bits) | "
          f"E_ss/k 2^{bits(m1['E_ss'] / kss):.2f} | dec {d0:+.6f} -> {d1:+.6f} ({d1 - d0:+.2e})")
maxdb = max(r[4] for r in S2)
maxdd = max(abs(r[7] - r[6]) for r in S2)
same = {nd: all(f"{r[2]:.{nd}f}" == f"{r[3]:.{nd}f}" for r in S2) for nd in (2, 3, 4, 5, 6)}
print(f"    max B_cts change {maxdb:.2e} bits, max dec change {maxdd:.2e} bits; B_cts equal when printed to "
      + ", ".join(f"{nd} dp: {same[nd]}" for nd in same))

# =============================================================================================
print("\nPART S3  chains at the certified optimum of every packing (lemma routing, honest seed, declared K_eff)",
      flush=True)


def run_chain(j, c, tau, routing):
    _, lD, degb, arc, _ = OPT[j]
    lq0, ns = lD + degb, 2 ** j
    D, q0 = 2.0 ** lD, 2.0 ** lq0
    beta = beta_of(c, tau, lD, lq0)
    Keff = keff_decl(c, tau, lD, lq0)
    Rl = RS[(c, tau, j)]
    B = fresh_B_in(lD, lq0)
    Ts, a_last, b_last, seedmax, stop, T_fail = [], None, None, 0.0, None, None
    for k in range(1, KMAX + 1):
        a = (D + B) / q0
        if B / D > beta:
            stop = "beta"
            break
        if a >= 0.5:
            stop = "a"
            break
        r = eval_S(lD, degb, arc, ns, Rl, B, Keff, routing, "honest")
        if r is None:
            stop = "a"
            break
        T = r[0] / D
        seedmax = max(seedmax, r[1]["seed"] / (1.01 * r[1]["eps"]))
        if not (T <= tau):
            stop, T_fail = "tau", T
            break
        Ts.append(T)
        a_last, b_last = a, B / D
        B = circuit_tag(c, D * T, lq0, lD)
    else:
        stop = f">={KMAX}"
    return dict(n=len(Ts), Ts=Ts, stop=stop, T_fail=T_fail, a_last=a_last, b_last=b_last, beta=beta,
                Keff=Keff, seedmax=seedmax)


CH = {}
for c in CIRCS:
    for tau in TAUS:
        for j in range(17):
            CH[(c, tau, j, "lemma")] = run_chain(j, c, tau, "lemma")
            CH[(c, tau, j, "port")] = run_chain(j, c, tau, "port")
        print(f"    {c} tau 2^{bits(tau):.0f} done ({time.time() - T0:.0f} s)", flush=True)


def fmt_chain(r, tau):
    Ts = r["Ts"]
    T1 = bits(Ts[0]) if Ts else (bits(r["T_fail"]) if r["T_fail"] else float("nan"))
    T2 = bits(Ts[1]) if len(Ts) > 1 else (bits(r["T_fail"]) if (len(Ts) == 1 and r["T_fail"]) else float("nan"))
    if len(Ts) > 1:
        rel = (Ts[1] - Ts[0]) / Ts[0]
    elif len(Ts) == 1 and r["T_fail"] and math.isfinite(r["T_fail"]):
        rel = (r["T_fail"] - Ts[0]) / Ts[0]
    else:
        rel = float("nan")
    last = bits(Ts[-1]) if Ts else float("nan")
    m_tau = bits(tau) - last if Ts else float("nan")
    m_a = bits(0.5) - bits(r["a_last"]) if r["a_last"] else float("nan")
    m_b = bits(r["beta"]) - bits(r["b_last"]) if r["b_last"] else float("nan")
    over = (bits(r["T_fail"]) - bits(tau)) if r["T_fail"] else float("nan")
    return (f"pass {r['n']:>4} | T1 2^{T1:+.3f} T2 2^{T2:+.3f} (T2-T1)/T1 {rel:.4f} | last 2^{last:+.3f} | "
            f"stop {r['stop']} (fail over tau by {over:.3f} bits) | margins at last: tau {m_tau:.3f} a {m_a:.2f} "
            f"beta {m_b:.3f} | seed/1.01eps max {r['seedmax']:.4f}")


for c in CIRCS:
    for tau in TAUS:
        print(f"\n    circuit {c}, tau 2^{bits(tau):.0f}")
        for j in range(17):
            r, rp = CH[(c, tau, j, "lemma")], CH[(c, tau, j, "port")]
            print(f"    2^{j:<3} {cellf(OPT[j]):>9} | {fmt_chain(r, tau)} | port routing pass {rp['n']}")
print(f"\n    total runtime {time.time() - T0:.0f} s")

# =============================================================================================
print("\n" + "=" * 100)
print("SUMMARY BEGIN")
print(f"S0 a {wa:.1e}/{wo:.1e} b {wb:.1e} c {wc:.1e} d {wd:+.2e} e {cert.max_abs_r:.12f}")
for j, cell, d0, dp, dl, dh, dc_ in S1:
    print(f"S1 2^{j} {cell}: locked {d0:+.4f} lemma {dl:+.4f} +seed {dh:+.4f} +Keff {dc_:+.4f}")
for j, cell, b0, b1, db, ess, d0, d1 in S2:
    print(f"S2 2^{j} {cell}: B_cts 2^{b0:.6f}->2^{b1:.6f} ({db:.2e}) E_ss/k 2^{ess:.2f} dec {d0:+.6f}->{d1:+.6f}")
print(f"S2 max B_cts change {maxdb:.2e} bits, max dec change {maxdd:.2e}; equal at "
      + " ".join(f"{nd}dp:{same[nd]}" for nd in same))
for c in CIRCS:
    for tau in TAUS:
        for j in range(17):
            r, rp = CH[(c, tau, j, "lemma")], CH[(c, tau, j, "port")]
            print(f"S3 {c} 2^{bits(tau):.0f} 2^{j}: {fmt_chain(r, tau)} | port {rp['n']}")
print(f"S runtime {time.time() - T0:.0f} s")
print("GATE FAILURES:", FAILS if FAILS else "none")
print("SUMMARY END")
print("=" * 100)
