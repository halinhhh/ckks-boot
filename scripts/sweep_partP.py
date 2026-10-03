#!/usr/bin/env python3
"""
sweep_partP.py  --  PART P (1 Oct 2026): the forced decisions of PART M, re-measured on the
certified configuration of PART O (cert+StC), and the log Q of every new optimum.

HOW IT RUNS
  It runs sweep_partO.py (which runs the sweep, PART 0..L, and certifies R_level) with its output
  captured, stops if any gate of either failed, and reuses their functions and certified R values.
  Neither file is modified. sweep_partO.py and the sweep must be in the folder of this script.
      python3 /path/to/sweep_partP.py
  Runtime: ~8 min on a 2-core machine (PART O 5 min, second certification pass 2.4 min, rest 0.8 min);
  on a laptop PART O took 3 min, so expect ~5 min. Overflowing tags (INFO) come from alternatives
  (c), (iv-e), (iv-f) at small Delta, where the residue terms explode; such cells are never optimal.

THE NEW BASE (= PART O cert+StC)
  front scale S spread as S^{1/5} over the five CtS levels; CtS diagonal residues certified
  (canonical norm of the actual encodings, R_level per cell); scalar residues of c_j, A^2, a;
  StC residues under the ||.||_1 check; KS_ds one level before the end; KS_sd after the back gate;
  one declared K_eff; per-stream real-part extraction.

ALTERNATIVES (each changes one decision; Delta(bits) = dec(alt) - dec(base) > 0 = worth that much)
  (ii)    KS_ds after the last client rescale (towers = 1), as in PART M.
  (iii)   KS_sd before EvalMod, as in PART M (dense-key c_rs in EvalMod, switch error on the input).
  (iv-e)  S folded at the END of CtS as one scalar constant at scale p (residue 1/2 on the unscaled
          message), the CtS diagonals at scale p without S^{1/5}: their R_level is CERTIFIED here
          for every cell (second certification pass, s1 = 1).
  (iv-f)  S folded at the FRONT of CtS as one scalar constant at scale p (residue 1/2 on nu_0), then
          the levels with the same s1 = 1 diagonals (certified R_level).
  (vi)    the fitted K and the normalisation K + a differ, as in PART M.
  (vii)   no per-stream real-part extraction, as in PART M.
  (c)     the paper's current check ||a~ - S f||_1 <= R for the CtS diagonals (R = N/2 per actual
          diagonal, counts 16/15/15/15/15) instead of the certified canonical norm: what option (A)
          is worth.
  (v)     the split (3,3) is still not computed (no Paterson-Stockmeyer recursion in the port).

GATES
  P0a  residues off, every option: eval_KP reproduces the recorded PART M run (base decs and M1 Deltas of
       (iii), (iv-f), (iv-s), (vi), (vii); (ii) inadmissible) at the 5 headline cells, 0.0015.
  P0b  residues on, spread and end folds: eval_KP = PART O's eval_KO at 5 headline + 40 random cells
       (1e-12), and = the recorded PART O2 values (cert+StC, end+cert) at the 5 headline cells (0.0015).
  P0c  the second certification pass reproduces PART O's s1 = 1 values at the 5 headline cells
       (1e-12 relative) and every residue |r_k| <= 1/2 + 1e-9.
  P0d  the re-optimised base optima reproduce PART O3 of the same run exactly (17 packings) and
       the recorded PART O3 values (1 slot, 2^3, 2^8, full; printed to 2 decimals, so 0.0051).

USAGE   python3 /path/to/sweep_partP.py
        The summary block is printed between SUMMARY BEGIN and SUMMARY END.
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
PART_O = os.path.join(HERE, "sweep_partO.py")
if not os.path.isfile(PART_O):
    print(f"Cannot find {PART_O}. Put sweep_partP.py next to sweep_partO.py and the sweep.")
    sys.exit(1)

print("=" * 100)
print("sweep_partP.py   PART P: forced decisions on the certified configuration; log Q of the optima")
print("=" * 100)
print("  running sweep_partO.py silently (sweep + certification, ~3 min) ...", flush=True)
_buf = io.StringIO()
try:
    with contextlib.redirect_stdout(_buf):
        O = runpy.run_path(PART_O, run_name="partO")
except SystemExit:
    print(_buf.getvalue())
    print("\nsweep_partO.py stopped at one of its gates (output above). Fix that first.")
    sys.exit(1)
if O["FAILS"]:
    print(_buf.getvalue())
    print("\nsweep_partO.py reports gate failures:", O["FAILS"])
    sys.exit(1)
print(f"  sweep_partO.py: all gates passed ({time.time() - T0:.0f} s)", flush=True)

N, n_full, K, ELL, DB, E0, TWOPI, DC = (O[k] for k in ("N", "n_full", "K", "ELL", "DB", "E0", "TWOPI", "DC"))
CTS_KS, CTS_ROT, R_STC = O["CTS_KS"], O["CTS_ROT"], O["R_STC"]
C_DENSE, C_SPARSE = O["C_DENSE"], O["C_SPARSE"]
KEFF_FIX, SUBSUM = O["KEFF_FIX"], O["SUBSUM"]
bks, depth, cheb_coeffs, q0lam = O["bks"], O["depth"], O["cheb_coeffs"], O["q0lam"]
parseval_factor, c_of, part_of = O["parseval_factor"], O["c_of"], O["part_of"]
amp_inj, amp_after_first, towers_layout = O["amp_inj"], O["amp_after_first"], O["towers_layout"]
sens_bounds, bits, dec = O["sens_bounds"], O["bits"], O["dec"]
eval_KO, stc_residue = O["eval_KO"], O["stc_residue"]
CELLS, HEAD_O, COUNTS, RC = O["CELLS"], O["HEAD"], O["COUNTS"], O["RC"]
R_L1 = O["R_L1"]
H = O["H"]
FAILS = []


def gate(name, ok, detail=""):
    print(f"    [{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def stop_if_failed(part):
    if FAILS:
        print(f"\n  {part} GATE FAILED:", FAILS, " -- see the output above.")
        sys.exit(1)


# =============================================================================================
# model with the decisions as options AND the residues (PART M's model_M + PART O's model_O)
# =============================================================================================
BASE = dict(fold="spread", ksd_before=False, kds_end=False, ext=True, mismatch=False, check="can")
OLD_BASE = dict(BASE, fold="end")            # PART M's base (with residues off)
ALTS = [("(ii)   KS_ds at the end", dict(BASE, kds_end=True)),
        ("(iii)  KS_sd before EvalMod", dict(BASE, ksd_before=True)),
        ("(iv-e) S folded at CtS end", dict(BASE, fold="end")),
        ("(iv-f) S folded at CtS front", dict(BASE, fold="front")),
        ("(vi)   K_eff mismatch", dict(BASE, mismatch=True)),
        ("(vii)  no per-stream extraction", dict(BASE, ext=False)),
        ("(c)    ||.||_1 check for CtS", dict(BASE, check="l1"))]

R_END = {}          # (lD, degb) -> certified R_level of the s1 = 1 diagonals (filled below)


def rlev_for(lD, degb, o, res):
    if not res:
        return None
    if o["check"] == "l1":
        return [c * R_L1 for c in COUNTS]
    if o["fold"] == "spread":
        return RC[(lD, degb, "spread", 0)]
    return R_END[(lD, degb)]


def model_P(lD, lq0, arc, ns, o, Rlev, scal, ss=None, dc=DC):
    ss = SUBSUM if ss is None else ss
    ss_ns = ns if (ss and ns < n_full) else None
    Rlev = [0.0] * len(CTS_KS) if Rlev is None else Rlev
    D, q0 = 2.0 ** lD, 2.0 ** lq0
    p = D
    eps = D / q0
    Keff = K + eps
    top, dem = towers_layout(arc, dc)
    B = N * E0 + N / 2.0
    for i in range(dc):
        B = (2 + B / D) * B + C_DENSE + bks(dc + 1 - i, lq0, lD) / D
    B_term = bks(1, lq0, lD) if o["kds_end"] else bks(2, lq0, lD) / D       # verbatim from PART M
    B += B_term
    B_in = B
    if KEFF_FIX:
        Keff = K + (D + B_in) / q0
    E_ss, extra_lvl = 0.0, 0
    if ss_ns is not None:
        kss = N // (2 * ss_ns)
        E_ss = (kss - 1) * bks(top + 1, lq0, lD) / D + C_SPARSE
        B += E_ss / kss
        extra_lvl = 1
    nu = N * q0 * (1 + H) / 2.0 + D + B_in
    add_cts = 0.0
    S = D / (q0 * Keff * 2.0 ** 16 * 2)
    if o["fold"] == "front":
        res_s = (0.5 * (nu + B) / p) if scal else 0.0
        B = S * B + C_SPARSE + res_s
        add_cts = res_s
        nu = S * nu
    s1 = S ** (1.0 / len(CTS_KS)) if o["fold"] == "spread" else 1.0
    for i, k in enumerate(CTS_KS):
        res = Rlev[i] * (nu + B) / p
        B = 2.0 ** k * s1 * B + C_SPARSE + CTS_ROT[i] * bks(top - i, lq0, lD) / D + res
        add_cts = 2.0 ** k * s1 * add_cts + res
        nu = 2.0 ** k * s1 * nu
    if o["fold"] == "end":
        res_s = (0.5 * (nu + B) / p) if scal else 0.0
        B = S * B + C_SPARSE + res_s
        add_cts = S * add_cts + res_s
    elif o["fold"] == "spread":
        B = B + C_SPARSE
    elif o["fold"] != "front":
        raise ValueError(o["fold"])
    em_top = top - len(CTS_KS)
    B_conj = bks(em_top, lq0, lD) / D
    B_sdb = bks(em_top, lq0, lD) / D if o["ksd_before"] else 0.0
    B_cts = 2 * B + B_conj + B_sdb
    CS = C_DENSE if o["ksd_before"] else C_SPARSE
    eD = evalmod_old(lD, lq0, arc, Keff, B_cts, CS, em_top, scal)
    stc_top = 1 + dc + R_STC
    B_sd = 0.0 if o["ksd_before"] else bks(stc_top, lq0, lD) / D
    return dict(B_cts=B_cts, eD=eD, B_sd=B_sd, logQ=lq0 + (top - 1 + extra_lvl) * lD, top=top,
                Keff=Keff, B_in=B_in, B_cj=bks(stc_top + 1, lq0, lD) / D, E_ss=E_ss, B_conj=B_conj,
                B_sdb=B_sdb, CS=CS, em_top=em_top, add_cts=2 * add_cts)


def scal_res(nu_over_D, B, D):
    return 0.5 * (nu_over_D * D + B) / D


def evalmod_old(lD, lq0, arc, Keff, B_cts, CS, em_top, scal):
    D, q0 = 2.0 ** lD, 2.0 ** lq0
    eps = D / q0
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
        nu_y = D * eps * 1.01
        B2 = (2 * nu_y * B + B * B) / D + CS + bks(tw, lq0, lD) / D
        nu_y2 = D * (eps * 1.01) ** 2
        a = TWOPI ** 2 / 6.0
        r_a = scal_res(eps * 1.01 * eps * 1.01, B2, D) if scal else 0.0
        B3 = a * (nu_y2 * B + nu_y * B2 + B * B2) / D + CS + bks(tw - 1, lq0, lD) / D + r_a
        B = B + 2 * CS + B3
    return B


def evalmod_mu(lD, lq0, arc, Keff, CS, em_top, sz, su, dy, scal):
    D, q0 = 2.0 ** lD, 2.0 ** lq0
    eps = D / q0
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
        yb = eps * 1.01 + dy
        nu_y = D * yb
        B2 = (2 * nu_y * B + B * B) / D + CS + KS_(tw)
        nu_y2 = D * yb ** 2
        a_ = TWOPI ** 2 / 6.0
        r_a = scal_res(yb * yb, B2, D) if scal else 0.0
        B3 = a_ * (nu_y2 * B + nu_y * B2 + B * B2) / D + CS + KS_(tw - 1) + r_a
        B = B + 2 * CS + B3
    return B


NO_SENS = []
NONFINITE = []
np.seterr(over="ignore")


def eval_KP(lD, degb, arc, ns, o, res=True, ss=None):
    """tau with the decisions as options; res = False switches every residue off (PART M)."""
    lq0 = lD + degb
    D, q0 = 2.0 ** lD, 2.0 ** lq0
    mo = model_P(lD, lq0, arc, ns, o, rlev_for(lD, degb, o, res), res, ss)
    eps_in = (D + mo["B_in"]) / q0
    if eps_in >= 0.5:
        return None
    A_sup = c_of(ns) * ns
    A_n = A_sup if o["ext"] else 2 * ns
    back = 2.0 ** degb
    P = part_of(ns)
    QL = q0lam(lq0, mo["Keff"])
    t = dict(approx=A_n * QL + q0 * parseval_factor(eps_in, arc),
             stc=C_DENSE * amp_inj(P), bsd=mo["B_sd"] * amp_after_first(P),
             conj=ns * back * mo["B_cj"] if o["ext"] else 0.0,
             post=(2 * ns - A_sup) * back * mo["CS"] if o["ext"] else 0.0,
             mism=A_n * q0 * (K + eps_in) * (eps_in / K) if o["mismatch"] else 0.0)
    B_im = mo["B_conj"] + mo["B_sdb"]
    delta, eta = mo["B_cts"] / D, B_im / D
    try:
        if math.exp(math.acosh(1 + delta)) * 1.01 >= 40.0:
            raise ValueError("rho grid")
        sb = sens_bounds(mo["Keff"], arc, delta, eta)
        eD_int = evalmod_mu(lD, lq0, arc, mo["Keff"], mo["CS"], mo["em_top"], sb["sz"], sb["su"],
                            delta * sb["L_R"] + eta * sb["L_I"], res)
        eD_K = eD_int + sb["L_R"] * mo["B_cts"] + sb["L_I"] * B_im
    except (ValueError, OverflowError):
        eD_int = eD_K = math.inf
        NO_SENS.append((lD, degb, arc, ns))
    t["noise"] = A_n * back * min(mo["eD"], eD_K)
    if res:
        t["stc_res"] = stc_residue(P, D, mo["B_in"], sum(t.values()))
    tot = sum(t.values())
    if not math.isfinite(tot):
        NONFINITE.append((lD, degb, arc, ns))
    return tot, dict(t=t, mo=mo)


# =============================================================================================
# second certification pass: s1 = 1 diagonals at every cell (folds iv-e and iv-f)
# =============================================================================================
print("\nPART P0  certification of the s1 = 1 diagonals at every cell (folds iv-e / iv-f)", flush=True)
jobs = {(lD, degb, "end", 0): (lD, 1.0, O["ntt_primes"](lD, N, 5)) for (lD, degb) in CELLS}
cert2 = O["Certifier"](N, jobs)
O["certify"](N, CTS_KS, O["E"], O["Gg"], O["BRg"], cert2, progress=False)
for (lD, degb) in CELLS:
    R_END[(lD, degb)] = cert2.R[(lD, degb, "end", 0)]
w = 0.0
for lab, lD, degb, arc, ns in HEAD_O:
    a_, b_ = RC[(lD, degb, "end", 0)], R_END[(lD, degb)]
    w = max(w, max(abs(x / y - 1) for x, y in zip(a_, b_)))
gate("P0c second pass = PART O's s1 = 1 values at the 5 headline cells; |r_k| <= 1/2 + 1e-9",
     w < 1e-12 and cert2.max_abs_r <= 0.5 + 1e-9, f"rel {w:.1e}, max|r| {cert2.max_abs_r:.12f} "
     f"({time.time() - T0:.0f} s)")
stop_if_failed("PART P0")

print("\nPART P0  GATES (model)")
PARTM_BASE = {"ref": 6.557, "2^0": -14.920, "2^3": -12.631, "2^8": -9.272, "2^16": -4.104}
PARTM_M1 = {"(iii)": (9.961, 9.586, 8.173, 9.808, 9.011), "(iv-f)": (9.502, 9.130, 7.697, 9.331, 8.170),
            "(iv-s)": (0.017, 0.014, 0.005, 0.016, 0.008), "(vi)": (9.793, 15.421, 15.983, 17.622, 20.458),
            "(vii)": (0.651, 0.406, 0.226, 0.615, 0.410)}
M_OPTS = {"(iii)": dict(OLD_BASE, ksd_before=True), "(iv-f)": dict(OLD_BASE, fold="front"),
          "(iv-s)": dict(OLD_BASE, fold="spread"), "(vi)": dict(OLD_BASE, mismatch=True),
          "(vii)": dict(OLD_BASE, ext=False)}
wa, inadm_ok = 0.0, True
for i, (lab, lD, degb, arc, ns) in enumerate(HEAD_O):
    b0 = dec(eval_KP(lD, degb, arc, ns, OLD_BASE, res=False)[0], lD)
    wa = max(wa, abs(b0 - PARTM_BASE[lab]))
    for nm, o in M_OPTS.items():
        wa = max(wa, abs(dec(eval_KP(lD, degb, arc, ns, o, res=False)[0], lD) - b0 - PARTM_M1[nm][i]))
    inadm_ok &= eval_KP(lD, degb, arc, ns, dict(OLD_BASE, kds_end=True), res=False) is None
gate("P0a residues off: base decs and M1 Deltas of PART M (iii, iv-f, iv-s, vi, vii; ii inadm.), 5 cells",
     wa < 0.0015 and inadm_ok, f"max dev {wa:.4f}, (ii) inadmissible at all 5: {inadm_ok}")
rng = np.random.default_rng(20261002)
rand = [(f"rnd{i}",) + CELLS[j] + (int(rng.choice([0, 3])), int(2 ** rng.integers(0, 17)))
        for i, j in enumerate(rng.choice(len(CELLS), 40, replace=False))]
wb = 0.0
for lab, lD, degb, arc, ns in HEAD_O + rand:
    a1 = eval_KO(lD, degb, arc, ns, RC[(lD, degb, "spread", 0)], True, "spread", True)[0]
    b1 = eval_KP(lD, degb, arc, ns, BASE)[0]
    wb = max(wb, abs(a1 / b1 - 1))
for lab, lD, degb, arc, ns in HEAD_O:
    a1 = eval_KO(lD, degb, arc, ns, R_END[(lD, degb)], True, "end", True)[0]
    b1 = eval_KP(lD, degb, arc, ns, dict(BASE, fold="end"))[0]
    wb = max(wb, abs(a1 / b1 - 1))
PARTO_O2 = {"ref": (8.013, 20.371), "2^0": (-13.805, -2.477), "2^3": (-12.173, -2.624),
            "2^8": (-8.159, 2.373), "2^16": (-3.622, 4.381)}
wc = 0.0
for lab, lD, degb, arc, ns in HEAD_O:
    wc = max(wc, abs(dec(eval_KP(lD, degb, arc, ns, BASE)[0], lD) - PARTO_O2[lab][0]),
             abs(dec(eval_KP(lD, degb, arc, ns, dict(BASE, fold="end"))[0], lD) - PARTO_O2[lab][1]))
gate("P0b residues on: eval_KP = eval_KO (spread 45 cells, end 5 cells); = the recorded PART O2 (cert+StC, end+cert)",
     wb < 1e-12 and wc < 0.0015, f"rel {wb:.1e}, max dev {wc:.4f}")
stop_if_failed("PART P0")


def best_over(ns, o):
    best = None
    for arc in (0, 3):
        for lD, degb in CELLS:
            r = eval_KP(lD, degb, arc, ns, o)
            if r is None:
                continue
            d_ = dec(r[0], lD)
            if best is None or d_ < best[0]:
                best = (d_, lD, degb, arc, r[1]["mo"]["logQ"])
    return best


cellf = lambda b: f"{b[1]}/{b[2]}/{'on' if b[3] else 'off'}"
OPT = {k: best_over(2 ** k, BASE) for k in range(0, 17)}
PARTO_O3 = {0: (-13.80, "58/6/on"), 3: (-12.17, "59/5/on"), 8: (-8.61, "60/4/on"), 16: (-3.62, "61/3/on")}
wd = max(abs(OPT[k][0] - v[0]) for k, v in PARTO_O3.items())        # recorded values have 2 decimals
okc = all(cellf(OPT[k]) == v[1] for k, v in PARTO_O3.items())
O3_own = {int(r[0][2:]): r[2] for r in O["O3"]}                       # PART O's own run, full precision
we = max(abs(OPT[k][0] - O3_own[k][0]) for k in OPT)
oke = all(OPT[k][1:4] == O3_own[k][1:4] for k in OPT)
gate("P0d re-optimised base optima = PART O3 of this run (17 packings, exact) and the recorded O3 (2 decimals)",
     wd <= 0.0051 and okc and we < 1e-12 and oke,
     f"vs recorded max dev {wd:.4f} (rounding <= 0.005), cells equal {okc}; vs PART O {we:.1e}, cells equal {oke}")
stop_if_failed("PART P0")

# =============================================================================================
# P1 fixed cells, P2 re-optimised
# =============================================================================================
HEAD = [("ref", 53, 7, 0, n_full)] + [(lab, OPT[k][1], OPT[k][2], OPT[k][3], 2 ** k)
                                      for lab, k in (("1 slot", 0), ("2^3", 3), ("2^8", 8), ("full", 16))]
print("\nPART P1  Delta(bits) = dec(alt) - dec(base) at the new base optimum cells ('inadm.' = a >= 1/2)")
base_dec = {h[0]: dec(eval_KP(h[1], h[2], h[3], h[4], BASE)[0], h[1]) for h in HEAD}
print(f"    {'base (cert+StC)':<32} | " + " | ".join(f"{h[0]:>6} {h[1]}/{h[2]}/{'on' if h[3] else 'off':<3} "
                                                      f"{base_dec[h[0]]:+7.3f}" for h in HEAD))
FIX, below = {}, []
for name, o in ALTS:
    outs = []
    for h in HEAD:
        r = eval_KP(h[1], h[2], h[3], h[4], o)
        if r is None:
            FIX[(name, h[0])] = None
            outs.append(f"{'inadm.':>22}")
            continue
        d_ = dec(r[0], h[1]) - base_dec[h[0]]
        FIX[(name, h[0])] = d_
        if d_ < -1e-9:
            below.append((name.strip(), h[0], round(d_, 4)))
        outs.append(f"{d_:>+22.3f}")
    print(f"    {name:<32} | " + " | ".join(outs))
print(f"    [INFO] alternatives below the base at a headline cell: {below if below else 'none'}")

print("\nPART P2  re-optimised (logD 36..62, degb 2..14, arcsine on/off): Delta and the alternative's best cell",
      flush=True)
PACK = [("1 slot", 0), ("2^3", 3), ("2^8", 8), ("full", 16)]
print(f"    {'base (cert+StC)':<32} | " + " | ".join(f"{lab:>6} {OPT[k][0]:+7.2f} {cellf(OPT[k]):<9}"
                                                      for lab, k in PACK))
REOPT = {}
for name, o in ALTS:
    outs = []
    for lab, k in PACK:
        b = best_over(2 ** k, o)
        REOPT[(name, lab)] = None if b is None else (b[0] - OPT[k][0], b)
        outs.append(f"{lab:>6} {'none admissible':>17}" if b is None else
                    f"{lab:>6} {b[0] - OPT[k][0]:+7.2f} {cellf(b):<9}")
    print(f"    {name:<32} | " + " | ".join(outs))

# =============================================================================================
# P3 SubSum at the new optima; P4 log Q of the optima
# =============================================================================================
print("\nPART P3  SubSum (projection level) at the new optimum of every sparse packing: off vs on")
P3 = []
for k in range(0, 16):
    _, lD, degb, arc, _ = OPT[k]
    lq0 = lD + degb
    m0 = model_P(lD, lq0, arc, 2 ** k, BASE, RC[(lD, degb, "spread", 0)], True, ss=False)
    m1 = model_P(lD, lq0, arc, 2 ** k, BASE, RC[(lD, degb, "spread", 0)], True, ss=True)
    d0 = dec(eval_KP(lD, degb, arc, 2 ** k, BASE, ss=False)[0], lD)
    d1 = dec(eval_KP(lD, degb, arc, 2 ** k, BASE, ss=True)[0], lD)
    b0, b1 = bits(m0["B_cts"]), bits(m1["B_cts"])
    P3.append((k, cellf(OPT[k]), b0, b1, f"{b0:.3f}" == f"{b1:.3f}", d0, d1))
    print(f"    2^{k:<3} {cellf(OPT[k]):>9} | B_cts 2^{b0:.4f} -> 2^{b1:.4f} (3 dec same {P3[-1][4]}) | "
          f"dec {d0:+.4f} -> {d1:+.4f} ({d1 - d0:+.1e})")

print("\nPART P4  log Q of the new optimum of every packing (tab:security records 1332 1385 1552 1657 1684)")
REC = (1332, 1385, 1552, 1657, 1684)
lq = sorted({OPT[k][4] for k in OPT})
need = [q for q in lq if q > max(REC)]
for q in lq:
    ks = [f"2^{k}" for k in OPT if OPT[k][4] == q]
    cover = min([r for r in REC if r >= q], default=None)
    print(f"    log Q {q}: {', '.join(ks):<40} " + (f"covered by the row at {cover}" if cover else "NOT ESTIMATED"))
print(f"    rows still to estimate: {need if need else 'none'}")
print(f"\n    [INFO] Lemma sens not evaluated (E*_eval = E_eval): {len(NO_SENS)}; overflowing tags: {len(NONFINITE)}")
print(f"    total runtime {time.time() - T0:.0f} s")

# =============================================================================================
print("\n" + "=" * 100)
print("SUMMARY BEGIN")
for h in HEAD:
    print(f"P base {h[0]} {h[1]}/{h[2]}/{'on' if h[3] else 'off'}: dec {base_dec[h[0]]:+.3f}")
for name, o in ALTS:
    print(f"P1 {name.strip()}: " + " | ".join(
        f"{h[0]} " + ("inadm." if FIX[(name, h[0])] is None else f"{FIX[(name, h[0])]:+.3f}") for h in HEAD))
for name, o in ALTS:
    print(f"P2 {name.strip()}: " + " | ".join(
        f"{lab} " + ("none admissible" if REOPT[(name, lab)] is None else
                     f"{REOPT[(name, lab)][0]:+.2f} at {cellf(REOPT[(name, lab)][1])}") for lab, _ in PACK))
for k, cell, b0, b1, same, d0, d1 in P3:
    print(f"P3 2^{k} {cell}: B_cts 2^{b0:.4f}->2^{b1:.4f} same3 {same} | dec {d0:+.4f}->{d1:+.4f}")
for k in range(0, 17):
    print(f"P4 2^{k}: {OPT[k][0]:+.2f} {cellf(OPT[k])} logQ {OPT[k][4]}")
print(f"P4 rows still to estimate: {need if need else 'none'}")
print(f"P no-sens {len(NO_SENS)} | overflow {len(NONFINITE)} | runtime {time.time() - T0:.0f} s")
print("GATE FAILURES:", FAILS if FAILS else "none")
print("SUMMARY END")
print("=" * 100)
