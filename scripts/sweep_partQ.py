#!/usr/bin/env python3
"""
sweep_partQ.py  --  PART Q (1 Oct 2026): certify the CtS residues of the SPARSE-packing instances.

WHY
  PART O certified R_level for the diagonals encoded at p_i S^{1/5} phi^{-1}(diag). For n_s < n
  slots the port keeps the full-packing CtS but its normalisation carries 1/k, k = N/(2 n_s)
  (sweep header, SUBSUM; App. subsum: "the sparse CtS carries the factor 1/k"). The encoded
  constants of a sparse instance are therefore p_i (S/k)^{1/5} phi^{-1}(diag): different numbers,
  different rounding residues. PARTS O and P used the full-packing R_level for every packing, which
  is a proxy, not a certificate, for n_s < 2^16. PART Q certifies R_level at s1 = (S/k)^{1/5} for
  every sparse packing and re-runs the optimum and the decision table.

  Everything else in the tag is kept as in the port: the model still scales the CtS by S (not S/k),
  which overstates nu and every injected term after the first level by up to k^{i/5}; the tag stays
  an upper bound, and only R_level changes.

PRUNING (exact, no MODEL)
  Every residue term is nonnegative and every tag is monotone in them, so the residue-free dec of a
  cell (R = 0, no scalar or StC residues) is a LOWER bound on its certified dec. For each packing
  and option the script (1) certifies the cell where the full-packing proxy was optimal, (2) takes
  its certified dec as an upper bound U, (3) certifies every cell whose lower bound is below U, and
  (4) takes the best certified cell. Gate Q0b checks that no uncertified cell has a lower bound
  below the reported optimum, so the optimum is certified over the whole domain.

OPTIONS RE-MEASURED  base (cert+StC), (iii), (vi), (vii): they use the spread-fold R_level.
  (ii) is inadmissible everywhere; (iv-e), (iv-f) and (c) do not depend on n_s through R (s1 = 1 or
  the ||.||_1 bound), so their PART P optima stand and only their Deltas move with the base.

GATES
  Q0a  certification at k = 1 (full packing) reproduces PART O's R_level at 5 cells exactly.
  Q0b  for every (packing, option): no uncertified cell has a lower bound below the optimum.
  Q0c  every rounding residue |r_k| <= 1/2 + 1e-9.
  Q0d  eval_KQ with PART P's R_level = PART P's eval_KP at the 5 headline + 40 random cells (1e-12).

USAGE   python3 /path/to/sweep_partQ.py   (next to sweep_partP.py, sweep_partO.py and the sweep)
        Runs PART P silently first (~5 min on a laptop), then ~2-4 min (2-core machine: 8 + 4 min).
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
PART_P = os.path.join(HERE, "sweep_partP.py")
if not os.path.isfile(PART_P):
    print(f"Cannot find {PART_P}. Put sweep_partQ.py next to sweep_partP.py, sweep_partO.py and the sweep.")
    sys.exit(1)

print("=" * 100)
print("sweep_partQ.py   PART Q: certified CtS residues of the sparse-packing instances (s1 = (S/k)^(1/5))")
print("=" * 100)
print("  running sweep_partP.py silently (it runs PART O and the sweep; ~5 min) ...", flush=True)
_buf = io.StringIO()
try:
    with contextlib.redirect_stdout(_buf):
        P = runpy.run_path(PART_P, run_name="partP")
except SystemExit:
    print(_buf.getvalue())
    print("\nsweep_partP.py stopped at one of its gates (output above). Fix that first.")
    sys.exit(1)
if P["FAILS"]:
    print(_buf.getvalue())
    print("\nsweep_partP.py reports gate failures:", P["FAILS"])
    sys.exit(1)
print(f"  sweep_partP.py: all gates passed ({time.time() - T0:.0f} s)", flush=True)

O = P["O"]
N, n_full, K, CTS_KS = O["N"], O["n_full"], O["K"], O["CTS_KS"]
C_DENSE, C_SPARSE = O["C_DENSE"], O["C_SPARSE"]
q0lam, parseval_factor, c_of, part_of = O["q0lam"], O["parseval_factor"], O["c_of"], O["part_of"]
amp_inj, amp_after_first, sens_bounds = O["amp_inj"], O["amp_after_first"], O["sens_bounds"]
bits, dec = O["bits"], O["dec"]
CELLS, RC, COUNTS, R_L1 = O["CELLS"], O["RC"], O["COUNTS"], O["R_L1"]
model_O = O["model_O"]
model_P, evalmod_mu, stc_residue = P["model_P"], P["evalmod_mu"], P["stc_residue"]
BASE, ALTS, R_END, OPT_P, REOPT_P = P["BASE"], P["ALTS"], P["R_END"], P["OPT"], P["REOPT"]
eval_KP, rlev_for_P = P["eval_KP"], P["rlev_for"]
FAILS = []


def gate(name, ok, detail=""):
    print(f"    [{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def stop_if_failed(part):
    if FAILS:
        print(f"\n  {part} GATE FAILED:", FAILS, " -- see the output above.")
        sys.exit(1)


kss_of = lambda ns: N // (2 * ns) if ns < n_full else 1
RQ = {}                       # (lD, degb, ns) -> certified R_level at s1 = (S/k)^(1/5)
for (lD, degb) in CELLS:      # full packing: PART O's values
    RQ[(lD, degb, n_full)] = RC[(lD, degb, "spread", 0)]


def s1_of(lD, degb, ns):
    S = model_O(lD, lD + degb, 0, n_full, None, True, "spread")["S"]
    return (S / kss_of(ns)) ** (1.0 / len(CTS_KS))


MAXR = [0.0]


def certify_batch(keys):
    """certify R_level for a list of (lD, degb, ns) keys in ONE pass over the diagonals."""
    keys = [k for k in dict.fromkeys(keys) if k not in RQ]
    if not keys:
        return 0
    jobs = {k: (k[0], s1_of(*k), O["ntt_primes"](k[0], N, 5)) for k in keys}
    cert = O["Certifier"](N, jobs)
    O["certify"](N, CTS_KS, O["E"], O["Gg"], O["BRg"], cert, progress=False)
    for k in keys:
        RQ[k] = cert.R[k]
    MAXR[0] = max(MAXR[0], cert.max_abs_r)
    return len(keys)


def rlev_q(lD, degb, ns, o, res):
    if not res:
        return None
    if o["check"] == "l1":
        return [c * R_L1 for c in COUNTS]
    if o["fold"] == "spread":
        return RQ[(lD, degb, ns)]
    return R_END[(lD, degb)]


NO_SENS, NONFINITE = [], []


def eval_KQ(lD, degb, arc, ns, o, Rlev, res=True, ss=None):
    """PART P's eval_KP with R_level passed explicitly (gate Q0d)."""
    lq0 = lD + degb
    D, q0 = 2.0 ** lD, 2.0 ** lq0
    mo = model_P(lD, lq0, arc, ns, o, Rlev, res, ss)
    eps_in = (D + mo["B_in"]) / q0
    if eps_in >= 0.5:
        return None
    A_sup = c_of(ns) * ns
    A_n = A_sup if o["ext"] else 2 * ns
    back = 2.0 ** degb
    Pp = part_of(ns)
    QL = q0lam(lq0, mo["Keff"])
    t = dict(approx=A_n * QL + q0 * parseval_factor(eps_in, arc),
             stc=C_DENSE * amp_inj(Pp), bsd=mo["B_sd"] * amp_after_first(Pp),
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
        t["stc_res"] = stc_residue(Pp, D, mo["B_in"], sum(t.values()))
    tot = sum(t.values())
    if not math.isfinite(tot):
        NONFINITE.append((lD, degb, arc, ns))
    return tot, dict(t=t, mo=mo)


def dec_q(lD, degb, arc, ns, o, res=True):
    r = eval_KQ(lD, degb, arc, ns, o, rlev_q(lD, degb, ns, o, res), res)
    if r is None:
        return None
    d_ = dec(r[0], lD)
    return d_ if math.isfinite(d_) else math.inf


# =============================================================================================
print("\nPART Q0  GATES", flush=True)
w = 0.0
rng = np.random.default_rng(20261003)
for lab, lD, degb, arc, ns in P["HEAD"] + [("rnd",) + CELLS[j] + (int(rng.choice([0, 3])), int(2 ** rng.integers(0, 17)))
                                          for j in rng.choice(len(CELLS), 40, replace=False)]:
    a1 = eval_KP(lD, degb, arc, ns, BASE)[0]
    b1 = eval_KQ(lD, degb, arc, ns, BASE, rlev_for_P(lD, degb, BASE, True))[0]
    w = max(w, abs(a1 / b1 - 1))
gate("Q0d eval_KQ with PART P's R_level = eval_KP (5 headline + 40 random cells)", w < 1e-12, f"rel {w:.1e}")
chk = [(53, 7), (58, 6), (59, 5), (60, 4), (61, 3)]
saved = {c: RQ.pop((c[0], c[1], n_full)) for c in chk}
certify_batch([(c[0], c[1], n_full) for c in chk])
wq = max(max(abs(x / y - 1) for x, y in zip(RQ[(c[0], c[1], n_full)], saved[c])) for c in chk)
gate("Q0a certification at k = 1 reproduces PART O's R_level at 5 cells", wq < 1e-12,
     f"rel {wq:.1e} ({time.time() - T0:.0f} s)")
stop_if_failed("PART Q0")

# =============================================================================================
# pruned, certified optimisation for sparse packings
# =============================================================================================
OPTS = [("base (cert+StC)", BASE)] + [(nm, o) for nm, o in ALTS
                                     if nm.split()[0] in ("(iii)", "(vi)", "(vii)")]
SPARSE = list(range(0, 16))
LB = {}                           # (name, ns) -> sorted list of (lower bound, lD, degb, arc)
for nm, o in OPTS:
    for j in SPARSE + [16]:
        ns = 2 ** j
        lst = []
        for arc in (0, 3):
            for lD, degb in CELLS:
                d_ = dec_q(lD, degb, arc, ns, o, res=False)
                if d_ is not None and math.isfinite(d_):
                    lst.append((d_, lD, degb, arc))
        lst.sort()
        LB[(nm, ns)] = lst


def proxy_best(nm, o, ns):
    """optimum with the full-packing R_level (PARTS O/P proxy)."""
    best = None
    for d0, lD, degb, arc in LB[(nm, ns)]:
        r = eval_KQ(lD, degb, arc, ns, o, rlev_q(lD, degb, n_full, o, True))
        if r is None:
            continue
        d_ = dec(r[0], lD)
        if math.isfinite(d_) and (best is None or d_ < best[0]):
            best = (d_, lD, degb, arc)
    return best


print("\nPART Q1  certification passes (pruned by the residue-free lower bound)", flush=True)
PROXY = {(nm, 2 ** j): proxy_best(nm, o, 2 ** j) for nm, o in OPTS for j in SPARSE}
n1 = certify_batch([(b[1], b[2], ns) for (nm, ns), b in PROXY.items()])
print(f"    pass 1: {n1} (cell, packing) instances certified ({time.time() - T0:.0f} s)", flush=True)
for it in range(2, 6):
    need = []
    for nm, o in OPTS:
        for j in SPARSE:
            ns = 2 ** j
            b = PROXY[(nm, ns)]
            U = dec_q(b[1], b[2], b[3], ns, o)
            for k_ in [x for x in LB[(nm, ns)] if x[0] < U]:
                if (k_[1], k_[2], ns) not in RQ:
                    need.append((k_[1], k_[2], ns))
    if not need:
        break
    n_ = certify_batch(need)
    print(f"    pass {it}: {n_} more instances certified ({time.time() - T0:.0f} s)", flush=True)

BEST, prune_ok, worst_gap = {}, True, math.inf
for nm, o in OPTS:
    for j in SPARSE:
        ns = 2 ** j
        best = None
        for d0, lD, degb, arc in LB[(nm, ns)]:
            if (lD, degb, ns) not in RQ:
                continue
            d_ = dec_q(lD, degb, arc, ns, o)
            if d_ is not None and (best is None or d_ < best[0]):
                best = (d_, lD, degb, arc,
                        model_P(lD, lD + degb, arc, ns, o, None, False)["logQ"])
        unc = [x[0] for x in LB[(nm, ns)] if (x[1], x[2], ns) not in RQ]
        gap = (min(unc) - best[0]) if unc else math.inf
        worst_gap = min(worst_gap, gap)
        prune_ok &= gap >= 0
        BEST[(nm, ns)] = best
    if nm == OPTS[0][0]:                      # full packing: PART O/P values are already certified
        BEST[(nm, n_full)] = OPT_P[16]
    else:
        r = REOPT_P[(nm, "full")]
        BEST[(nm, n_full)] = None if r is None else r[1]
gate("Q0b every uncertified cell has a residue-free lower bound >= the certified optimum",
     prune_ok, f"smallest margin {worst_gap:+.2e} bits (>= 0 required); certified instances {len(RQ) - len(CELLS)} "
     f"(of {16 * len(CELLS)} sparse)")
gate("Q0c every rounding residue |r_k| <= 1/2 + 1e-9", MAXR[0] <= 0.5 + 1e-9, f"max {MAXR[0]:.12f}")
stop_if_failed("PART Q1")

cellf = lambda b: f"{b[1]}/{b[2]}/{'on' if b[3] else 'off'}"
BN = OPTS[0][0]
print("\n    base (cert+StC): certified sparse instance vs the full-packing proxy of PARTS O/P")
print(f"    {'n_s':>5} | {'proxy (P4)':>18} | {'certified':>18} {'diff':>7} | R_level at the certified cell (bits)"
      f" | s1")
Q1 = []
for j in SPARSE:
    ns = 2 ** j
    bp, bq = OPT_P[j], BEST[(BN, ns)]
    R_ = RQ[(bq[1], bq[2], ns)]
    Q1.append((j, bp, bq, R_))
    print(f"    2^{j:<3} | {bp[0]:+7.2f} {cellf(bp):>10} | {bq[0]:+7.2f} {cellf(bq):>10} {bq[0] - bp[0]:+7.3f} | "
          + " ".join(f"{bits(x):5.2f}" for x in R_) + f" | 2^{bits(s1_of(bq[1], bq[2], ns)):.2f}")
allq = np.array([RQ[k] for k in RQ if k[2] < n_full])
print("    over all certified sparse instances, per level: min " + " ".join(f"{bits(x):.2f}" for x in allq.min(0))
      + " | max " + " ".join(f"{bits(x):.2f}" for x in allq.max(0)))

# =============================================================================================
print("\nPART Q2  decisions, re-optimised, against the certified base (sparse: PART Q; full: PART P)")
PACK = [("1 slot", 0), ("2^3", 3), ("2^8", 8), ("full", 16)]
base_best = {lab: BEST[(BN, 2 ** j)] for lab, j in PACK}
print(f"    {'base (cert+StC)':<32} | " + " | ".join(f"{lab:>6} {base_best[lab][0]:+7.2f} {cellf(base_best[lab]):<9}"
                                                      for lab, _ in PACK))
Q2 = {}
for nm, o in ALTS:
    outs = []
    for lab, j in PACK:
        ns = 2 ** j
        if any(nm == x[0] for x in OPTS) and ns < n_full:
            b = BEST[(nm, ns)]
        else:
            r = REOPT_P[(nm, lab)]
            b = None if r is None else r[1]
        Q2[(nm, lab)] = None if b is None else (b[0] - base_best[lab][0], b)
        outs.append(f"{lab:>6} {'none admissible':>17}" if b is None else
                    f"{lab:>6} {b[0] - base_best[lab][0]:+7.2f} {cellf(b):<9}")
    print(f"    {nm:<32} | " + " | ".join(outs))

print("\nPART Q3  fixed cells: Delta at the certified base optimum (sparse instances certified)")
HEADQ = [("1 slot", 0), ("2^3", 3), ("2^8", 8)]
Q3 = {}
for nm, o in ALTS:
    outs = []
    for lab, j in HEADQ:
        ns = 2 ** j
        bq = BEST[(BN, ns)]
        lD, degb, arc = bq[1], bq[2], bq[3]
        need = o["fold"] == "spread" and o["check"] == "can"
        if need and (lD, degb, ns) not in RQ:
            certify_batch([(lD, degb, ns)])
        d_ = dec_q(lD, degb, arc, ns, o)
        Q3[(nm, lab)] = None if d_ is None else d_ - bq[0]
        outs.append(f"{'inadm.':>16}" if d_ is None else f"{d_ - bq[0]:>+16.3f}")
    print(f"    {nm:<32} | " + " | ".join(f"{lab} {x}" for (lab, _), x in zip(HEADQ, outs)))

print("\nPART Q4  thresholds and log Q (base, certified)")
ALLB = {j: (BEST[(BN, 2 ** j)] if j < 16 else OPT_P[16]) for j in range(17)}
TH = {}
for thr in (-5.0, -10.0):
    ok = [j for j in ALLB if ALLB[j][0] < thr]
    TH[thr] = f"n_s<=2^{max(ok)}" if ok else "none"
    print(f"    largest n_s with T < 2^{thr:g}: {TH[thr]}")
lqs = sorted({ALLB[j][4] for j in ALLB})
for q in lqs:
    print(f"    log Q {q}: " + ", ".join(f"2^{j}" for j in ALLB if ALLB[j][4] == q))
print(f"\n    [INFO] Lemma sens not evaluated: {len(NO_SENS)}; overflowing tags: {len(NONFINITE)}")
print(f"    total runtime {time.time() - T0:.0f} s")

# =============================================================================================
print("\n" + "=" * 100)
print("SUMMARY BEGIN")
for j, bp, bq, R_ in Q1:
    print(f"Q1 2^{j}: proxy {bp[0]:+.3f} {cellf(bp)} | cert {bq[0]:+.3f} {cellf(bq)} ({bq[0] - bp[0]:+.3f}) logQ {bq[4]}"
          f" | R " + " ".join(f"{bits(x):.2f}" for x in R_))
print("Q1 all sparse instances min " + " ".join(f"{bits(x):.2f}" for x in allq.min(0)) + " | max "
      + " ".join(f"{bits(x):.2f}" for x in allq.max(0)))
for lab, _ in PACK:
    print(f"Q2 base {lab}: {base_best[lab][0]:+.2f} at {cellf(base_best[lab])}")
for nm, o in ALTS:
    print(f"Q2 {nm.strip()}: " + " | ".join(
        f"{lab} " + ("none admissible" if Q2[(nm, lab)] is None else
                     f"{Q2[(nm, lab)][0]:+.2f} at {cellf(Q2[(nm, lab)][1])}") for lab, _ in PACK))
for nm, o in ALTS:
    print(f"Q3 {nm.strip()}: " + " | ".join(
        f"{lab} " + ("inadm." if Q3[(nm, lab)] is None else f"{Q3[(nm, lab)]:+.3f}") for lab, _ in HEADQ))
for thr, v in TH.items():
    print(f"Q4 threshold T<2^{thr:g}: {v}")
for q in lqs:
    print(f"Q4 logQ {q}: " + ", ".join(f"2^{j}" for j in ALLB if ALLB[j][4] == q))
print(f"Q certified sparse instances {len(RQ) - len(CELLS)} | prune margin {worst_gap:+.2e} | max|r| {MAXR[0]:.12f}"
      f" | no-sens {len(NO_SENS)} | overflow {len(NONFINITE)} | runtime {time.time() - T0:.0f} s")
print("GATE FAILURES:", FAILS if FAILS else "none")
print("SUMMARY END")
print("=" * 100)
