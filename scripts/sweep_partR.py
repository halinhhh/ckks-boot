#!/usr/bin/env python3
"""
sweep_partR.py  --  PART R (1 Oct 2026): the numbers of Sec. 5.4, Sec. 7 and tab:refconst that the
paper still quotes from the residue-free port, re-measured on the certified configuration
(cert+StC: spread fold, certified CtS residues, sparse instances of PART Q).

HOW IT RUNS
  It runs sweep_partQ.py silently (which runs P, O and the sweep; ~8 min on a laptop) and reuses
  its certified residues. Cells that a re-optimisation below needs and that PART Q did not certify
  are certified here with the same pruning (residue-free tag = lower bound). Then ~2-4 min.
      python3 /path/to/sweep_partR.py    (next to sweep_partQ.py, sweep_partP.py, sweep_partO.py)

WHAT IT MEASURES (everything on the certified configuration unless stated)
  R1  the reference cell, term by term (the three blocks of tab:refconst).
  R2  at the reference cell and the optimum of every packing (17): L_R / K_eff - 1, L_I, the gap
      B_cts - B_cj (bits), the input share (L_R B_cts + L_I B_cj) / E*, the tag amplification of
      the input error (E_eval - E_int) / B_cts against L_R (the "seven bits" of the abstract), the
      margins of E_cj and E_post below the dominant term, and the dominant term.
  R3  the full-packing optimum: sine floor, its gap to T, and noise vs sine.
  R4  ablations, re-optimised at 1, 2^3, 2^8 and 2^16 slots: without Lemma parseval (sup-norm sine
      term on the enlarged domain, as V4c) and without Lemma sens (gate-by-gate E_eval).

GATES
  R0a-c with every residue off and S folded at the end (the residue-free port), eval_R reproduces
       the sweep's eval_K, eval_K_noP and the tau with the old noise tag (45 cells, 1e-12 of that tau).
  (The pruning lower bound is the residue-free value with the SPREAD fold, below the certified one.)
  R0d base optima reproduce PART Q (17 packings, exact).
  R0e every re-optimisation is certified over the whole domain (pruning margin >= 0).
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
    print(f"Cannot find {PART_Q}. Put sweep_partR.py next to sweep_partQ.py, P, O and the sweep.")
    sys.exit(1)
print("=" * 100)
print("sweep_partR.py   PART R: residue-free numbers of the paper, re-measured on the certified configuration")
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
G = O["G"]
N, n_full, K = O["N"], O["n_full"], O["K"]
C_DENSE, C_SPARSE = O["C_DENSE"], O["C_SPARSE"]
q0lam, parseval_factor, c_of, part_of = O["q0lam"], O["parseval_factor"], O["c_of"], O["part_of"]
amp_inj, amp_after_first, sens_bounds = O["amp_inj"], O["amp_after_first"], O["sens_bounds"]
bks, bits, dec, sup_f = O["bks"], O["bits"], O["dec"], G["sup_f"]
CELLS, n_cells = O["CELLS"], len(O["CELLS"])
model_P, evalmod_mu, stc_residue = P["model_P"], P["evalmod_mu"], P["stc_residue"]
BASE, OLD = P["BASE"], P["OLD_BASE"]        # OLD = the residue-free port (S folded at the end)
RQ, certify_batch, rlev_q, eval_KQ = Q["RQ"], Q["certify_batch"], Q["rlev_q"], Q["eval_KQ"]
BEST_Q, BN, OPT_P = Q["BEST"], Q["BN"], P["OPT"]
FAILS = []
np.seterr(over="ignore", invalid="ignore")


def gate(name, ok, detail=""):
    print(f"    [{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def stop_if_failed(part):
    if FAILS:
        print(f"\n  {part} GATE FAILED:", FAILS, " -- see the output above.")
        sys.exit(1)


def eval_R(lD, degb, arc, ns, mode="base", res=True, o=None):
    """eval_KQ (base options) with the ablations: mode 'noPars' (sup-norm sine term on the enlarged
    domain, V4c) and 'noSens' (noise from the gate-by-gate tag E_eval). Returns (tau, detail)."""
    o = BASE if o is None else o
    lq0 = lD + degb
    D, q0 = 2.0 ** lD, 2.0 ** lq0
    mo = model_P(lD, lq0, arc, ns, o, rlev_q(lD, degb, ns, o, res), res)
    eps_in = (D + mo["B_in"]) / q0
    if eps_in >= 0.5:
        return None
    A_sup = c_of(ns) * ns
    back = 2.0 ** degb
    Pp = part_of(ns)
    QL = q0lam(lq0, mo["Keff"])
    interp = A_sup * QL
    if mode == "noPars":
        Xi = (mo["B_cts"] + mo["B_conj"] + bks(2, lq0, lD) / D) / D * q0 * mo["Keff"]
        sine = A_sup * q0 * sup_f((D + Xi) / q0, arc)
    else:
        sine = q0 * parseval_factor(eps_in, arc)
    t = dict(approx=interp + sine, stc=C_DENSE * amp_inj(Pp), bsd=mo["B_sd"] * amp_after_first(Pp),
             conj=ns * back * mo["B_cj"], post=(2 * ns - A_sup) * back * mo["CS"])
    B_im = mo["B_conj"] + mo["B_sdb"]
    delta, eta = mo["B_cts"] / D, B_im / D
    sb, eD_int, eD_K = None, math.inf, math.inf
    try:
        if math.exp(math.acosh(1 + delta)) * 1.01 >= 40.0:
            raise ValueError("rho grid")
        sb = sens_bounds(mo["Keff"], arc, delta, eta)
        eD_int = evalmod_mu(lD, lq0, arc, mo["Keff"], mo["CS"], mo["em_top"], sb["sz"], sb["su"],
                            delta * sb["L_R"] + eta * sb["L_I"], res)
        eD_K = eD_int + sb["L_R"] * mo["B_cts"] + sb["L_I"] * B_im
    except (ValueError, OverflowError):
        pass
    E_use = mo["eD"] if mode == "noSens" else min(mo["eD"], eD_K)
    t["noise"] = A_sup * back * E_use
    if res:
        t["stc_res"] = stc_residue(Pp, D, mo["B_in"], sum(t.values()))
    tot = sum(t.values())
    if not math.isfinite(tot):
        tot = math.inf
    return tot, dict(t=t, mo=mo, sb=sb, eD_int=eD_int, eD_K=eD_K, E_use=E_use, interp=interp, sine=sine,
                     A_sup=A_sup, back=back, D=D, q0=q0)


def d_R(lD, degb, arc, ns, mode, res=True):
    r = eval_R(lD, degb, arc, ns, mode, res)
    return None if r is None else dec(r[0], lD)


# =============================================================================================
print("\nPART R0  GATES", flush=True)
rng = np.random.default_rng(20261004)
RND = [CELLS[j] + (int(rng.choice([0, 3])), int(2 ** rng.integers(0, 17)))
       for j in rng.choice(n_cells, 45, replace=False)]
wa = wb = wc = 0.0
for lD, degb, arc, ns in RND:
    b3 = G["eval_K"](lD, degb, arc, ns)[1]          # tau with the old noise tag
    a1 = eval_R(lD, degb, arc, ns, "base", False, OLD)[0]
    b1 = G["eval_K"](lD, degb, arc, ns)[0]
    a2 = eval_R(lD, degb, arc, ns, "noPars", False, OLD)[0]
    b2 = G["eval_K_noP"](lD, degb, arc, ns)[0]
    a3 = eval_R(lD, degb, arc, ns, "noSens", False, OLD)[0]
    # eval_K and eval_K_noP form tau_old - noise_old + noise_K, so their own rounding is relative to
    # tau_old (as in gates M0b and N0b)
    wa = max(wa, abs(a1 - b1) / b3)
    wb = max(wb, abs(a2 - b2) / b3)
    wc = max(wc, abs(a3 / b3 - 1))
gate("R0a residues off, end fold: base = the sweep's eval_K (45 cells)", wa < 1e-12, f"rel {wa:.1e}")
gate("R0b residues off, end fold, no-Parseval = the sweep's eval_K_noP (45 cells; 1e-9: sup_f is a grid max)",
     wb < 1e-9, f"rel {wb:.1e}")
gate("R0c residues off, end fold, no-sens = the sweep's tau with the old noise tag (45 cells)", wc < 1e-12, f"rel {wc:.1e}")
OPT = {j: (BEST_Q[(BN, 2 ** j)] if j < 16 else OPT_P[16]) for j in range(17)}
wd = 0.0
for j, b in OPT.items():
    wd = max(wd, abs(d_R(b[1], b[2], b[3], 2 ** j, "base") - b[0]))
gate("R0d base optima reproduce PART Q / P (17 packings)", wd < 1e-12, f"max dev {wd:.1e}")
stop_if_failed("PART R0")


# =============================================================================================
def best_certified(ns, mode):
    """certified optimum of a mode over the whole domain; pruning by the residue-free value."""
    lb = []
    for arc in (0, 3):
        for lD, degb in CELLS:
            d0 = d_R(lD, degb, arc, ns, mode, res=False)
            if d0 is not None and math.isfinite(d0):
                lb.append((d0, lD, degb, arc))
    lb.sort()
    need = lambda c: (c[1], c[2], ns) not in RQ
    certify_batch([(c[1], c[2], ns) for c in lb[:1] if need(c)])
    for _ in range(6):
        cert = [(d_R(c[1], c[2], c[3], ns, mode), c) for c in lb if not need(c)]
        cert = [(d, c) for d, c in cert if d is not None and math.isfinite(d)]
        best = min(cert, key=lambda x: x[0])
        more = [(c[1], c[2], ns) for c in lb if c[0] < best[0] and need(c)]
        if not more:
            unc = [c[0] for c in lb if need(c)]
            margin = (min(unc) - best[0]) if unc else math.inf
            c = best[1]
            return (best[0], c[1], c[2], c[3]), margin
        certify_batch(more)
    raise RuntimeError("pruning did not converge")


print("\nPART R4  ablations, re-optimised and certified (logD 36..62, degb 2..14, arcsine on/off)", flush=True)
cellf = lambda b: f"{b[1]}/{b[2]}/{'on' if b[3] else 'off'}"
PACK = [("1 slot", 0), ("2^3", 3), ("2^8", 8), ("full", 16)]
ABL, margins = {}, []
for lab, j in PACK:
    for mode in ("noPars", "noSens"):
        b, m = best_certified(2 ** j, mode)
        ABL[(mode, lab)] = b
        margins.append(m)
    print(f"    {lab:>6}: base {OPT[j][0]:+.3f} {cellf(OPT[j])} | no Parseval {ABL[('noPars', lab)][0]:+.3f} "
          f"{cellf(ABL[('noPars', lab)])} (gain {ABL[('noPars', lab)][0] - OPT[j][0]:+.3f}) | no sens "
          f"{ABL[('noSens', lab)][0]:+.3f} {cellf(ABL[('noSens', lab)])} (gain {ABL[('noSens', lab)][0] - OPT[j][0]:+.3f})",
          flush=True)
gate("R0e every ablation optimum is certified over the whole domain (pruning margin >= 0)",
     min(margins) >= 0, f"smallest margin {min(margins):+.2e} ({time.time() - T0:.0f} s)")
stop_if_failed("PART R4")


# =============================================================================================
def row(lab, lD, degb, arc, ns):
    tot, d = eval_R(lD, degb, arc, ns, "base")
    t, mo, sb = d["t"], d["mo"], d["sb"]
    D = d["D"]
    Keff = mo["Keff"]
    terms = dict(noise=t["noise"], stc=t["stc"], post=t["post"], interp=d["interp"], sine=d["sine"],
                 conj=t["conj"], bsd=t["bsd"], stc_res=t.get("stc_res", 0.0))
    dom = max(terms, key=terms.get)
    others = max(v for k_, v in terms.items() if k_ != dom)
    LRB = sb["L_R"] * mo["B_cts"]
    LIB = sb["L_I"] * (mo["B_conj"] + mo["B_sdb"])
    return dict(lab=lab, cell=f"{lD}/{degb}/{'on' if arc else 'off'}", dec=dec(tot, lD), terms=terms, dom=dom,
                dom_gap=bits(terms[dom]) - bits(others), B_cts=mo["B_cts"], B_cj=mo["B_conj"],
                cts_res=mo["add_cts"], eD=mo["eD"], eD_int=d["eD_int"], LRB=LRB, LIB=LIB, Estar=d["E_use"],
                L_R=sb["L_R"], L_I=sb["L_I"], Keff=Keff, share=(LRB + LIB) / d["E_use"],
                amp=(mo["eD"] - d["eD_int"]) / mo["B_cts"],
                cj_gap=bits(terms[dom]) - bits(t["conj"]), post_gap=bits(terms[dom]) - bits(t["post"]),
                sine_dec=bits(d["sine"]) - lD, noise_vs_sine=bits(t["noise"]) - bits(d["sine"]))


ROWS = [row("ref", 53, 7, 0, n_full)] + [row(f"2^{j}", OPT[j][1], OPT[j][2], OPT[j][3], 2 ** j) for j in range(17)]
ref = ROWS[0]
print("\nPART R1  reference cell 53/7/off, certified configuration (polynomial units at scale Delta, log2)")
print(f"    B_cts 2^{bits(ref['B_cts']):.3f} (CtS residue 2^{bits(ref['cts_res']):.2f}) | E_eval 2^{bits(ref['eD']):.3f} | "
      f"E_int 2^{bits(ref['eD_int']):.2f} | L_R B_cts 2^{bits(ref['LRB']):.2f} | L_I B_cj 2^{bits(ref['LIB']):.2f} | "
      f"E* 2^{bits(ref['Estar']):.2f} | L_R {ref['L_R']:.7f} (K_eff {ref['Keff']:.7f})")
print("    terms: " + " | ".join(f"{k_} 2^{bits(v):.2f}" for k_, v in ref["terms"].items() if v > 0)
      + f" | log2 T {ref['dec']:+.3f} | dominant {ref['dom']}, {ref['dom_gap']:.2f} bits above the next")

print("\nPART R2  reference cell and the optimum of every packing")
print(f"    {'cell':>14} | {'dec':>8} | L_R/Keff-1  L_I    | Bcts-Bcj | input share | amp/L_R (bits) | "
      f"E_cj, E_post below dom | dom")
for r in ROWS:
    print(f"    {r['lab'] + ' ' + r['cell']:>14} | {r['dec']:+8.3f} | 2^{bits(r['L_R'] / r['Keff'] - 1):6.2f}  2^{bits(r['L_I']):5.2f} | "
          f"{bits(r['B_cts']) - bits(r['B_cj']):8.2f} | {100 * r['share']:10.2f}% | {bits(r['amp']) - bits(r['L_R']):14.2f} | "
          f"{r['cj_gap']:6.2f} {r['post_gap']:6.2f} | {r['dom']}")
rep = ROWS          # every reported cell
S2 = dict(LRK=max(bits(r["L_R"] / r["Keff"] - 1) for r in rep), LI=max(bits(r["L_I"]) for r in rep),
          gap=min(bits(r["B_cts"]) - bits(r["B_cj"]) for r in rep), share=max(r["share"] for r in rep[1:]),
          share_min=min(r["share"] for r in rep[1:]), amp=min(bits(r["amp"]) - bits(r["L_R"]) for r in rep),
          amp_ref=bits(ref["amp"]), cj=min(r["cj_gap"] for r in rep), post=min(r["post_gap"] for r in rep),
          doms=sorted(set(r["dom"] for r in rep)))
full = ROWS[17]
print("\nPART R3  full-packing optimum " + full["cell"])
print(f"    sine floor 2^{full['sine_dec']:+.3f} (decoded), T 2^{full['dec']:+.3f}, gap {full['sine_dec'] - full['dec']:.3f} bits;"
      f" noise - sine {full['noise_vs_sine']:+.3f} bits; reference-cell sine 2^{ref['sine_dec']:+.3f}")
print(f"\n    total runtime {time.time() - T0:.0f} s")

# =============================================================================================
print("\n" + "=" * 100)
print("SUMMARY BEGIN")
print(f"R1 ref: B_cts 2^{bits(ref['B_cts']):.3f} CtSres 2^{bits(ref['cts_res']):.2f} E_eval 2^{bits(ref['eD']):.3f} "
      f"E_int 2^{bits(ref['eD_int']):.2f} LRB 2^{bits(ref['LRB']):.2f} LIB 2^{bits(ref['LIB']):.2f} E* 2^{bits(ref['Estar']):.2f} "
      f"L_R {ref['L_R']:.7f} Keff {ref['Keff']:.7f}")
print("R1 ref terms: " + " ".join(f"{k_} 2^{bits(v):.2f}" for k_, v in ref["terms"].items() if v > 0)
      + f" | dec {ref['dec']:+.3f} | dominant {ref['dom']} +{ref['dom_gap']:.2f}")
for r in ROWS:
    print(f"R2 {r['lab']} {r['cell']}: dec {r['dec']:+.3f} LRK 2^{bits(r['L_R'] / r['Keff'] - 1):.2f} LI 2^{bits(r['L_I']):.2f} "
          f"gap {bits(r['B_cts']) - bits(r['B_cj']):.2f} share {100 * r['share']:.2f}% amp-LR {bits(r['amp']) - bits(r['L_R']):.2f} "
          f"cj {r['cj_gap']:.2f} post {r['post_gap']:.2f} dom {r['dom']} +{r['dom_gap']:.2f} noise-sine {r['noise_vs_sine']:+.2f}")
print(f"R2 summary: L_R/Keff-1 <= 2^{S2['LRK']:.2f} | L_I <= 2^{S2['LI']:.2f} | B_cts-B_cj >= {S2['gap']:.2f} | "
      f"input share {100 * S2['share_min']:.1f}%..{100 * S2['share']:.1f}% at the optima | amp/L_R >= {S2['amp']:.2f} bits "
      f"(ref amp 2^{S2['amp_ref']:.2f}) | E_cj >= {S2['cj']:.2f}, E_post >= {S2['post']:.2f} below dom | dominant: {S2['doms']}")
print(f"R3 full {full['cell']}: sine 2^{full['sine_dec']:+.3f} T 2^{full['dec']:+.3f} gap {full['sine_dec'] - full['dec']:.3f} "
      f"noise-sine {full['noise_vs_sine']:+.3f} | ref sine 2^{ref['sine_dec']:+.3f}")
for lab, j in PACK:
    a, b = ABL[("noPars", lab)], ABL[("noSens", lab)]
    print(f"R4 {lab}: base {OPT[j][0]:+.3f} {cellf(OPT[j])} | noPars {a[0]:+.3f} {cellf(a)} ({a[0] - OPT[j][0]:+.3f}) | "
          f"noSens {b[0]:+.3f} {cellf(b)} ({b[0] - OPT[j][0]:+.3f})")
print(f"R pruning margin {min(margins):+.2e} | certified instances {len(RQ)} | runtime {time.time() - T0:.0f} s")
print("GATE FAILURES:", FAILS if FAILS else "none")
print("SUMMARY END")
print("=" * 100)
