#!/usr/bin/env python3
"""
estimate_security.py  --  lattice-estimator rows for the ephemeral sparse key s'
                          (N = 2^17, H = 64) at the top of the chain, for the new
                          landscape cells.

WHAT IS ESTIMATED
  The sparse key s' exists from Q_boot downward, so it is read at the TOP of the
  chain (log Q of the cell). m = N, as in the recorded table (tab:security).
  Secret: balanced sparse ternary, H/2 entries +1 and H/2 entries -1.
  Error:  discrete Gaussian, sigma = 3.19 (default; --sigma to change).
  Attacks: primal_bdd, and primal_hybrid at mitm=True, babai=True (the recorded
  "hybrid" column, bdd_mitm_hybrid). The binding figure is min(bdd, hybrid).

HARD GATE (runs first, always)
  Reproduce two recorded rows of tab:security at commit 53da598:
    N=2^17, H=64, log Q=1332: bdd 375.7, hybrid 168.0
    N=2^17, H=64, log Q=1385: bdd 359.1, hybrid 165.4
  Tolerance 0.2 bits. If the gate fails, STOP: the error distribution or the
  estimator commit differs from the recorded runs, and no new row is comparable.

WHICH log Q
  Default list = the best cells of the sweep, with and without the SubSum level:
    no SubSum : 1432 1480 1528 1552 1572 1598 1624 1650 1676
    SubSum    : 1489 1539 1589 1657 1684 1711   (= above + lD for sparse cells)
  After the sweep's PART G / PART F, pass the exact values instead:
    --logq 1552 1657 ...
  Security is decreasing in log Q, so a row at a larger log Q is a lower bound
  for any cell with a smaller log Q.

USAGE (SageMath is required by the estimator):
  sage -python estimate_security.py --estimator /path/to/lattice-estimator --gate-only
  sage -python estimate_security.py --estimator /path/to/lattice-estimator
  sage -python estimate_security.py --estimator ... --logq 1552 1657 1684
  Each hybrid call can take minutes at N = 2^17; --timeout (default 3600 s per call).
  The summary block is printed between SUMMARY BEGIN / SUMMARY END.
"""
import argparse
import math
import os
import signal
import subprocess
import sys
import time

REC_COMMIT = "53da598"
LOGN = 17
H = 64
GATE_ROWS = [(1332, 375.7, 168.0), (1385, 359.1, 165.4)]
DEFAULT_LOGQ = sorted({1432, 1480, 1528, 1552, 1572, 1598, 1624, 1650, 1676,
                       1489, 1539, 1589, 1657, 1684, 1711})
TOL = 0.2


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


def rop_bits(res):
    try:
        v = float(res["rop"])
    except Exception:
        return float("nan")
    return math.log2(v) if v > 0 and math.isfinite(v) else float("inf")


def run_attack(fn, timeout):
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(int(timeout))
    t0 = time.time()
    try:
        res = fn()
        return rop_bits(res), time.time() - t0, ""
    except Timeout:
        return float("nan"), time.time() - t0, "timeout"
    except Exception as ex:  # report, do not hide
        return float("nan"), time.time() - t0, f"error: {type(ex).__name__}: {ex}"
    finally:
        signal.alarm(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimator", default=os.environ.get("ESTIMATOR_PATH", "."),
                    help="path to the lattice-estimator checkout")
    ap.add_argument("--logq", type=int, nargs="*", default=None)
    ap.add_argument("--sigma", type=float, default=3.19)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--gate-only", action="store_true")
    ap.add_argument("--skip-gate", action="store_true",
                    help="ONLY for a rerun after the gate already passed at the same commit")
    a = ap.parse_args()

    sys.path.insert(0, os.path.abspath(a.estimator))
    try:
        from estimator import LWE, ND
    except Exception as ex:
        sys.exit(f"cannot import the estimator from {a.estimator}: {ex}\n"
                 f"run with `sage -python` and pass --estimator /path/to/lattice-estimator")
    try:
        commit = subprocess.check_output(["git", "-C", a.estimator, "rev-parse", "--short=7", "HEAD"],
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = "unknown"

    N = 2 ** LOGN
    Xs = ND.SparseTernary(n=N, p=H // 2, m=H // 2)   # keywords valid in old and new signatures
    Xe = ND.DiscreteGaussian(a.sigma)

    def row(logq):
        params = LWE.Parameters(n=N, q=2 ** logq, Xs=Xs, Xe=Xe, m=N, tag=f"N2^{LOGN}_H{H}_logQ{logq}")
        b, tb, nb = run_attack(lambda: LWE.primal_bdd(params), a.timeout)
        print(f"    logQ={logq}: bdd {b:7.1f}  ({tb:6.0f}s) {nb}", flush=True)
        h, th, nh = run_attack(lambda: LWE.primal_hybrid(params, mitm=True, babai=True), a.timeout)
        print(f"    logQ={logq}: hybrid {h:7.1f}  ({th:6.0f}s) {nh}", flush=True)
        return dict(logq=logq, bdd=b, hyb=h, note=(nb + " " + nh).strip(), t=tb + th)

    print("=" * 90)
    print(f"estimate_security.py  N=2^{LOGN}  H={H}  m=N  sigma={a.sigma}  commit={commit} "
          f"(recorded {REC_COMMIT})")
    print("=" * 90)
    if commit != REC_COMMIT:
        print(f"  NOTE: estimator commit {commit} != recorded {REC_COMMIT}; the gate decides comparability.")

    fails, gate_rows = [], []
    if not a.skip_gate:
        print("\nGATE  reproduce recorded rows of tab:security")
        for logq, want_b, want_h in GATE_ROWS:
            r = row(logq)
            gate_rows.append(r)
            ok = abs(r["bdd"] - want_b) <= TOL and abs(r["hyb"] - want_h) <= TOL
            print(f"    [{'PASS' if ok else 'FAIL'}] logQ={logq}: bdd {r['bdd']:.1f} (want {want_b}), "
                  f"hybrid {r['hyb']:.1f} (want {want_h})", flush=True)
            if not ok:
                fails.append(logq)
        if fails:
            print("\n  GATE FAILED -- see the output above; no new row is comparable.")
            sys.exit(1)
        print("  GATE PASSED.")

    rows = []
    if not a.gate_only:
        todo = a.logq if a.logq else DEFAULT_LOGQ
        print(f"\nROWS  log Q in {todo}")
        for logq in todo:
            rows.append(row(logq))

    print("\n" + "=" * 90)
    print("SUMMARY BEGIN")
    print(f"commit={commit} sigma={a.sigma} m=N N=2^{LOGN} H={H} gate={'skipped' if a.skip_gate else 'passed'}")
    for r in gate_rows + rows:
        mn = min(r["bdd"], r["hyb"])
        print(f"EST logQ={r['logq']}: bdd={r['bdd']:.1f} hybrid={r['hyb']:.1f} min={mn:.1f} "
              f"t={r['t']:.0f}s {r['note']}")
    print("SUMMARY END")
    print("=" * 90)


if __name__ == "__main__":
    main()
