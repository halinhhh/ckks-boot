#!/usr/bin/env python3
"""
measure_ref_n17.py  --  Step 4, task B (1 Oct 2026)

PURPOSE
One matched measured reference for Sec. VII ("The gap ...") and (E5): the maximum decoded error
of an UNVERIFIED library bootstrap at the paper's ring dimension N = 2^17, FULL packing
(2^16 slots), unit-scale slots.
It is NOT a certificate and NOT a parameter source for any statement of the paper.

WHAT IS MEASURED
  OpenFHE CKKS (auxiliary-modulus HYBRID key switch, FLEXIBLEAUTO), EvalBootstrap at full
  packing. Input: a fresh encryption at level depth-1 (a ciphertext that has used its levels).
  Error of one trial = max over all 2^16 slots of |decoded - input| (complex, imaginary part kept).
  Reported: log2 of the maximum over all trials and key sets, and per input family.
  Input families per key set (all with ||z||_inf <= 1, real):
    ones  z = (1, ..., 1)        the sine floor of Lemma parseval is attained here
    alt   z = (+1, -1, +1, ...)
    unif  z uniform in [-1, 1], one seed per trial
  Default: KEYSETS = 3 key sets x TRIALS = 10 trials = 30 trials (trial 0 = ones, 1 = alt).

ARCHITECTURE (no OOM accumulation)
  The parent runs one SUBPROCESS PER KEY SET (setup + keygen + its trials), then exits it.
  Every trial is appended to ref_n17_results.jsonl with a config fingerprint; a rerun skips key
  sets that are already complete for the same fingerprint, so an interrupted run resumes.

USAGE  (any Python environment with the openfhe package)
  python3 measure_ref_n17.py --dry    # 1 key set, 1 bootstrap: time + RSS
  python3 measure_ref_n17.py          # full run (resumable)
  python3 measure_ref_n17.py --report # summary block from the jsonl only
  Set DEV = True first for a fast smoke test at N = 2^12 (security off). DEV numbers are NEVER
  quoted: the summary block says so and the report refuses to mix DEV and FINAL records.

MEMORY WARNING
  Full packing at N = 2^17 needs many rotation keys. Run --dry FINAL first and check its
  "[mem]" line: if peak RSS is close to your RAM, raise LEVEL_BUDGET (e.g. [5, 5]) before the
  full run; the budget is part of the fingerprint and of the summary block.
"""
import json
import hashlib
import math
import os
import platform
import resource
import subprocess
import sys
import time

# ----------------------------------------------------------------------------- knobs
DEV = False                    # True: N = 2^12, HEStd_NotSet. NEVER quote DEV numbers.
RING_FINAL = 1 << 17           # the paper's N
RING_DEV = 1 << 12
SCALE_MOD = 59                 # OpenFHE's recommended bootstrapping setting for 64-bit native
FIRST_MOD = 60
LEVEL_BUDGET = [4, 4]          # [CtS, StC]
LEVELS_AFTER = 4               # levels left after bootstrap (as in the paired N = 2^17 experiment)
KEYDIST = "SPARSE_TERNARY"     # or "UNIFORM_TERNARY", or "SPARSE_ENCAPSULATED" if your build has it
KEYSETS = 3
TRIALS = 10                    # per key set; trial 0 = ones, trial 1 = alt, others = unif
SEED = 20261001

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "ref_n17_results.jsonl")


def config():
    return dict(DEV=DEV, N=RING_DEV if DEV else RING_FINAL, scale_mod=SCALE_MOD,
                first_mod=FIRST_MOD, level_budget=LEVEL_BUDGET, levels_after=LEVELS_AFTER,
                keydist=KEYDIST, trials=TRIALS, seed=SEED)


def fingerprint():
    return hashlib.sha256(json.dumps(config(), sort_keys=True).encode()).hexdigest()[:12]


def rss_gib():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2 ** 30 if platform.system() == "Darwin" else r / 2 ** 20   # bytes vs KiB


def lib_version():
    try:
        import openfhe
    except Exception as e:
        return f"import failed: {e}"
    for name in ("get_native_library_version", "GetOpenFHEVersion", "__version__"):
        v = getattr(openfhe, name, None)
        if v is not None:
            try:
                return str(v() if callable(v) else v)
            except Exception:
                pass
    try:
        from importlib.metadata import version
        return version("openfhe")
    except Exception:
        return "unknown: record the OpenFHE commit by hand (git -C <openfhe src> rev-parse HEAD)"


def inputs(family, n, trial_seed):
    if family == "ones":
        return [1.0] * n
    if family == "alt":
        return [1.0 if j % 2 == 0 else -1.0 for j in range(n)]
    try:
        import numpy as np
        return np.random.default_rng(trial_seed).uniform(-1.0, 1.0, n).tolist()
    except Exception:
        import random
        rng = random.Random(trial_seed)
        return [rng.uniform(-1.0, 1.0) for _ in range(n)]


def family_of(t):
    return "ones" if t == 0 else ("alt" if t == 1 else "unif")


# ----------------------------------------------------------------------------- worker
def worker(ks, dry=False):
    from openfhe import (CCParamsCKKSRNS, GenCryptoContext, PKESchemeFeature, ScalingTechnique,
                         SecretKeyDist, SecurityLevel, FHECKKSRNS)
    try:
        from openfhe import get_native_int
        if get_native_int() != 64:
            sys.exit(f"native int is {get_native_int()}, the knobs assume 64-bit native")
    except ImportError:
        pass
    kd = getattr(SecretKeyDist, KEYDIST, None)
    if kd is None:
        sys.exit(f"SecretKeyDist.{KEYDIST} not in this build; pick another KEYDIST")
    p = CCParamsCKKSRNS()
    p.SetSecretKeyDist(kd)
    if DEV:
        p.SetSecurityLevel(SecurityLevel.HEStd_NotSet)
        p.SetRingDim(RING_DEV)
    else:
        p.SetSecurityLevel(SecurityLevel.HEStd_128_classic)
        p.SetRingDim(RING_FINAL)          # above the 128-bit minimum for this log Q: allowed
    p.SetScalingModSize(SCALE_MOD)
    p.SetFirstModSize(FIRST_MOD)
    p.SetScalingTechnique(ScalingTechnique.FLEXIBLEAUTO)
    depth = LEVELS_AFTER + FHECKKSRNS.GetBootstrapDepth(LEVEL_BUDGET, kd)
    p.SetMultiplicativeDepth(depth)

    t0 = time.time()
    cc = GenCryptoContext(p)
    for f in (PKESchemeFeature.PKE, PKESchemeFeature.KEYSWITCH, PKESchemeFeature.LEVELEDSHE,
              PKESchemeFeature.ADVANCEDSHE, PKESchemeFeature.FHE):
        cc.Enable(f)
    N = cc.GetRingDimension()
    want = RING_DEV if DEV else RING_FINAL
    if N != want:
        sys.exit(f"ring dimension {N}, expected {want}: refusing to measure")
    n = N // 2
    cc.EvalBootstrapSetup(LEVEL_BUDGET, [0, 0], n)
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalBootstrapKeyGen(keys.secretKey, n)
    t_key = time.time() - t0
    print(f"[ks {ks}] N={N} slots={n} depth={depth} keydist={KEYDIST} budget={LEVEL_BUDGET} "
          f"keygen {t_key:.0f}s", flush=True)
    print(f"[mem] ks {ks}: peak RSS after keygen {rss_gib():.1f} GiB", flush=True)

    ntr = 1 if dry else TRIALS
    for t in range(ntr):
        fam = family_of(t)
        z = inputs(fam, n, SEED + 1000 * ks + t)
        pt = cc.MakeCKKSPackedPlaintext(z, 1, depth - 1)
        pt.SetLength(n)
        ct = cc.Encrypt(keys.publicKey, pt)
        tb = time.time()
        ct2 = cc.EvalBootstrap(ct)
        t_boot = time.time() - tb
        out = cc.Decrypt(ct2, keys.secretKey)
        out.SetLength(n)
        try:
            dec = out.GetCKKSPackedValue()
            cplx = True
        except Exception:
            dec = out.GetRealPackedValue()
            cplx = False
        if len(dec) < n:
            sys.exit(f"decoded length {len(dec)} < {n}")
        err = max(abs(complex(dec[j]) - z[j]) for j in range(n))
        lvl = ct2.GetLevel() if hasattr(ct2, "GetLevel") else -1
        rec = dict(fp=fingerprint(), cfg=config(), ks=ks, trial=t, family=fam, max_err=err,
                   log2_err=math.log2(err) if err > 0 else float("-inf"), complex_decode=cplx,
                   boot_s=t_boot, keygen_s=t_key, rss_gib=rss_gib(), out_level=lvl, depth=depth,
                   lib=lib_version(), dry=dry)
        print(f"[ks {ks}] trial {t} {fam:>4}: log2 max err {rec['log2_err']:+.3f}  "
              f"boot {t_boot:.0f}s  levels left {depth - lvl}  RSS {rec['rss_gib']:.1f} GiB", flush=True)
        if not dry:
            with open(OUT, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
    print(f"[mem] ks {ks}: peak RSS {rss_gib():.1f} GiB", flush=True)


# ----------------------------------------------------------------------------- parent
def load():
    if not os.path.exists(OUT):
        return []
    return [json.loads(l) for l in open(OUT) if l.strip()]


def report():
    fp = fingerprint()
    recs = [r for r in load() if r["fp"] == fp and not r.get("dry")]
    others = {r["fp"] for r in load()} - {fp}
    print("=" * 90)
    print("SUMMARY BEGIN")
    if DEV:
        print("*** DEV RUN (N = 2^12, security off): DO NOT QUOTE ANY NUMBER BELOW ***")
    c = config()
    print(f"B-ref config: N=2^{int(math.log2(c['N']))} full packing, scale {SCALE_MOD}/first {FIRST_MOD}, "
          f"FLEXIBLEAUTO, budget {LEVEL_BUDGET}, levels after {LEVELS_AFTER}, keydist {KEYDIST}, fp {fp}")
    if not recs:
        print("B-ref: no records for this fingerprint")
    else:
        print(f"B-ref lib: {recs[-1]['lib']}")
        print(f"B-ref complex decode: {all(r['complex_decode'] for r in recs)}")
        done = sorted({r['ks'] for r in recs})
        print(f"B-ref: {len(recs)} trials over key sets {done}")
        for ks in done:
            rr = [r for r in recs if r["ks"] == ks]
            print(f"B-ref ks {ks}: {len(rr)} trials, max log2 err {max(r['log2_err'] for r in rr):+.3f}, "
                  f"keygen {rr[0]['keygen_s']:.0f}s, median boot "
                  f"{sorted(r['boot_s'] for r in rr)[len(rr) // 2]:.0f}s, peak RSS "
                  f"{max(r['rss_gib'] for r in rr):.1f} GiB, levels left {rr[0]['depth'] - rr[0]['out_level']}")
        for fam in ("ones", "alt", "unif"):
            rr = [r for r in recs if r["family"] == fam]
            if rr:
                print(f"B-ref family {fam}: {len(rr)} trials, max log2 err {max(r['log2_err'] for r in rr):+.3f}")
        m = max(recs, key=lambda r: r["log2_err"])
        print(f"B-ref MAX over all {len(recs)} trials: log2 err {m['log2_err']:+.3f} "
              f"(ks {m['ks']}, trial {m['trial']}, {m['family']})")
    if others:
        print(f"B-ref note: jsonl also holds other fingerprints {sorted(others)} (ignored)")
    print("SUMMARY END")
    print("=" * 90)


def main():
    a = sys.argv[1:]
    if a[:1] == ["--worker"]:
        worker(int(a[1]), dry=(a[2:3] == ["dry"]))
        return
    if a[:1] == ["--report"]:
        report()
        return
    if a[:1] == ["--dry"]:
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", "0", "dry"])
        sys.exit(r.returncode)
    fp = fingerprint()
    for ks in range(KEYSETS):
        have = {r["trial"] for r in load() if r["fp"] == fp and r["ks"] == ks and not r.get("dry")}
        if len(have) >= TRIALS:
            print(f"[parent] key set {ks}: complete, skipped")
            continue
        if have:
            sys.exit(f"[parent] key set {ks} is partial ({len(have)}/{TRIALS} trials); delete its "
                     f"lines with fp {fp} and ks {ks} from {OUT}, then rerun")
        print(f"[parent] key set {ks}: starting subprocess", flush=True)
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", str(ks)])
        if r.returncode != 0:
            sys.exit(f"[parent] key set {ks} failed (exit {r.returncode}); see its output above")
    report()


if __name__ == "__main__":
    main()
