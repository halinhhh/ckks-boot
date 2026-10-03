#!/usr/bin/env python3
"""
inventory_ledger.py  --  Step 1 (1 Oct 2026): rebuild tab:inventory, add what the construction now
has, and recompute the range rows of the ledger.

WHAT IS COUNTED (the rules of the reference audit's tab:inventory, recovered from its text)
  Chain at T towers: q0 (lq0 bits) and T-1 chain primes (lD bits).
  (A.2) key-switch digits: a key switch at T towers has m(T) = ceil(lq0/20) + (T-1) ceil(lD/20)
        digits (gadget 2^20); each digit is one ring element and one chunk (bound 2^21, table 2^21).
  (A.3) a rescale from T towers commits, per ciphertext component (2), a quotient of
        lq0 + (T-2) lD bits and a remainder below 3p: ceil(bits/20) chunks each.
  ModRaise: two range checks per component at b bits (q0 < 2^b): 2 ceil(b/20) chunks.
  Tower positions are those of the port (sweep_degb_delta_arcsine.py): client levels at
  dc+1..2, KS_ds at 2, [SubSum at top+1], CtS at top..top-4, conjugation at em_top = top-5,
  EvalMod from em_top down, per-stream extraction at stc_top+1, KS_sd at stc_top, StC at
  stc_top..stc_top-3, with stc_top = 1 + dc + 4.

TWO WAYS TO COUNT EvalMod (per stream; there are two streams)
  'level'  the reference audit's table: one key switch and one rescale per EvalMod level.
  'port'   the circuit the port audits: base tier (18,1) = 17 ciphertext products (T_2..T_18), each
           with its key switch and rescale; each T_j brought to the common level by 5 - depth(j)
           rescales (the port charges one c_rs per level of difference); one rescale of the
           plaintext combination; 6 double-angle squarings; arcsine tier = 2 products and 4
           rescales (y^2, y^3, and 2 level-matching rescales of y), as the port charges.
  'drop'   as 'port', but T_j is brought to the common level by dropping towers (exact reduction to
           a smaller modulus: no rescale, no witness). The port's (5 - depth(j)) c_rs charge is then
           an over-approximation of this circuit's noise, so the landscape stays a valid bound.
  The noise port charges the 'port' circuit. If the paper keeps the 'level' count, inventory and
  audit describe different circuits.

LEDGER (range rows only; one convention for every inventory, so differences are meaningful)
  Each range call on n ring elements at bound b with c chunks (padded to a power of two, gamma =
  log2 c, beta = 2^{b/c}, 2^{l*} = n N c):
     delta_call = 2/|S| [ (l + 2 nu + 2 gamma) + 2 (2^{l*} + beta) + 4 (l* + b/c) + 2 ],
  |S| = p_min^4, p_min = 2^{min(lD, lq0) - 1}, nu = log2 N.
  Calls: one digit batch (b = 21; c = 1 as the text states, and c = 2 padded, the reference audit's charge),
  one quotient batch per rescale tower, one remainder batch, the ModRaise call.
  decomposition row = sum over calls of 2 (l + 2 nu + 2 gamma)/|S|; circuit row = M_ns N/|S| with the
  recorded M_ns = 24. The COMMITMENT row (2^-171.43, the binding one) is NOT recomputed: its formula
  (mu, M) is not used here; it is carried as recorded, at the reference cell only.

GATES (the script stops on any FAIL)
  I0  'level' count at the reference cell reproduces tab:inventory exactly: every block, 5157 /
      2724 / 7881, quotients 2508, remainders 216, 5151 digit elements, 24 rescale towers against
      24 chain primes, l* = 29.33, and the shares 65.4% / 60.3% / 38.2%.
  L0  the ledger formula reproduces the recorded rows at the reference cell: ModRaise -185.96 and
      circuit -186.42 (to 0.01); rescale batches -177.10 (to 0.05); padded digit batch -175.61
      (to 0.1, the recorded digit row used a slightly larger batch); PIOP subtotal -175.17 and
      interactive total -171.33 (to 0.1).
  C0  the chain of every optimum has the log Q of tab:landscape; C1 every chain prime is consumed
      by exactly the towers that carry rescales (no unbudgeted rescale, no idle prime).
      The tolerances are stated because the original ledger script is not available; the
      conventions above are the ones that reproduce the rows best.
  Pure python. The summary block is printed between SUMMARY BEGIN and SUMMARY END.
"""
import math
import sys

N = 2 ** 17
NU = 17
W = 20
DIGIT_BITS = 21
DC = 3
CTS_ROT = [12, 8, 8, 8, 8]
STC_ROT = [8, 12, 12, 18]
DB = 18
ELL = 6
COMMIT_ROW = -171.43          # recorded, reference cell only (not recomputed)
M_NS = 24                     # recorded circuit-row constant
FAILS = []


def gate(name, ok, detail=""):
    print(f"    [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILS.append(name)


def depth(j):
    return 0 if j == 1 else math.ceil(math.log2(j))


def pow2(c):
    return 1 if c <= 1 else 1 << math.ceil(math.log2(c))


def l2sum(xs):
    return math.log2(sum(2.0 ** x for x in xs))


class Cell:
    def __init__(self, lD, lq0, arc, ns, subsum=True, dc=DC, label=""):
        self.lD, self.lq0, self.arc, self.ns, self.dc, self.label = lD, lq0, arc, ns, dc, label
        self.dem = 12 + (2 if arc else 0)
        self.top = 1 + dc + 4 + self.dem + len(CTS_ROT)
        self.sparse = ns < N // 2
        self.ss = subsum and self.sparse
        self.em_top = self.top - len(CTS_ROT)
        self.stc_top = 1 + dc + 4

    def m(self, T):
        return math.ceil(self.lq0 / W) + (T - 1) * math.ceil(self.lD / W)

    def qbits(self, T):                      # quotient after a rescale from T towers
        return self.lq0 + (T - 2) * self.lD

    def rbits(self):
        return math.log2(3) + self.lD


def events(c, mode, ext):
    """list of (block, kind, towers); kind in {'ks','rs'}."""
    ev = []
    for i in range(c.dc):                                        # client circuit
        T = c.dc + 1 - i
        ev += [("client circuit", "ks", T), ("client circuit", "rs", T)]
    ev.append(("KS_ds", "ks", 2))
    if c.ss:                                                      # projection level (SubSum)
        k = (N // 2) // c.ns
        ev += [("projection level", "ks", c.top + 1)] * int(round(math.log2(k)))
        ev.append(("projection level", "rs", c.top + 1))
    for i, rot in enumerate(CTS_ROT):                             # CtS
        T = c.top - i
        ev += [("CtS", "ks", T)] * rot + [("CtS", "rs", T)]
    ev.append(("conjugation (CtS)", "ks", c.em_top))
    em = []
    if mode == "level":
        for L in range(c.dem):
            T = c.em_top - L
            em += [("EvalMod", "ks", T), ("EvalMod", "rs", T)]
    else:
        for j in range(2, DB + 1):                                # base-tier products
            T = c.em_top - (depth(j) - 1)
            em += [("EvalMod", "ks", T), ("EvalMod", "rs", T)]
        if mode == "port":                                        # level matching by rescale
            for j in range(1, DB + 1):
                d = depth(j)
                for a in range(5 - d):
                    em.append(("EvalMod", "rs", c.em_top - d - a))
        # mode == "drop": level matching by dropping towers (no witness, no rescale)
        em.append(("EvalMod", "rs", c.em_top - 5))                # plaintext combination
        for idx in range(ELL):                                    # double-angle tier
            T = c.em_top - 6 - idx
            em += [("EvalMod", "ks", T), ("EvalMod", "rs", T)]
        if c.arc:                                                 # arcsine tier
            T = c.em_top - 12
            em += [("EvalMod", "ks", T), ("EvalMod", "rs", T), ("EvalMod", "ks", T - 1),
                   ("EvalMod", "rs", T - 1), ("EvalMod", "rs", T), ("EvalMod", "rs", T - 1)]
    ev += em * 2                                                  # two streams
    if ext:
        ev += [("conjugation (per stream)", "ks", c.stc_top + 1)] * 2
    ev.append(("KS_sd", "ks", c.stc_top))
    for i, rot in enumerate(STC_ROT):                             # StC (full-packing counts)
        T = c.stc_top - i
        ev += [("StC", "ks", T)] * rot + [("StC", "rs", T)]
    return ev


BLOCKS = ["client circuit", "KS_ds", "ModRaise wiring", "projection level", "CtS", "conjugation (CtS)",
          "EvalMod", "conjugation (per stream)", "KS_sd", "StC"]


def inventory(c, mode, ext):
    ev = events(c, mode, ext)
    rows = {b: [0, 0] for b in BLOCKS}
    qchunks = rchunks = 0
    n_digit = 0
    qbatch = {}                  # tower -> ring elements
    n_rs = 0
    for blk, kind, T in ev:
        if kind == "ks":
            rows[blk][0] += c.m(T)
            n_digit += c.m(T)
        else:
            qc = 2 * math.ceil(c.qbits(T) / W)
            rc = 2 * math.ceil(c.rbits() / W)
            rows[blk][1] += qc + rc
            qchunks += qc
            rchunks += rc
            qbatch[T] = qbatch.get(T, 0) + 2
            n_rs += 1
    rows["ModRaise wiring"][0] += 2 * math.ceil(c.lq0 / W)
    a2 = sum(r[0] for r in rows.values())
    a3 = sum(r[1] for r in rows.values())
    tot = a2 + a3
    trans = sum(sum(rows[b]) for b in ("CtS", "StC"))
    evalm = sum(sum(rows[b]) for b in ("EvalMod", "conjugation (CtS)", "conjugation (per stream)"))
    n_primes = c.top - 1 + (1 if c.ss else 0)
    return dict(rows=rows, a2=a2, a3=a3, tot=tot, q=qchunks, r=rchunks, n_digit=n_digit, qbatch=qbatch,
                n_rs=n_rs, n_tow=len(qbatch), n_primes=n_primes, share_ks=n_digit / tot, share_tr=trans / tot,
                share_em=evalm / tot, lstar=math.log2(n_digit) + NU,
                logQ=c.lq0 + n_primes * c.lD)


def call(n, b, cpad, logS):
    g = math.log2(cpad)
    ell = math.log2(n)
    ls = ell + NU + g
    beta = 2.0 ** (b / cpad)
    num = (ell + 2 * NU + 2 * g) + 2 * (2.0 ** ls + beta) + 4 * (ls + b / cpad) + 2
    return math.log2(2 * num) - logS, 2 * (ell + 2 * NU + 2 * g), ls


def ledger(c, inv):
    logS = 4 * (min(c.lD, c.lq0) - 1)
    rows, decomp = {}, []
    v, d_, ls1 = call(inv["n_digit"], DIGIT_BITS, 1, logS)
    rows["digit batch (c=1)"] = v
    decomp.append(d_)
    v2, _, ls2 = call(inv["n_digit"], DIGIT_BITS, 2, logS)
    rows["digit batch (c=2, padded)"] = v2
    rs = []
    for T, n in inv["qbatch"].items():
        b = c.qbits(T)
        v, d_, _ = call(n, b, pow2(math.ceil(b / W)), logS)
        rs.append(v)
        decomp.append(d_)
    rb = c.rbits()
    v, d_, _ = call(2 * inv["n_rs"], rb, pow2(math.ceil(rb / W)), logS)
    rs.append(v)
    decomp.append(d_)
    rows["rescale batches"] = l2sum(rs)
    v, d_, _ = call(2, c.lq0, pow2(math.ceil(c.lq0 / W)), logS)
    rows["ModRaise"] = v
    decomp.append(d_)
    rows["decomposition"] = math.log2(sum(decomp)) - logS
    rows["circuit"] = math.log2(M_NS * N) - logS
    piop = l2sum([rows["digit batch (c=2, padded)"], rows["rescale batches"], rows["ModRaise"],
                  rows["decomposition"], rows["circuit"]])
    piop_c1 = l2sum([rows["digit batch (c=1)"], rows["rescale batches"], rows["ModRaise"],
                     rows["decomposition"], rows["circuit"]])
    return dict(rows=rows, piop=piop, piop_c1=piop_c1, logS=logS, ls1=ls1, ls2=ls2,
                calls=2 + len(inv["qbatch"]) + 1)


def show_inv(title, inv):
    print(f"\n  {title}")
    print(f"    {'block':<26} {'(A.2)':>7} {'(A.3)':>7} {'total':>7}")
    for b in BLOCKS:
        r = inv["rows"][b]
        if r[0] or r[1]:
            print(f"    {b:<26} {r[0]:>7} {r[1]:>7} {r[0] + r[1]:>7}")
    print(f"    {'total':<26} {inv['a2']:>7} {inv['a3']:>7} {inv['tot']:>7}")
    print(f"    quotients {inv['q']}, remainders {inv['r']}; digit elements {inv['n_digit']} "
          f"(l* = {inv['lstar']:.2f}); rescale operations {inv['n_rs']} on {inv['n_tow']} towers "
          f"(chain primes {inv['n_primes']}); log Q {inv['logQ']}")
    print(f"    shares: key-switch digits {100 * inv['share_ks']:.1f}%, transforms {100 * inv['share_tr']:.1f}%, "
          f"EvalMod incl. conjugations {100 * inv['share_em']:.1f}%")


def show_led(title, c, led, with_commit):
    print(f"    ledger {title}: |S| = 2^{led['logS']}, {led['calls']} range calls")
    for k, v in led["rows"].items():
        print(f"      {k:<26} 2^{v:8.2f}")
    print(f"      {'PIOP subtotal (padded)':<26} 2^{led['piop']:8.2f}    (c=1 digit: 2^{led['piop_c1']:.2f})")
    if with_commit:
        tot = l2sum([led["piop"], COMMIT_ROW])
        print(f"      {'commitment (recorded)':<26} 2^{COMMIT_ROW:8.2f}")
        print(f"      {'interactive total':<26} 2^{tot:8.2f}")
        return tot
    return None


print("=" * 100)
print("inventory_ledger.py   Step 1: tab:inventory rebuilt, conjugations and projection level added, ledger rows")
print("=" * 100)

# ---------------------------------------------------------------- PART 0: gates
print("\nPART 0  GATES at the reference cell (Delta = 2^53, 60-bit q0, full packing, arcsine off)")
REF = Cell(53, 60, 0, N // 2, label="ref")
I0 = inventory(REF, "level", ext=False)
want = {"client circuit": (27, 54), "KS_ds": (6, 0), "ModRaise wiring": (6, 0), "CtS": (3060, 620),
        "conjugation (CtS)": (60, 0), "EvalMod": (1044, 1904), "KS_sd": (24, 0), "StC": (930, 146)}
bad = [b for b, v in want.items() if tuple(I0["rows"][b]) != v]
gate("I0 every block of tab:inventory", not bad, f"mismatch: {[(b, I0['rows'][b], want[b]) for b in bad]}" if bad else "")
gate("I0 totals 5157 / 2724 / 7881", (I0["a2"], I0["a3"], I0["tot"]) == (5157, 2724, 7881),
     f"{I0['a2']} / {I0['a3']} / {I0['tot']}")
gate("I0 quotients 2508, remainders 216", (I0["q"], I0["r"]) == (2508, 216), f"{I0['q']}, {I0['r']}")
gate("I0 digit batch 5151 elements, l* = 29.33", I0["n_digit"] == 5151 and abs(I0["lstar"] - 29.33) < 0.005,
     f"{I0['n_digit']}, {I0['lstar']:.3f}")
gate("I0 24 rescales on 24 towers = 24 chain primes, log Q 1332",
     I0["n_rs"] == 24 + 12 and I0["n_tow"] == 24 and I0["n_primes"] == 24 and I0["logQ"] == 1332,
     f"ops {I0['n_rs']} (EvalMod doubled), towers {I0['n_tow']}, primes {I0['n_primes']}, log Q {I0['logQ']}")
sh = (round(100 * I0["share_ks"], 1), round(100 * I0["share_tr"], 1), round(100 * I0["share_em"], 1))
gate("I0 shares 65.4% / 60.3% / 38.2%", sh == (65.4, 60.3, 38.2), f"{sh}")
L0 = ledger(REF, I0)
r = L0["rows"]
tot0 = l2sum([L0["piop"], COMMIT_ROW])
gate("L0 ModRaise -185.96, circuit -186.42", abs(r["ModRaise"] + 185.96) < 0.01 and abs(r["circuit"] + 186.42) < 0.01,
     f"{r['ModRaise']:.3f}, {r['circuit']:.3f}")
gate("L0 rescale batches -177.10 (0.05)", abs(r["rescale batches"] + 177.10) < 0.05, f"{r['rescale batches']:.3f}")
gate("L0 padded digit batch -175.61 (0.1)", abs(r["digit batch (c=2, padded)"] + 175.61) < 0.1,
     f"{r['digit batch (c=2, padded)']:.3f}")
gate("L0 PIOP subtotal -175.17, interactive total -171.33 (0.1)",
     abs(L0["piop"] + 175.17) < 0.1 and abs(tot0 + 171.33) < 0.1, f"{L0['piop']:.3f}, {tot0:.3f}")
print(f"    [INFO] decomposition row with this convention 2^{r['decomposition']:.2f} (recorded 2^-196.72)")
if FAILS:
    print("\n  GATE FAILED:", FAILS, " -- see the output above.")
    sys.exit(1)
print("  GATES PASSED.")

# ---------------------------------------------------------------- PART A: level count + conjugations
print("\nPART A  reference cell, the reference audit's count + the two per-stream conjugations")
IA = inventory(REF, "level", ext=True)
show_inv("level count + per-stream conjugations", IA)
LA = ledger(REF, IA)
totA = show_led("A", REF, LA, True)

# ---------------------------------------------------------------- PART B: the circuit the port audits
print("\nPART B  reference cell, EvalMod counted as the port audits it (17 + 6 products per stream, level matching)")
IB = inventory(REF, "port", ext=True)
show_inv("port count + per-stream conjugations", IB)
LB = ledger(REF, IB)
totB = show_led("B", REF, LB, True)
print(f"    [INFO] EvalMod block: level count {sum(IA['rows']['EvalMod'])} chunks, port count "
      f"{sum(IB['rows']['EvalMod'])} chunks; key switches per stream: level {12}, port {17 + 6}")

# ---------------------------------------------------------------- PART C: the optima of tab:landscape
print("\nPART C  the optima of tab:landscape (port count, per-stream conjugations, projection level for sparse rows,")
print("        StC at full-packing rotation counts, as the port keeps the full-packing transforms)")
OPT = [Cell(58, 64, 3, 1, label="1 slot 58/6/on"), Cell(59, 64, 3, 8, label="2^3 59/5/on"),
       Cell(59, 64, 3, 256, label="2^8 59/5/on"), Cell(61, 64, 3, N // 2, label="full 61/3/on")]
want_logq = {"1 slot 58/6/on": 1630, "2^3 59/5/on": 1657, "2^8 59/5/on": 1657, "full 61/3/on": 1650}
IC = {}
for c in OPT:
    inv = inventory(c, "port", ext=True)
    IC[c.label] = (c, inv, ledger(c, inv))
    show_inv(c.label, inv)
    show_led(c.label, c, IC[c.label][2], False)
gate("C1 chain closure: every chain prime is consumed by a rescale tower (A, B, every optimum)",
     all(inv["n_tow"] == inv["n_primes"] for inv in [IA, IB] + [v[1] for v in IC.values()]),
     f"{[(inv['n_tow'], inv['n_primes']) for inv in [IA, IB] + [v[1] for v in IC.values()]]}")
gate("C0 log Q of every optimum = tab:landscape", all(IC[k][1]["logQ"] == v for k, v in want_logq.items()),
     f"{[(k, IC[k][1]['logQ']) for k in want_logq]}")
if FAILS:
    print("\n  GATE FAILED:", FAILS, " -- see the output above.")
    sys.exit(1)

# ---------------------------------------------------------------- PART D: level matching by tower drop
print("\nPART D  as PART B/C, but T_j is brought to the common level by DROPPING towers (no rescale, no")
print("        witness); the port's (5 - depth) c_rs charge then over-approximates this circuit")
ID = inventory(REF, "drop", ext=True)
show_inv("reference cell, drop count + per-stream conjugations", ID)
LD_ = ledger(REF, ID)
totD = show_led("D", REF, LD_, True)
IDC = {}
for c in OPT:
    inv = inventory(c, "drop", ext=True)
    IDC[c.label] = (c, inv, ledger(c, inv))
gate("D0 chain closure holds without the level-matching rescales (ref + every optimum)",
     all(inv["n_tow"] == inv["n_primes"] for inv in [ID] + [v[1] for v in IDC.values()]),
     f"{[(inv['n_tow'], inv['n_primes']) for inv in [ID] + [v[1] for v in IDC.values()]]}")
for k, (c, inv, led) in IDC.items():
    print(f"    {k:<16} total {inv['a2']}/{inv['a3']}/{inv['tot']}  digits {inv['n_digit']}  rs_ops {inv['n_rs']}  "
          f"PIOP 2^{led['piop']:.2f}")

# ---------------------------------------------------------------- summary block
print("\n" + "=" * 100)
print("SUMMARY BEGIN")
for lab, inv in (("0 level", I0), ("A level+cj", IA), ("B port+cj", IB)):
    rr = inv["rows"]
    print(f"INV {lab}: " + " ".join(f"{b.split()[0]}={rr[b][0]}/{rr[b][1]}" for b in BLOCKS if rr[b][0] or rr[b][1])
          + f" | total {inv['a2']}/{inv['a3']}/{inv['tot']} q={inv['q']} r={inv['r']} digits={inv['n_digit']} "
          f"l*={inv['lstar']:.2f} rs_ops={inv['n_rs']} towers={inv['n_tow']} | shares "
          f"{100 * inv['share_ks']:.1f}/{100 * inv['share_tr']:.1f}/{100 * inv['share_em']:.1f}")
rr = ID["rows"]
print(f"INV D drop+cj: " + " ".join(f"{b.split()[0]}={rr[b][0]}/{rr[b][1]}" for b in BLOCKS if rr[b][0] or rr[b][1])
      + f" | total {ID['a2']}/{ID['a3']}/{ID['tot']} q={ID['q']} r={ID['r']} digits={ID['n_digit']} "
      f"l*={ID['lstar']:.2f} rs_ops={ID['n_rs']} towers={ID['n_tow']} | shares "
      f"{100 * ID['share_ks']:.1f}/{100 * ID['share_tr']:.1f}/{100 * ID['share_em']:.1f}")
for lab, led, tot in (("0", L0, tot0), ("A", LA, totA), ("B", LB, totB), ("D", LD_, totD)):
    print(f"LED {lab}: " + " ".join(f"{k.split()[0]}{'_c1' if 'c=1' in k else ('_c2' if 'c=2' in k else '')}="
                                    f"{v:.2f}" for k, v in led["rows"].items())
          + f" | PIOP {led['piop']:.2f} (c1 {led['piop_c1']:.2f}) | total {tot:.2f} (commitment {COMMIT_ROW} recorded)")
for k, (c, inv, led) in IC.items():
    print(f"OPT {k}: total {inv['a2']}/{inv['a3']}/{inv['tot']} digits={inv['n_digit']} l*={inv['lstar']:.2f} "
          f"(slack to p_min {min(c.lD, c.lq0) - 1 - inv['lstar']:.1f} bits) rs_ops={inv['n_rs']} towers={inv['n_tow']} "
          f"logQ={inv['logQ']} | PIOP {led['piop']:.2f} at |S|=2^{led['logS']} | proj.level "
          f"{inv['rows']['projection level'][0]}/{inv['rows']['projection level'][1]}")
for k, (c, inv, led) in IDC.items():
    print(f"OPTD {k}: total {inv['a2']}/{inv['a3']}/{inv['tot']} digits={inv['n_digit']} l*={inv['lstar']:.2f} "
          f"rs_ops={inv['n_rs']} | PIOP {led['piop']:.2f} at |S|=2^{led['logS']}")
print("GATE FAILURES:", FAILS if FAILS else "none")
print("SUMMARY END")
print("=" * 100)
