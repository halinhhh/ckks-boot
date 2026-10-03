#!/usr/bin/env python3
"""
sweep_partO.py  --  PART O (1 Oct 2026): certify the residue of the encoded CtS diagonals
(option A) and re-run the landscape with it.

WHY
  PART N showed that the residue term R (nu + B) / p of the encoded CtS diagonals is not small,
  because nu_0 = N q0 (1 + H)/2 is the only certified bound on the lifted phase. With the paper's
  check ||a~ - S f||_1 <= R (R = N/2) the landscape loses 6-7 bits; with a random-residue MODEL of
  the canonical norm it loses 1.3-2.5 bits. Option (A): the audit computes the canonical norm of
  the residue of the ACTUAL encoded diagonals. PART O computes that number.

WHAT IT BUILDS
  1. The CtS factorisation. V0[j,k] = zeta^{g_j k}, g_j = 5^j mod 2N (HEAAN fftSpecial; gate O0a).
     V0^{-1} = BR T_2 ... T_n / n, where T_L is the inverse butterfly stage of length L
     (application order T_n first). The stages are merged into the levels CTS_KS = [4,3,3,3,3]
     in application order (largest strides first). BR is not encoded: EvalMod is slot-wise and
     StC starts with the same permutation. 1/n and the conjugation 1/2 are inside S.
     The first level has 16 diagonals, not 2^5 - 1 = 31: its strides n/2..n/16 wrap mod n
     (offsets +n/2 and -n/2 coincide). The port's rotation counts CTS_ROT are kept (conservative).
  2. The encoded constants. Level i, diagonal d: the plaintext a~ = round(p_i s1 phi^{-1}(diag_d)),
     where s1 = S^{1/5} (spread fold, PART M iv-s), p_i = the prime dropped after level i (the five
     largest NTT-friendly primes p = 1 mod 2N below 2^lD, largest first), and phi^{-1} = the exact
     inverse canonical embedding. phi^{-1} is computed in double-double (~2^-104) arithmetic with
     twiddles from 45-digit decimal values, so the residue r = a~ - p_i s1 phi^{-1}(diag_d) is known
     to ~2^-40 per coefficient. Then ||sigma(r)||_inf = max_j |sum_k r_k zeta^{g_j k}| by one FFT.
  3. R_level(i) = sum_d ( ||sigma(r_d)||_inf + MARGIN ) * 2^lD / p_i. The sum of per-diagonal sup
     norms is valid for any baby-step/giant-step alignment: a pre-rotated diagonal is the image of
     the coefficient vector under X -> X^{5^g}, a signed permutation, so its rounding residue is the
     permuted residue and its canonical sup norm is the same. MARGIN = 2^-8 covers the double-double
     error of m (<= N * 2^-30 in canonical norm) and the FFT error (<= 2^-20); both are a priori
     bounds, not interval arithmetic (the audit would use outward rounding).
  4. The level tag:  B' = 2^k s1 B + c_rs + rot * bks / Delta + R_level(i) (nu + B) / p  (PART N,
     with R_level in place of d * R_diag). StC residues use the paper's ||.||_1 check, R = N/2 per
     diagonal, which the paper already certifies: level i of part_of(n_s) with d_i = 2^{k_i+1} - 1
     diagonals adds d_i (N/2) (nu_i + tau) / p, nu_i = sqrt2 (Delta + B_in) 2^{sum k_<i}, amplified
     by 2^{sum k_>i} (tau, the total, bounds the error tag at every StC level).
  The scalar residues of c_j, A^2 and a (R = 1/2) are as in PART N.

GATES (any FAIL stops the run)
  O0a  fft_special(u) = V0 u (N = 2^4..2^10); sigma(t) = V0 (t_lo + i t_hi) for real t (N = 2^8);
       the n-point formula sigma_j = n ifft(u zeta^k)[(g_j - 1)/4] gives the same values.
  O0b  double-double: twiddle table |E|^2 = 1 and E[a] E[b] = E[a+b] (2000 pairs, 1e-30);
       dd butterfly inverse = dd direct DFT (N = 2^8); dd inverse then dd forward = identity at
       N = 2^17 on two real CtS diagonals (relative 1e-26).
  O0c  merged-level diagonals: count <= 2^{k+1} - 1 per level; levels by diagonals = stages (float);
       BR (levels) / n = V0^{-1} densely at N = 2^10 (levels [3,3,3]) and on random vectors at 2^17.
  O0d  every rounding residue |r_k| <= 1/2 + 1e-9 (checked on every coefficient of every encoding).
  O0e  with all residues off, model_O / eval_KO reproduce model() (12 cells) and eval_K() (45 cells).
  O0f  with R_level = d * R_can (PART N's MODEL), spread fold and scalar residues, eval_KO
       reproduces the recorded PART N v2 spread+can AND spread+l1 decs at the 5 headline cells (0.0015).

USAGE   python3 /path/to/sweep_partO.py   (finds the sweep in its own folder, as PART M/N do)
        Runtime on a 2-core machine: sweep ~1.8 min, certification ~2.7 min, rest ~0.7 min.
        Memory: < 1 GB.
        The summary block is printed between SUMMARY BEGIN and SUMMARY END.
"""
import contextlib
import decimal
import glob
import io
import math
import os
import runpy
import sys
import time

import numpy as np

T0 = time.time()
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
# double-double arithmetic (numpy, elementwise)
# =============================================================================================
SPL = 134217729.0          # 2^27 + 1


def split(a):
    t = SPL * a
    hi = t - (t - a)
    return hi, a - hi


def two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def qts(a, b):
    s = a + b
    return s, b - (s - a)


def two_prod(a, b):
    p = a * b
    ah, al = split(a)
    bh, bl = split(b)
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


def dd_add(x, y):
    s, e = two_sum(x[0], y[0])
    t, f = two_sum(x[1], y[1])
    e = e + t
    s, e = qts(s, e)
    e = e + f
    return qts(s, e)


def dd_neg(x):
    return (-x[0], -x[1])


def dd_mul(x, y):
    p, e = two_prod(x[0], y[0])
    e = e + (x[0] * y[1] + x[1] * y[0])
    return qts(p, e)


# complex dd: ((re_hi, re_lo), (im_hi, im_lo))
def c_add(x, y):
    return (dd_add(x[0], y[0]), dd_add(x[1], y[1]))


def c_sub(x, y):
    return (dd_add(x[0], dd_neg(y[0])), dd_add(x[1], dd_neg(y[1])))


def c_mul(x, y):
    return (dd_add(dd_mul(x[0], y[0]), dd_neg(dd_mul(x[1], y[1]))),
            dd_add(dd_mul(x[0], y[1]), dd_mul(x[1], y[0])))


def c_conj(x):
    return (x[0], dd_neg(x[1]))


def c_map(f, x):
    return ((f(x[0][0]), f(x[0][1])), (f(x[1][0]), f(x[1][1])))


def c_cat(xs, axis):
    return tuple(tuple(np.concatenate([x[a][b] for x in xs], axis=axis) for b in (0, 1)) for a in (0, 1))


def c_zero(n):
    z = np.zeros(n)
    return ((z, z.copy()), (z.copy(), z.copy()))


def c_float(x):
    return (x[0][0] + x[0][1]) + 1j * (x[1][0] + x[1][1])


def c_absdiff(x, y):
    d = c_sub(x, y)
    return np.abs(c_float(d))


def c_scale_pow2(x, s):
    return c_map(lambda a: a * s, x)


# =============================================================================================
# twiddles: E[a] = exp(2 pi i a / M), a < M, in dd, from 45-digit decimal base tables
# =============================================================================================
PI_STR = "3.14159265358979323846264338327950288419716939937510582097494459230781640628620899"


def _dec_cos_sin(num, den, ctx):
    pi = decimal.Decimal(PI_STR)
    x = ctx.divide(ctx.multiply(2 * pi, num), den)
    if x > pi:
        x = x - 2 * pi
    c, s = decimal.Decimal(0), decimal.Decimal(0)
    term = decimal.Decimal(1)
    k = 0
    eps = decimal.Decimal(10) ** (-(ctx.prec + 2))
    while True:              # term = x^k / k!
        if k % 4 == 0:
            c += term
        elif k % 4 == 1:
            s += term
        elif k % 4 == 2:
            c -= term
        else:
            s -= term
        k += 1
        term = term * x / k
        if abs(term) < eps and k > 4:
            break
    return c, s


def _dec_to_dd(d):
    hi = float(d)
    lo = float(d - decimal.Decimal(hi))
    return hi, lo


def twiddle_table(M):
    L = M.bit_length() - 1
    B = 1 << ((L + 1) // 2)
    A = M // B
    ctx = decimal.Context(prec=45)
    with decimal.localcontext(ctx):
        def tab(nums):
            rh, rl, ih, il = [], [], [], []
            for a in nums:
                c, s = _dec_cos_sin(a, M, ctx)
                h, l_ = _dec_to_dd(c)
                rh.append(h)
                rl.append(l_)
                h, l_ = _dec_to_dd(s)
                ih.append(h)
                il.append(l_)
            return ((np.array(rh), np.array(rl)), (np.array(ih), np.array(il)))
        fine = tab(range(B))
        coarse = tab([a * B for a in range(A)])
    idx = np.arange(M)
    E = c_mul(c_map(lambda t: t[idx // B], coarse), c_map(lambda t: t[idx % B], fine))
    return E


# =============================================================================================
# special FFT (HEAAN), V0, merged CtS levels
# =============================================================================================
def gens(N):
    n = N // 2
    g = np.empty(n, dtype=np.int64)
    x = 1
    for j in range(n):
        g[j] = x
        x = (x * 5) % (2 * N)
    return g


def bitrev_idx(n):
    b = n.bit_length() - 1
    i = np.arange(n)
    r = np.zeros(n, dtype=np.int64)
    for k in range(b):
        r |= ((i >> k) & 1) << (b - 1 - k)
    return r


def V0_dense(N):
    n = N // 2
    g = gens(N)
    k = np.arange(n)
    return np.exp(1j * np.pi * ((np.outer(g, k)) % (2 * N)) / N)


def stage_w(N, L, G):
    """float twiddles of the forward stage of length L: exp(2 pi i (5^j mod 4L) / 4L), j < L/2."""
    h = L // 2
    return np.exp(2j * np.pi * (G[:h] % (4 * L)) / (4 * L))


def fft_special(u, N, G, BR):
    n = N // 2
    v = u[BR].astype(complex)
    L = 2
    while L <= n:
        h = L // 2
        w = stage_w(N, L, G)
        v = v.reshape(-1, L)
        a = v[:, :h].copy()
        b = v[:, h:] * w
        v = np.concatenate([a + b, a - b], axis=1).reshape(-1)
        L *= 2
    return v


def dd_w(E, N, L, G, conj):
    h = L // 2
    idx = (G[:h] % (4 * L)) * (2 * N // (4 * L))
    w = c_map(lambda t: t[idx], E)
    return c_conj(w) if conj else w


def inv_special_dd(z, N, E, G, BR):
    """u = V0^{-1} z in dd (z: complex dd of length n)."""
    n = N // 2
    v = z
    L = n
    while L >= 2:
        h = L // 2
        w = dd_w(E, N, L, G, conj=True)
        v2 = c_map(lambda t: t.reshape(-1, L), v)
        a = c_map(lambda t: t[:, :h], v2)
        b = c_map(lambda t: t[:, h:], v2)
        na = c_add(a, b)
        nb = c_mul(c_sub(a, b), c_map(lambda t: t[None, :], w))
        v = c_map(lambda t: t.reshape(-1), c_cat([na, nb], axis=1))
        L //= 2
    v = c_map(lambda t: t[BR], v)
    return c_scale_pow2(v, 1.0 / n)


def fwd_special_dd(u, N, E, G, BR):
    n = N // 2
    v = c_map(lambda t: t[BR], u)
    L = 2
    while L <= n:
        h = L // 2
        w = dd_w(E, N, L, G, conj=False)
        v2 = c_map(lambda t: t.reshape(-1, L), v)
        a = c_map(lambda t: t[:, :h], v2)
        b = c_mul(c_map(lambda t: t[:, h:], v2), c_map(lambda t: t[None, :], w))
        v = c_map(lambda t: t.reshape(-1), c_cat([c_add(a, b), c_sub(a, b)], axis=1))
        L *= 2
    return v


def stage_diags_dd(N, L, E, G):
    """inverse stage T_L as {offset mod n: complex dd diagonal}: out[t] = sum_d D_d[t] in[t + d]."""
    n = N // 2
    h = L // 2
    t = np.arange(n)
    pos = t % L
    low = pos < h
    jj = np.where(low, 0, pos - h)
    wc = c_conj(c_map(lambda a: a[jj], dd_w(E, N, L, G, conj=False)))
    one = ((np.ones(n), np.zeros(n)), (np.zeros(n), np.zeros(n)))
    mask = lambda x, m: c_map(lambda a: np.where(m, a, 0.0), x)
    D = {}

    def put(off, val):
        off %= n
        D[off] = c_add(D[off], val) if off in D else val
    put(0, c_add(mask(one, low), mask(c_map(lambda a: -a, wc), ~low)))
    put(h, mask(one, low))
    put(-h, mask(wc, ~low))
    return D


def compose_dd(A, Bd, n):
    """diagonals of A o B (B applied first): C_{a+b}[t] += A_a[t] B_b[t + a]."""
    C = {}
    for a, Da in A.items():
        for b, Db in Bd.items():
            prod = c_mul(Da, c_map(lambda x: np.roll(x, -a), Db))
            off = (a + b) % n
            C[off] = c_add(C[off], prod) if off in C else prod
    return C


def level_lengths(N, ks):
    n = N // 2
    lens = [n >> i for i in range(n.bit_length() - 1)]       # n, n/2, ..., 2 (application order)
    assert sum(ks) == len(lens)
    out, i = [], 0
    for k in ks:
        out.append(lens[i:i + k])
        i += k
    return out


def build_levels_dd(N, ks, E, G):
    n = N // 2
    levels = []
    for lens in level_lengths(N, ks):
        M = None
        for L in lens:
            S_ = stage_diags_dd(N, L, E, G)
            M = S_ if M is None else compose_dd(S_, M, n)
        levels.append(M)
    return levels


def apply_diags(D, x):
    y = np.zeros_like(x, dtype=complex)
    for off, d in D.items():
        y += c_float(d) * np.roll(x, -off)
    return y


def apply_stage_inv_float(x, N, L, G):
    h = L // 2
    w = np.conj(stage_w(N, L, G))
    v = x.reshape(-1, L)
    a, b = v[:, :h], v[:, h:]
    return np.concatenate([a + b, (a - b) * w], axis=1).reshape(-1)


# =============================================================================================
# primes and residue certification
# =============================================================================================
def is_prime(m):
    if m < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if m % p == 0:
            return m == p
    d, s = m - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, m)
        if x in (1, m - 1):
            continue
        for _ in range(s - 1):
            x = x * x % m
            if x == m - 1:
                break
        else:
            return False
    return True


_PR = {}


def ntt_primes(lD, N, count):
    key = (lD, N)
    if key not in _PR or len(_PR[key]) < count:
        out, step = [], 2 * N
        c = ((2 ** lD - 1) // step) * step + 1
        if c >= 2 ** lD:
            c -= step
        while len(out) < count:
            if is_prime(c):
                out.append(c)
            c -= step
        _PR[key] = out
    return _PR[key][:count]


MARGIN = 2.0 ** -8


class Certifier:
    """Accumulates, over the diagonals of every level, sum_d (||sigma(r_d)||_inf + MARGIN) * 2^lD / p
    for a list of jobs (key -> (lD, s1, primes per level))."""

    def __init__(self, N, jobs, info_keys=()):
        self.N, self.n = N, N // 2
        self.jobs = jobs
        self.R = {k: [0.0] * len(jobs[k][2]) for k in jobs}
        self.info = {k: {} for k in info_keys}     # level -> accumulated |sigma| vector
        self.ndiag = {}
        self.max_abs_r = 0.0
        self.near_ties = 0
        self.zeta = np.exp(1j * np.pi * np.arange(self.n) / N)
        self.cs = {}
        for k, (lD, s1, primes) in jobs.items():
            cl = []
            for p in primes:
                ph = float(p)
                pl = float(p - int(ph))
                ch, cl_ = dd_mul((np.float64(ph), np.float64(pl)), (np.float64(s1), np.float64(0.0)))
                chh, chl = split(ch)
                cl.append((ch, cl_, chh, chl, 2.0 ** lD / p))
            self.cs[k] = cl

    def add_diagonal(self, level, u):
        """u = phi^{-1}-part of one diagonal: complex dd of length n (coefficients Re u, Im u)."""
        self.ndiag[level] = self.ndiag.get(level, 0) + 1
        xh = np.concatenate([u[0][0], u[1][0]])
        xl = np.concatenate([u[0][1], u[1][1]])
        xhh, xhl = split(xh)
        n = self.n
        for k, cl in self.cs.items():
            ch, cl_, chh, chl, pf = cl[level]
            p_ = ch * xh
            e = ((chh * xhh - p_) + chh * xhl + chl * xhh) + chl * xhl
            e = e + (ch * xl + cl_ * xh)
            s, e = qts(p_, e)
            a = np.rint(s)
            f = (s - a) + e
            r = np.rint(f) - f
            mr = float(np.max(np.abs(r)))
            self.max_abs_r = max(self.max_abs_r, mr)
            self.near_ties += int(np.count_nonzero(np.abs(r) > 0.5 - 1e-9))
            sig = np.abs(n * np.fft.ifft((r[:n] + 1j * r[n:]) * self.zeta))
            self.R[k][level] += (float(np.max(sig)) + MARGIN) * pf
            if k in self.info:
                acc = self.info[k]
                acc[level] = acc[level] + sig if level in acc else sig


def certify(N, ks, E, G, BR, cert, progress=True):
    n = N // 2
    for lev, lens in enumerate(level_lengths(N, ks)):
        M = None
        for L in lens:
            S_ = stage_diags_dd(N, L, E, G)
            M = S_ if M is None else compose_dd(S_, M, n)
        for off in sorted(M):
            u = inv_special_dd(M[off], N, E, G, BR)
            cert.add_diagonal(lev, u)
        if progress:
            print(f"      level {lev} (k = {ks[lev]}): {len(M)} diagonals done, {time.time() - T0:6.0f} s", flush=True)


# =============================================================================================
# PART O0 (algebra gates, independent of the sweep)
# =============================================================================================
def run_algebra_gates():
    print("\nPART O0  GATES (algebra, double-double, factorisation)")
    rng = np.random.default_rng(20261001)
    # O0a
    w1 = 0.0
    for N_ in (2 ** 4, 2 ** 6, 2 ** 8, 2 ** 10):
        n_ = N_ // 2
        G_, BR_ = gens(N_), bitrev_idx(n_)
        u = rng.standard_normal(n_) + 1j * rng.standard_normal(n_)
        A = V0_dense(N_) @ u
        w1 = max(w1, np.max(np.abs(A - fft_special(u, N_, G_, BR_))) / np.max(np.abs(A)))
    N_ = 2 ** 8
    n_ = N_ // 2
    G_ = gens(N_)
    t = rng.uniform(-0.5, 0.5, N_)
    zeta_all = np.exp(1j * np.pi * (np.outer(G_, np.arange(N_)) % (2 * N_)) / N_)
    sig_def = zeta_all @ t
    sig_v0 = V0_dense(N_) @ (t[:n_] + 1j * t[n_:])
    sig_n = n_ * np.fft.ifft((t[:n_] + 1j * t[n_:]) * np.exp(1j * np.pi * np.arange(n_) / N_))
    mprime = (G_ - 1) // 4
    w2 = max(np.max(np.abs(sig_def - sig_v0)), np.max(np.abs(sig_def - sig_n[mprime]))) / np.max(np.abs(sig_def))
    ok_perm = len(set(mprime.tolist())) == n_ and int(mprime.min()) == 0 and int(mprime.max()) == n_ - 1
    gate("O0a fft_special = V0 (N=2^4..2^10); sigma(t) = V0(t_lo + i t_hi) = n ifft(u zeta^k)[(g-1)/4] (N=2^8)",
         w1 < 1e-12 and w2 < 1e-12 and ok_perm, f"rel err {w1:.1e} / {w2:.1e}, (g-1)/4 a permutation: {ok_perm}")
    # O0b
    N_ = 2 ** 8
    n_ = N_ // 2
    E_ = twiddle_table(2 * N_)
    a_ = rng.integers(0, 2 * N_, 2000)
    b_ = rng.integers(0, 2 * N_, 2000)
    Ea, Eb, Eab = (c_map(lambda x, i=i: x[i], E_) for i in (a_, b_, (a_ + b_) % (2 * N_)))
    e1 = float(np.max(c_absdiff(c_mul(Ea, Eb), Eab)))
    mod2 = dd_add(dd_mul(E_[0], E_[0]), dd_mul(E_[1], E_[1]))
    e2 = float(np.max(np.abs((mod2[0] - 1.0) + mod2[1])))
    G_, BR_ = gens(N_), bitrev_idx(n_)
    z = ((rng.standard_normal(n_), np.zeros(n_)), (rng.standard_normal(n_), np.zeros(n_)))
    u_bf = inv_special_dd(z, N_, E_, G_, BR_)
    u_dir = c_zero(n_)
    k = np.arange(n_)
    for j in range(n_):
        e_ = c_conj(c_map(lambda x: x[(int(G_[j]) * k) % (2 * N_)], E_))
        zj = c_map(lambda x: np.full(n_, x[j]), z)
        u_dir = c_add(u_dir, c_mul(e_, zj))
    u_dir = c_scale_pow2(u_dir, 1.0 / n_)
    e3 = float(np.max(c_absdiff(u_bf, u_dir)) / np.max(np.abs(c_float(u_dir))))
    gate("O0b dd twiddles: |E E' - E''| and ||E|^2 - 1| (N = 2^8); dd butterfly inverse = dd direct DFT",
         e1 < 1e-30 and e2 < 1e-30 and e3 < 1e-28, f"{e1:.1e} / {e2:.1e} / {e3:.1e}")
    # O0c small N dense
    N_ = 2 ** 10
    n_ = N_ // 2
    ks_ = [3, 3, 3]
    E_ = twiddle_table(2 * N_)
    G_, BR_ = gens(N_), bitrev_idx(n_)
    lev_ = build_levels_dd(N_, ks_, E_, G_)
    cnt_ok = all(len(D) <= 2 ** (k_ + 1) - 1 for D, k_ in zip(lev_, ks_))
    x = rng.standard_normal(n_) + 1j * rng.standard_normal(n_)
    y1, y2 = x.copy(), x.copy()
    for D, lens in zip(lev_, level_lengths(N_, ks_)):
        y1 = apply_diags(D, y1)
        for L in lens:
            y2 = apply_stage_inv_float(y2, N_, L, G_)
    e4 = np.max(np.abs(y1 - y2)) / np.max(np.abs(y2))
    Cm = np.zeros((n_, n_), dtype=complex)
    for c in range(n_):
        ec = np.zeros(n_, dtype=complex)
        ec[c] = 1.0
        y = ec
        for D in lev_:
            y = apply_diags(D, y)
        Cm[:, c] = y[BR_] / n_
    e5 = np.max(np.abs(Cm - np.linalg.inv(V0_dense(N_))))
    gate("O0c (N=2^10, levels [3,3,3]) diag counts <= 2^(k+1)-1; levels = stages; BR(levels)/n = V0^-1 densely",
         cnt_ok and e4 < 1e-12 and e5 < 1e-12,
         f"counts {[len(D) for D in lev_]}, {e4:.1e}, {e5:.1e}")
    return rng


def run_big_gates(N, ks, E, G, BR, rng):
    n = N // 2
    lev = build_levels_dd(N, ks, E, G)
    counts = [len(D) for D in lev]
    cnt_ok = all(c <= 2 ** (k + 1) - 1 for c, k in zip(counts, ks))
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    y1, y2 = x.copy(), x.copy()
    for D, lens in zip(lev, level_lengths(N, ks)):
        y1 = apply_diags(D, y1)
        for L in lens:
            y2 = apply_stage_inv_float(y2, N, L, G)
    e1 = np.max(np.abs(y1 - y2)) / np.max(np.abs(y2))
    back = fft_special(y1[BR] / n, N, G, BR)
    e2 = np.max(np.abs(back - x)) / np.max(np.abs(x))
    gate(f"O0c (N=2^17, levels {ks}) diag counts <= 2^(k+1)-1; levels = stages; V0 (BR(levels)/n) = id",
         cnt_ok and e1 < 1e-12 and e2 < 1e-12, f"counts {counts}, {e1:.1e}, {e2:.1e}")
    e3 = 0.0
    for lv, off in ((0, sorted(lev[0])[1]), (4, sorted(lev[4])[-1])):
        d = lev[lv][off]
        u = inv_special_dd(d, N, E, G, BR)
        z = fwd_special_dd(u, N, E, G, BR)
        e3 = max(e3, float(np.max(c_absdiff(z, d)) / np.max(np.abs(c_float(d)))))
    gate("O0b dd inverse then dd forward = identity on two CtS diagonals at N = 2^17", e3 < 1e-26, f"rel {e3:.1e}")
    return counts


# =============================================================================================
# the sweep
# =============================================================================================
def find_sweep():
    if len(sys.argv) > 1:
        return os.path.abspath(sys.argv[1])
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, "sweep_degb_delta_arcsine.py")
    if os.path.isfile(cand):
        return cand
    me = os.path.abspath(__file__)
    for f in sorted(glob.glob(os.path.join(here, "sweep*.py"))):
        if os.path.abspath(f) == me:
            continue
        try:
            txt = open(f, encoding="utf-8").read()
        except OSError:
            continue
        if "def eval_K(" in txt and "PART K" in txt:
            return f
    return None


print("=" * 100)
print("sweep_partO.py   PART O: certified canonical-norm residue of the encoded CtS diagonals (option A)")
print("=" * 100, flush=True)
RNG = run_algebra_gates()
stop_if_failed("PART O0")

SWEEP = find_sweep()
if SWEEP is None or not os.path.isfile(SWEEP):
    print("Cannot find the sweep. Put sweep_partO.py next to it, or pass its path as the first argument.")
    sys.exit(1)
_buf = io.StringIO()
try:
    with contextlib.redirect_stdout(_buf):
        G = runpy.run_path(SWEEP, run_name="sweep")
except SystemExit:
    print(_buf.getvalue())
    print("\nThe sweep stopped at one of its own gates (output above). Fix that first.")
    sys.exit(1)
if G["FAILS"]:
    print(_buf.getvalue())
    print("\nThe sweep reports gate failures:", G["FAILS"])
    sys.exit(1)
print(f"\n  sweep {SWEEP}: PART 0..L ran silently, gates passed ({time.time() - T0:.0f} s)", flush=True)

N, n_full, K, H, ELL, DB, E0, TWOPI, DC = (G[k] for k in ("N", "n_full", "K", "H", "ELL", "DB", "E0", "TWOPI", "DC"))
CTS_KS, CTS_ROT, R_STC = G["CTS_KS"], G["CTS_ROT"], G["R_STC"]
C_DENSE, C_SPARSE = G["C_DENSE"], G["C_SPARSE"]
KEFF_FIX, SUBSUM = G["KEFF_FIX"], G["SUBSUM"]
bks, depth, cheb_coeffs, q0lam = G["bks"], G["depth"], G["cheb_coeffs"], G["q0lam"]
parseval_factor, c_of, part_of = G["parseval_factor"], G["c_of"], G["part_of"]
amp_inj, amp_after_first, towers_layout = G["amp_inj"], G["amp_after_first"], G["towers_layout"]
model, eval_K, sens_bounds, cells_ext = G["model"], G["eval_K"], G["sens_bounds"], G["cells_ext"]
bits, dec, K_ROWS = G["bits"], G["dec"], G["K_ROWS"]
LEVD = [2 ** (k + 1) - 1 for k in CTS_KS]


# =============================================================================================
# model with per-level residues (PART N's model_N with R_level in place of d * R_diag)
# =============================================================================================
def model_O(lD, lq0, arc, ns, Rlev=None, scal=True, fold="spread", ss=None, dc=DC):
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
    B_term = bks(2, lq0, lD) / D
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
    s1 = S ** (1.0 / len(CTS_KS)) if fold == "spread" else 1.0
    for i, k in enumerate(CTS_KS):
        res = Rlev[i] * (nu + B) / p
        B = 2.0 ** k * s1 * B + C_SPARSE + CTS_ROT[i] * bks(top - i, lq0, lD) / D + res
        add_cts = 2.0 ** k * s1 * add_cts + res
        nu = 2.0 ** k * s1 * nu
    if fold == "end":
        res_s = (0.5 * (nu + B) / p) if scal else 0.0
        B = S * B + C_SPARSE + res_s
        add_cts = S * add_cts + res_s
    elif fold == "spread":
        B = B + C_SPARSE
    else:
        raise ValueError(fold)
    em_top = top - len(CTS_KS)
    B_conj = bks(em_top, lq0, lD) / D
    B_cts = 2 * B + B_conj
    eD, add_em = evalmod_old(lD, lq0, arc, Keff, B_cts, em_top, scal)
    B_sig = B_cts + B_conj + B_term
    Xi = B_sig / D * q0 * Keff
    stc_top = 1 + dc + R_STC
    return dict(B_cts=B_cts, eD=eD, Xi=Xi, B_sd=bks(stc_top, lq0, lD) / D,
                logQ=lq0 + (top - 1 + extra_lvl) * lD, top=top, Keff=Keff, B_in=B_in,
                B_cj=bks(stc_top + 1, lq0, lD) / D, E_ss=E_ss, B_conj=B_conj, em_top=em_top,
                add_cts=2 * add_cts, add_em=add_em, scal=scal, S=S, s1=s1)


def scal_res(nu_over_D, B, D):
    return 0.5 * (nu_over_D * D + B) / D


def evalmod_old(lD, lq0, arc, Keff, B_cts, em_top, scal):
    D, q0 = 2.0 ** lD, 2.0 ** lq0
    eps = D / q0
    CS = C_SPARSE
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
        r_A = 0.5 if scal else 0.0
        B = 4 * A * B + 2 * B * B / D + CS + bks(tw, lq0, lD) / D + r_A
        add = 4 * A * add + r_A
    if arc:
        tw = em_top - 12
        nu_y = D * eps * 1.01
        B2 = (2 * nu_y * B + B * B) / D + CS + bks(tw, lq0, lD) / D
        nu_y2 = D * (eps * 1.01) ** 2
        a = TWOPI ** 2 / 6.0
        r_a = scal_res(eps * 1.01 * eps * 1.01, B2, D) if scal else 0.0
        B3 = a * (nu_y2 * B + nu_y * B2 + B * B2) / D + CS + bks(tw - 1, lq0, lD) / D + r_a
        B = B + 2 * CS + B3
        add += r_a
    return B, add


def evalmod_mu(lD, lq0, arc, Keff, em_top, sz, su, dy, scal):
    D, q0 = 2.0 ** lD, 2.0 ** lq0
    eps = D / q0
    CS = C_SPARSE
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


R_L1 = N / 2.0


def stc_residue(P, D, B_in, tau):
    """StC encoded-diagonal residues under the paper's ||.||_1 check (R = N/2 per diagonal)."""
    tot = 0.0
    for i, k in enumerate(P):
        nu_i = math.sqrt(2.0) * (D + B_in) * 2.0 ** sum(P[:i])
        tot += (2 ** (k + 1) - 1) * R_L1 * (nu_i + tau) / D * 2.0 ** sum(P[i + 1:])
    return tot


NO_SENS = []
NONFINITE = []          # tags that overflow (residues too large at that cell): the cell is never optimal
np.seterr(over="ignore")


def eval_KO(lD, degb, arc, ns, Rlev=None, scal=True, fold="spread", stc=True):
    lq0 = lD + degb
    D, q0 = 2.0 ** lD, 2.0 ** lq0
    mo = model_O(lD, lq0, arc, ns, Rlev, scal, fold=fold)
    eps_in = (D + mo["B_in"]) / q0
    if eps_in >= 0.5:
        return None
    A_sup = c_of(ns) * ns
    back = 2.0 ** degb
    P = part_of(ns)
    QL = q0lam(lq0, mo["Keff"])
    t = dict(approx=A_sup * QL + q0 * parseval_factor(eps_in, arc), stc=C_DENSE * amp_inj(P),
             bsd=mo["B_sd"] * amp_after_first(P), conj=ns * back * mo["B_cj"],
             post=(2 * ns - A_sup) * back * C_SPARSE)
    delta, eta = mo["B_cts"] / D, mo["B_conj"] / D
    try:
        if math.exp(math.acosh(1 + delta)) * 1.01 >= 40.0:
            raise ValueError("rho grid")
        sb = sens_bounds(mo["Keff"], arc, delta, eta)
        eD_int = evalmod_mu(lD, lq0, arc, mo["Keff"], mo["em_top"], sb["sz"], sb["su"],
                            delta * sb["L_R"] + eta * sb["L_I"], scal)
        eD_K = eD_int + sb["L_R"] * mo["B_cts"] + sb["L_I"] * mo["B_conj"]
    except (ValueError, OverflowError):
        eD_int = eD_K = math.inf
        NO_SENS.append((lD, degb, arc, ns))
    t["noise"] = A_sup * back * min(mo["eD"], eD_K)
    if stc:
        t["stc_res"] = stc_residue(P, D, mo["B_in"], sum(t.values()))
    tot = sum(t.values())
    if not math.isfinite(tot):
        NONFINITE.append((lD, degb, arc, ns))
    return tot, dict(t=t, mo=mo, eD_K=eD_K, eD_int=eD_int)


def can_residue(nsamp=8, seed=20261001):
    """PART N's MODEL value (for gate O0f only)."""
    rng = np.random.default_rng(seed)
    best = 0.0
    tw = np.exp(1j * np.pi * np.arange(N) / N)
    for _ in range(nsamp):
        r = rng.uniform(-0.5, 0.5, N)
        v = np.fft.fft(r * tw)
        best = max(best, float(np.max(np.abs(v))))
    return best


# =============================================================================================
# O0e / O0f (model gates)
# =============================================================================================
print("\nPART O0  GATES (model)")
fields = ("B_cts", "eD", "Xi", "B_sd", "logQ", "Keff", "B_in", "B_cj", "E_ss")
worst = 0.0
for (lD, lq0, arc, ns) in ((53, 60, 0, n_full), (58, 64, 3, 1), (59, 64, 3, 8), (59, 64, 3, 256),
                           (61, 64, 3, n_full), (62, 64, 0, n_full), (60, 64, 3, 2 ** 12),
                           (40, 50, 0, n_full), (45, 52, 3, 4), (50, 60, 0, 2 ** 15),
                           (56, 64, 0, 2), (36, 39, 3, n_full)):
    a_ = model(lD, lq0, DC, arc, ns)
    b_ = model_O(lD, lq0, arc, ns, None, scal=False, fold="end")
    for f in fields:
        x, y = a_[f], b_[f]
        worst = max(worst, abs(x - y) / abs(x) if x else abs(y))
HEAD = [("ref", 53, 7, 0, n_full)]
for r in K_ROWS:
    if r[0] in ("2^0", "2^3", "2^8", "2^16"):
        lD_, degb_, arc_ = r[4].split("/")
        HEAD.append((r[0], int(lD_), int(degb_), 3 if arc_ == "on" else 0, 2 ** int(r[0][2:])))
CELLS = list(cells_ext())
rand = [(f"rnd{i}",) + CELLS[j] + (int(RNG.choice([0, 3])), int(2 ** RNG.integers(0, 17)))
        for i, j in enumerate(RNG.choice(len(CELLS), 40, replace=False))]
worst2 = 0.0
for lab, lD, degb, arc, ns in HEAD + rand:
    tK, t_old, _ = eval_K(lD, degb, arc, ns)
    tO, _ = eval_KO(lD, degb, arc, ns, None, scal=False, fold="end", stc=False)
    worst2 = max(worst2, abs(tO - tK) / t_old)
gate("O0e residues off: model_O = model() (12 cells, every field); eval_KO = eval_K() (5 + 40 cells)",
     worst < 1e-12 and worst2 < 1e-12, f"{worst:.1e} / {worst2:.1e}")
R_CAN = can_residue()
PARTN = {"ref": (10.060, 17.410), "2^0": (-11.928, -4.638), "2^3": (-10.937, -3.985),
         "2^8": (-6.279, 1.011), "2^16": (-2.359, 4.614)}           # recorded PART N v2: spread+can, spread+l1
w3 = 0.0
for lab, lD, degb, arc, ns in HEAD:
    dc_ = dec(eval_KO(lD, degb, arc, ns, [d * R_CAN for d in LEVD], True, "spread", stc=False)[0], lD)
    dl_ = dec(eval_KO(lD, degb, arc, ns, [d * R_L1 for d in LEVD], True, "spread", stc=False)[0], lD)
    w3 = max(w3, abs(dc_ - PARTN[lab][0]), abs(dl_ - PARTN[lab][1]))
gate("O0f R_level = d R_can / d N/2 reproduce PART N v2 spread+can / spread+l1 at the 5 headline cells (0.0015)",
     w3 < 0.0015, f"max dev {w3:.4f}")
stop_if_failed("PART O0")

# =============================================================================================
# PART O1: certification
# =============================================================================================
print("\nPART O1  certify R_level on the actual CtS diagonals (N = 2^17, double-double phi^-1)", flush=True)
n = N // 2
Gg, BRg = gens(N), bitrev_idx(n)
E = twiddle_table(2 * N)
print(f"    twiddle table 2N = 2^{(2 * N).bit_length() - 1} built ({time.time() - T0:.0f} s)", flush=True)
COUNTS = run_big_gates(N, CTS_KS, E, Gg, BRg, RNG)
stop_if_failed("PART O1")

NGRP = 8
VAR_CELLS = [HEAD[0], HEAD[-1]]          # ref and the full-packing optimum
jobs = {}
cell_s1 = {}
for (lD, degb) in CELLS:
    mo = model_O(lD, lD + degb, 0, n_full, None, True, "spread")
    cell_s1[(lD, degb)] = mo["s1"]
    jobs[(lD, degb, "spread", 0)] = (lD, mo["s1"], ntt_primes(lD, N, 5))
for lab, lD, degb, arc, ns in VAR_CELLS:
    pr = ntt_primes(lD, N, 5 * NGRP)
    for g_ in range(1, NGRP):
        jobs[(lD, degb, "spread", g_)] = (lD, cell_s1[(lD, degb)], pr[5 * g_:5 * g_ + 5])
for lab, lD, degb, arc, ns in HEAD:
    jobs[(lD, degb, "end", 0)] = (lD, 1.0, ntt_primes(lD, N, 5))
info_keys = [(lD, degb, "spread", 0) for _, lD, degb, _, _ in HEAD]
print(f"    {len(jobs)} encodings per diagonal ({len(CELLS)} cells x 1 instance, 2 cells x {NGRP - 1} more prime "
      f"groups, 5 headline cells without the spread scale)", flush=True)
cert = Certifier(N, jobs, info_keys)
certify(N, CTS_KS, E, Gg, BRg, cert)
gate("O0d every rounding residue |r_k| <= 1/2 + 1e-9", cert.max_abs_r <= 0.5 + 1e-9,
     f"max {cert.max_abs_r:.12f}; |r| > 1/2 - 1e-9 (near-ties, either rounding is fine): {cert.near_ties}")
stop_if_failed("PART O1")
RC = {k: cert.R[k] for k in jobs}
Rcell = lambda lD, degb: RC[(lD, degb, "spread", 0)]

print(f"    diagonals per level (actual / port's 2^(k+1)-1): "
      + ", ".join(f"{c}/{d}" for c, d in zip(COUNTS, LEVD)))
print(f"    R_level in bits at the headline cells (sum over diagonals of ||sigma(r_d)||_inf; MODEL d R_can = "
      + "/".join(f"{bits(d * R_CAN):.2f}" for d in LEVD) + ")")
O1 = []
for lab, lD, degb, arc, ns in HEAD:
    k_ = (lD, degb, "spread", 0)
    R_ = RC[k_]
    tight = [float(np.max(cert.info[k_][lv])) for lv in range(len(CTS_KS))]
    O1.append((lab, lD, degb, R_, tight))
    print(f"    {lab:>5} {lD}/{degb}: s1 = 2^{bits(cell_s1[(lD, degb)]):.3f} | R_level 2^"
          + " 2^".join(f"{bits(x):.2f}" for x in R_) + " | max_j sum_d (no BSGS) 2^"
          + " 2^".join(f"{bits(x):.2f}" for x in tight))
allR = np.array([Rcell(*c) for c in CELLS])
print("    over all " + str(len(CELLS)) + " cells, per level: min 2^" + " 2^".join(f"{bits(x):.2f}" for x in allR.min(0))
      + " | max 2^" + " 2^".join(f"{bits(x):.2f}" for x in allR.max(0)))
VAR = []
for lab, lD, degb, arc, ns in VAR_CELLS:
    Rs = np.array([RC[(lD, degb, "spread", g_)] for g_ in range(NGRP)])
    ds = [dec(eval_KO(lD, degb, arc, ns, list(Rs[g_]), True, "spread", True)[0], lD) for g_ in range(NGRP)]
    VAR.append((lab, lD, degb, Rs.min(0), Rs.max(0), min(ds), max(ds)))
    print(f"    prime-instance spread at {lab} {lD}/{degb} ({NGRP} groups of 5 primes): R_level min 2^"
          + " 2^".join(f"{bits(x):.2f}" for x in Rs.min(0)) + " | max 2^" + " 2^".join(f"{bits(x):.2f}" for x in Rs.max(0))
          + f" | dec {min(ds):+.3f} .. {max(ds):+.3f}")
print(f"    ({time.time() - T0:.0f} s)", flush=True)

# =============================================================================================
# PART O2: headline cells
# =============================================================================================
print("\nPART O2  at the base optimum cells")
print("    base = the port; spread = S^(1/5) per level, no residues; cert = spread + certified CtS residues")
print("    + scalar residues (c_j, A^2, a); cert+StC = cert + StC residues under the ||.||_1 check (the")
print("    certified configuration); end+cert = S folded at the end, CtS diagonals at scale p certified")
print("    (decision iv); MODEL = PART N spread+can.")
O2 = []
for lab, lD, degb, arc, ns in HEAD:
    lq0 = lD + degb
    Rs = Rcell(lD, degb)
    Re = RC[(lD, degb, "end", 0)]
    cfg = [("base", None, False, "end", False), ("spread", None, False, "spread", False),
           ("cert", Rs, True, "spread", False), ("cert+StC", Rs, True, "spread", True),
           ("end+cert", Re, True, "end", True), ("MODEL", [d * R_CAN for d in LEVD], True, "spread", False)]
    ds = []
    for _, R_, sc, fo, st in cfg:
        ds.append(dec(eval_KO(lD, degb, arc, ns, R_, sc, fo, st)[0], lD))
    mb = model_O(lD, lq0, arc, ns, None, False, "end")
    mc = model_O(lD, lq0, arc, ns, Rs, True, "spread")
    tc = eval_KO(lD, degb, arc, ns, Rs, True, "spread", True)[1]["t"]
    O2.append((lab, f"{lD}/{degb}/{'on' if arc else 'off'}", ds, bits(mb["B_cts"]), bits(mc["B_cts"]),
               bits(mc["add_cts"]), bits(tc["stc_res"]) if tc["stc_res"] > 0 else float("-inf"),
               bits(tc["stc"]) if tc["stc"] > 0 else float("-inf")))
    print(f"    {lab:>5} {O2[-1][1]:<10} | " + " ".join(f"{c[0]} {x:+.3f}" for c, x in zip(cfg, ds)))
    r_ = O2[-1]
    print(f"          B_cts base 2^{r_[3]:.3f} -> cert 2^{r_[4]:.3f} (CtS residue 2^{r_[5]:.2f}) | "
          + (f"StC residue 2^{r_[6]:.2f} vs StC injection 2^{r_[7]:.2f}" if r_[7] > -1e9 else "no StC levels"))

# =============================================================================================
# PART O3: re-optimised landscape
# =============================================================================================
print("\nPART O3  re-optimised best cell per packing (logD 36..62, degb 2..14, arcsine on/off)", flush=True)
PACK = [(f"2^{k}", 2 ** k) for k in range(0, 17)]


def best_over(ns, mode):
    best = None
    for arc in (0, 3):
        for lD, degb in CELLS:
            if mode == "base":
                r = eval_KO(lD, degb, arc, ns, None, False, "end", False)
            else:
                r = eval_KO(lD, degb, arc, ns, Rcell(lD, degb), True, "spread", True)
            if r is None:
                continue
            d_ = dec(r[0], lD)
            if best is None or d_ < best[0]:
                best = (d_, lD, degb, arc, r[1]["mo"]["logQ"])
    return best


O3 = []
cellf = lambda b: f"{b[1]}/{b[2]}/{'on' if b[3] else 'off'}"
print(f"    {'n_s':>5} | {'base':>18} | {'cert+StC':>18} {'loss':>6} | logQ")
for lab, ns in PACK:
    b0, b1 = best_over(ns, "base"), best_over(ns, "cert")
    O3.append((lab, b0, b1))
    print(f"    {lab:>5} | {b0[0]:+7.2f} {cellf(b0):>10} | {b1[0]:+7.2f} {cellf(b1):>10} {b1[0] - b0[0]:+6.2f} | {b1[4]}")
TH = {}
for j, nm in ((1, "base"), (2, "cert+StC")):
    for thr in (-5.0, -10.0):
        ok = [int(r[0][2:]) for r in O3 if r[j][0] < thr]
        TH[(nm, thr)] = f"n_s<=2^{max(ok)}" if ok else "none"
        print(f"    {nm}: largest n_s with T < 2^{thr:g}: {TH[(nm, thr)]}")
print(f"\n    [INFO] evaluations where Lemma sens was not evaluated (E*_eval = E_eval): {len(NO_SENS)}")
print(f"    [INFO] evaluations whose tag overflows (never optimal): {len(NONFINITE)}")
print(f"    total runtime {time.time() - T0:.0f} s")

# =============================================================================================
print("\n" + "=" * 100)
print("SUMMARY BEGIN")
print(f"O counts {'/'.join(map(str, COUNTS))} (port {'/'.join(map(str, LEVD))}) | MARGIN 2^{bits(MARGIN):.0f} | "
      f"max|r| {cert.max_abs_r:.12f} near-ties {cert.near_ties} | R_can 2^{bits(R_CAN):.2f}")
for lab, lD, degb, R_, tight in O1:
    print(f"O1 {lab} {lD}/{degb}: R_level " + " ".join(f"{bits(x):.2f}" for x in R_)
          + " | no-BSGS " + " ".join(f"{bits(x):.2f}" for x in tight))
print("O1 all cells min " + " ".join(f"{bits(x):.2f}" for x in allR.min(0)) + " | max "
      + " ".join(f"{bits(x):.2f}" for x in allR.max(0)))
for lab, lD, degb, mn, mx, dmin, dmax in VAR:
    print(f"O1 primes {lab} {lD}/{degb}: min " + " ".join(f"{bits(x):.2f}" for x in mn) + " | max "
          + " ".join(f"{bits(x):.2f}" for x in mx) + f" | dec {dmin:+.3f}..{dmax:+.3f}")
for r_ in O2:
    print(f"O2 {r_[0]} {r_[1]}: " + " ".join(f"{c} {x:+.3f}" for c, x in
                                             zip(("base", "spread", "cert", "cert+StC", "end+cert", "MODEL"), r_[2]))
          + f" | B_cts 2^{r_[3]:.3f}->2^{r_[4]:.3f} CtSres 2^{r_[5]:.2f} StCres 2^{r_[6]:.2f} StCinj 2^{r_[7]:.2f}")
for lab, b0, b1 in O3:
    print(f"O3 {lab}: base {b0[0]:+.2f} {cellf(b0)} | cert+StC {b1[0]:+.2f} {cellf(b1)} ({b1[0] - b0[0]:+.2f}) logQ {b1[4]}")
for (nm, thr), v in TH.items():
    print(f"O3 {nm} threshold T<2^{thr:g}: {v}")
print(f"O no-sens evaluations: {len(NO_SENS)} | overflowing evaluations: {len(NONFINITE)} | runtime {time.time() - T0:.0f} s")
print("GATE FAILURES:", FAILS if FAILS else "none")
print("SUMMARY END")
print("=" * 100)
