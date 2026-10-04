# Verifiable CKKS Bootstrapping with Certified Numerical Soundness

These scripts compute or check every numerical claim in the paper's evaluation, landscape and appendix tables. Each script starts with **gates**: checks against known values or independent computations. A script stops with exit code 1 if any gate fails, so a number is printed only after its gates pass.

Every script is pure NumPy on **ideal (unencrypted) values**. None implements the composed argument Π_pipe, the pinned key switch, the lift gadget or the annotation layer. There are two exceptions:

- `estimate_security.py` calls the lattice estimator, which runs under SageMath.
- `measure_ref_n17.py` measures an *unverified* OpenFHE bootstrap. It is the reference point (E5) only, not a certificate, and no parameter in the paper comes from it.

## Requirements

See `requirements.txt`.

| component | version | used by |
|---|---|---|
| Python | 3.11 (verified with 3.11.17; also runs on 3.13) | all scripts |
| NumPy | 2.4.6 (verified; 2.5.3 also verified) | all scripts |
| SageMath + [lattice-estimator](https://github.com/malb/lattice-estimator) | estimator commit `53da598` | `estimate_security.py` |
| OpenFHE + Python bindings | `openfhe-development` commit `ed361af`, bindings 1.5.1.0.24.4, 64-bit native | `measure_ref_n17.py` |

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cd scripts
```

## Layout and run order

All scripts live in `scripts/` and **must stay in the same folder**. The certified-landscape scripts form a chain: each one runs its predecessor in-process with output suppressed, stops if any predecessor gate failed, and reuses its results.

```
sweep_degb_delta_arcsine.py   (the port of the audit; hard gate against the reference values)
   └─ sweep_partO.py          (certifies the residues of the encoded CtS diagonals)
        └─ sweep_partP.py     (forced decisions; full-packing optimum)
             └─ sweep_partQ.py (certified sparse-packing optima)
                  ├─ sweep_partR.py   (reference-cell constants, Lemma sens numbers, ablations)
                  └─ sweep_partS.py   (sequential composition, projection level)
```

Recommended order:

```bash
python3 check_stc_amplification.py     # Lemma stc: rho(CtS), rho(V0), c(n)
python3 check_units_sparse_l2.py       # Lemma parseval, Lemma units
python3 check_stc_attainment.py        # Prop. attain (exact worst case of a rescale injection)
python3 check_lipschitz_bound.py       # Lemmas da-cheb, sens-real
python3 inventory_ledger.py            # tab:inventory, range rows of tab:ledger-full
python3 sweep_partQ.py                 # tab:landscape, tab:packing, thresholds, decisions (runs O, P, the port)
python3 sweep_partR.py                 # tab:refconst, Sec. 5 / Sec. 7 constants, ablations (runs Q)
python3 sweep_partS.py                 # tab:compose, projection level (runs Q)
sage -python estimate_security.py --estimator /path/to/lattice-estimator --logq 1552 1657 1684 1711
python3 measure_ref_n17.py --dry && python3 measure_ref_n17.py   # E5, needs OpenFHE
```

Every script prints a compact block between `SUMMARY BEGIN` and `SUMMARY END` at the end. The paper's tables report log2 T rounded **upward** to the printed precision (the conservative direction for an upper bound): for example the script prints `-13.798` and `+8.013`, and the paper reports −13.79 and +8.02. The *output line* column of the claim map below refers to the prefixes of the lines in that block (for example `R2 summary:`).

### Runtime and memory

Wall-clock times measured on a 2-vCPU cloud VM with two scripts running at once. Expect them to be shorter on an idle laptop.

| script | time | peak memory |
|---|---|---|
| `check_stc_amplification.py` (default N ≤ 2^12) | ≈ 5 s | < 1 GB |
| `check_units_sparse_l2.py` | ≈ 2 s | < 1 GB |
| `check_stc_attainment.py` | ≈ 10 s | < 1 GB |
| `check_lipschitz_bound.py` | ≈ 1.5 min | < 1 GB |
| `inventory_ledger.py` | < 1 s | negligible |
| `sweep_degb_delta_arcsine.py` alone | ≈ 1.5 min | < 1 GB |
| `sweep_partP.py` (includes O and the port) | ≈ 6 min | < 1 GB |
| `sweep_partQ.py` (includes P) | ≈ 8 min | < 1 GB |
| `sweep_partR.py` (includes Q) | ≈ 10 min | < 1 GB |
| `sweep_partS.py` (includes Q) | ≈ 9.5 min | < 1 GB |
| `estimate_security.py` | minutes per row; one hybrid call can take up to the `--timeout` (3600 s) | Sage |
| `measure_ref_n17.py` | ≈ 130 s per bootstrap; 30 trials plus key generation | ≈ 3.4 GiB |

The chained scripts print `RuntimeWarning`s from `numpy/polynomial/chebyshev.py` (overflow/invalid). These come from alternative configurations that are rejected at small Δ, where the residue terms overflow. Such cells are never optimal, and their count is reported as `overflow` in the summary.

## Gates and expected output

A clean run ends with `GATE FAILURES: none`. Every gate is printed as `[PASS]` or `[FAIL]` with its measured value.

| script | gates | what they check (expected) |
|---|---|---|
| `sweep_degb_delta_arcsine.py` | PART 0 (hard gate) | reproduces the reference implementation of the audit at Δ = 2^53, q0 = 2^60, d_bg = 7: B_CtS = 2^25.504 (±0.01), E_eval = 2^38.548 (±0.02), Ξ = 2^37.504, E = 4.51·10^-13 at ρ* = 12.07, δ' = 2^41.718, all ten cells of the scale–depth sweep (±0.02 bits), legacy V4 = −10.95, V5 = −3.55 |
| | 0b, E | the Parseval variant never exceeds the sup-norm variant (2952 and the extended combinations) |
| | I0, J0, J1, K0–K3 | noise-tag decomposition reproduces E_eval (1e-12); chain-rule derivatives match finite differences; analytic L_R ≥ grid max \|p'\| at every optimum; Bernstein-ellipse containment and closed-form tail sums |
| `sweep_partO.py` | O0a–O0f | special FFT = V0; double-double twiddles (1e-30) and inverse (1e-28); merged CtS levels = V0^-1 at N = 2^10 densely and at N = 2^17 on random vectors; every rounding residue \|r_k\| ≤ 1/2; residues off ⇒ identical to the port (1e-12); recorded PART N values (±0.0015) |
| `sweep_partP.py` | P0a–P0d | residue-free decisions reproduce the recorded PART M values (±0.0015); certified tags = PART O (1e-12) and the recorded PART O2 values; optima = PART O3 (exact) |
| `sweep_partQ.py` | Q0a–Q0d | full-packing certification reproduces PART O; **Q0b: no uncertified cell has a residue-free lower bound below the reported optimum**, so the optimum is certified over the whole domain; \|r_k\| ≤ 1/2 |
| `sweep_partR.py` | R0a–R0e | residue-free mode = the port's `eval_K`, `eval_K_noP` and old tag (1e-12, 1e-9, 1e-12); optima = Q/P (exact); every ablation optimum certified (pruning margin ≥ 0) |
| `sweep_partS.py` | S0a–S0e | port routing = Q (1e-12) and the 17 optima; seeded EvalMod copies = PART P (1e-12); re-certification = Q (1e-12); lemma routing ≤ port routing at k = 1; \|r_k\| ≤ 1/2 |
| `check_stc_amplification.py` | Part A, B, C gates | V[:, n:] = iV0, V0V0* = nI, decode identity; the adversarial real vector attains the sup; ideal CtS/EvalMod channels exact |
| `check_units_sparse_l2.py` | PART 0, A, B | reference constants; δ' = 2^41.718 is in x-units; **Parseval: slot-bounded ratio ≤ 1** at every N |
| `check_stc_attainment.py` | G1–G6 | exact dual norm = vertex enumeration (N = 8) and ≥ θ-grid; Lemma stc (ii) reproduced (c(n)·n, 1e-9); G6: worst case ≤ Cauchy–Schwarz bound and ≤ certificate at every N |
| `check_lipschitz_bound.py` | A0–A5 | double-angle identity; grid max \|p0'\| = K_eff; tail bounds; **A3: grid max \|p'\| ≤ L_an** for every case |
| `inventory_ledger.py` | I0, L0, C0, C1, D0 | reproduces the original inventory (7881 chunks) and recorded ledger rows; chain closure (every chain prime consumed by a rescale) |
| `estimate_security.py` | GATE | reproduces two recorded rows at commit `53da598`: log Q = 1332 (bdd 375.7, hybrid 168.0) and 1385 (bdd 359.1, hybrid 165.4), ±0.2 bits |
| `measure_ref_n17.py` | — | refuses to run if the ring dimension is not 2^17; `DEV = True` (N = 2^12, insecure) is a smoke test only and is flagged in its output |

"Reference implementation" means an independent implementation of the same audit, which produced the hard-gate values of `tab:refconst` (first block).

## Claim map: paper → script → output line

| paper item | value(s) | script | output line |
|---|---|---|---|
| Lemma stc: ρ(CtS) = 1, ρ(V0) = n, c(n) → 4/π; random errors ≈ √n | c(2^12 slots) = 1.273239 | `check_stc_amplification.py` | `SUMMARY B:` table |
| (E2) attaining vector reproduces c(n)·n | R_adv/n → 1.2732 | `check_stc_amplification.py` | `SUMMARY C:` table |
| Lemma parseval: slot-bounded ratio ≤ 1, = 1 at z = 1; coefficient-aligned grows ∝ n | N = 2^6 … 2^11 | `check_units_sparse_l2.py` | `B N=2^k:` |
| Lemma units (δ' in x-units, 2^d_bg) | 2^41.718 | `check_units_sparse_l2.py` | PART A |
| Prop. attain, exact worst case: ∝ N^1.49, within 1.2 bits of the bound | N = 2^6 … 2^12 | `check_stc_attainment.py` | `FIT …` slope, `ATT …` `cs/A` |
| Lemmas da-cheb, sens-real: L_R ≤ K_eff (1 + 2^-22.1) | 2^-22.15 | `check_lipschitz_bound.py` | `SUMMARY max over cases` |
| tab:landscape, tab:packing (log2 T, cell, log Q), all 17 packings | −13.79 … −3.62 | `sweep_partQ.py` | `Q1 2^j:` (`cert`, sparse), `Q2 base`; full packing from `sweep_partP.py` `P4 2^16` |
| same values, re-checked on the certified configuration | | `sweep_partR.py` | `R2 2^j <cell>: dec` |
| thresholds: T < 2^-10 up to 2^5 slots, T < 2^-5 up to 2^13 slots | | `sweep_partQ.py` | `Q4 threshold` |
| decisions (ii) inadmissible; (iii) 5.5, (iv) 5.1, (vi) 14.2, (vii) 0.2, (viii) 5.4 bits | re-optimised | `sweep_partQ.py` | `Q2 (ii) …` … `Q2 (c) …` (sparse); `sweep_partP.py` `P2 …` (full) |
| (iv) without encoding residues ≤ 0.02 bits | | `sweep_partP.py` | gate P0a (iv-s values) |
| tab:refconst, block 1 (hard gate) | B_CtS 2^25.504, E_eval 2^38.548, Ξ, E, δ_int, δ' | `sweep_degb_delta_arcsine.py` | PART 0, `REFERENCE CELL` |
| tab:refconst, block 2: R_1 … R_5 | 2^12.51 … 2^4.69 | `sweep_partO.py` | `O1 ref 53/7: R_level` |
| tab:refconst, residue term and B_CtS 2^32.03; block 3 (E_int, L_R B_CtS, L_I B_cj, E*, L_R) | | `sweep_partR.py` | `R1 ref:` |
| tab:refconst, block 4 (refresh bound per term) and log2 T = +8.02 | | `sweep_partR.py` | `R1 ref terms:` |
| Sec. 5: L_R/K_eff − 1 ≤ 2^-22.2, L_I ≤ 2^11.6, B_CtS − B_cj ≥ 18 bits, input share 47–62 %, gate-by-gate tags overstate by > 7 bits, E_cj ≥ 25 and E_post ≥ 14 bits below the dominant term | | `sweep_partR.py` | `R2 summary:` |
| Sec. 7 (c), (d): sine floor 2^-5.3 (full) and 2^-11.3 (ref), 1.72 bits; noise vs sine 1.2 bits | | `sweep_partR.py` | `R3 full …` |
| Sec. 7 ablations: no Lemma parseval −13.74 / +2.11; no Lemma sens −8.98 / +0.85; gains 5.7, 0.05, 4.3–4.9 bits | | `sweep_partR.py` | `R4 …` |
| the StC factor: 15.85 bits | | `sweep_degb_delta_arcsine.py` | PART D `reference:` |
| projection level: B_CtS changes ≤ 1.3·10^-5 bits, T ≤ 3.2·10^-6 bits, E_ss/k ≥ 8.8 bits below B_CtS | | `sweep_partS.py` | `S2 …`, `S2 max …` |
| tab:compose and app:compose (refresh counts; second refresh adds T1 within 0.9 % / 4T1 within 1.0 %; a ≥ 2.9 bits below 1/2; port routing 0.002–0.06 bits higher; seed exceeded by up to 10.1 %) | | `sweep_partS.py` | `S3 <circuit> 2^<log tau> 2^j:`, `S1 …` |
| tab:inventory (6369 / 4984 / 11353 chunks), 6363 digit elements, ℓ* = 29.64, 60 rescales, 27 range calls, shares 56.0 / 41.9 / 57.1 % | | `inventory_ledger.py` | `INV D drop+cj:`; range-call count in PART D |
| tab:ledger-full, range rows: ModRaise 2^-185.96, digit, rescale, decomposition (within 0.06 bits) | | `inventory_ledger.py` | `LED D:` |
| tab:security (primal hybrid) | 168.1 … 151.1 | `estimate_security.py` | `EST logQ=…: … hybrid=…` |
| (E5) unverified OpenFHE bootstrap at N = 2^17, full packing: max error 2^-11.19 (ones), 2^-12.12 (alt), 2^-15.01 (uniform) | | `measure_ref_n17.py` | `B-ref family …`, `B-ref MAX …` |

## Not in this repository

- (E1) the exact prototype (41-gate bootstrap at N = 2).
- (E3) the branch certificates and the N = 32 CtS audit.
- (E4) the range-check prototype over F_{2^50+55}.
- (E5) the paired three-refresh experiment.
- The commitment and circuit rows of `tab:ledger-full`, its interactive total, and ε_arg at the optima.

`inventory_ledger.py` prints only the range rows. Its PART C (`OPT …` lines) uses an earlier set of optimum cells and is not used by the paper.

`check_stc_amplification.py` and `check_units_sparse_l2.py` also evaluate the error budget of the reference implementation (their Part D and PARTs C–E, variants V1–V5). Those parts are diagnostics and are not used by the paper.

## Floating point

The scripts evaluate the audit in double precision. φ^-1 of the encoded CtS diagonals uses double-double arithmetic. The evaluation is not outward-rounded: L_R, L_I and the residues R_i are the paper's bounds evaluated in floating point.

Each diagonal residue carries an a-priori margin of 2^-8 (`MARGIN` in `sweep_partO.py`). The margin covers the double-double inverse, which round-trips at N = 2^17 to a relative error below 2^-86, and the error of the double-precision FFT. An audit as defined in the paper evaluates the same expressions with outward rounding.

## License

MIT, see `LICENSE`.
