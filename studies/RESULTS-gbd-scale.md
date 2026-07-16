# Estimator precision at the published GBD 2010 design scale

Source: `studies/gbd_scale.py`. Per-state output: `studies/results/gbd_scale.csv` (440 rows: 2 seeds x 220 states, per-seed scalars broadcast onto every row). All numbers below are pasted from the script's stdout.

## Design

K=220 synthetic states with true disability weights logit-spaced over [0.003, 0.95] (`simulate_true_dws(220, low=0.003, high=0.95)` — low passed explicitly; the simulator's own default is 0.005). Paired comparisons: 14,000 respondents x 15 pairs = 210,000 rows. PHE: 4,000 respondents x 3 questions = 12,000 rows over 30 anchor states with the GBD c-grid (1,000 deaths vs c in {1500, 2000, 3000, 5000, 10000}, `simulate_phe` defaults). Two seeds (0 and 1), with the PC and PHE draws decorrelated within a replicate (pc rng=seed, phe rng=seed+1000). Fits run through `estimate_dws` unchanged — no stage logic reimplemented, no design matrix rebuilt outside `fit_pc`.

Anchor selection: 95 of the 220 states have true dw in the mid-range band [0.05, 0.70] where the PHE thresholds logit(1000/c) bracket the latent distribution. The rubric asks for exactly 30, so the study takes 30 evenly spaced by rank within the 95 eligible (realized dw range [0.051, 0.696]). The within-band selection rule is a design choice of this study, not pinned by the instrument.

DGP constants slope=-1.0, intercept=0.3, sigma=0.8 match `tests/test_recovery.py` for internal consistency — a chosen convention, not a feature of the GBD design.

## Scope

This report presents simulated-design precision ONLY: how well the pipeline recovers known true weights from a survey of the published size, when respondents follow the assumed DGP exactly. Comparison against the published GBD 2010 uncertainty intervals is left to the reader — the published tables sit outside this repo's engine surface by project policy and were not consulted. Two seeds are illustrative, not a formal Monte Carlo standard error.

## Headline recovery

Seed 0: rank corr 0.99974, mean |err| 0.00470, max |err| 0.03223. Seed 1: rank corr 0.99980, mean |err| 0.00475, max |err| 0.03872. Pooled over both seeds: rank corr 0.99972, mean |err| 0.00472, max |err| 0.03872. Errors are in probability space (the units DALY weights are reported in). At this scale the estimator recovers the ranking of 220 states essentially perfectly and the levels to under 0.005 on average and under 0.04 at worst.

## Anchor diagnostics

Seed 0: slope -0.9913, R^2 0.9942, n_shared 30. Seed 1: slope -0.9913, R^2 0.9958, n_shared 30. No one-sided PHE states were dropped in either seed (12,000 responses over 30 mid-range states leave every state two-sided). The slope sits within 1% of the true DGP value of -1.0 and the R^2 warning threshold (0.9) is never approached.

## Error by true-weight decile: the extremes are NOT worst in probability space

The rubric's prior — extreme states will be worst — holds in logit space but INVERTS in probability space, and the mechanism is worth stating because probability space is where the weights are used. The affine anchor map is fit only on the 30 mid-range anchors and extrapolated linearly in logit space to all 220 states, so logit-space error is roughly flat across deciles with a mild uptick at the two extremes. But expit's derivative shrinks toward zero near 0 and 1, so a given logit error compresses when mapped back: the lightest decile shows mean |err| 0.00035 and the heaviest 0.00355, while the worst decile is the upper-middle (decile 8, dw 0.586-0.766) at mean 0.01086, and the single worst state error (0.03872) sits in decile 7 (dw 0.370-0.576) where expit's derivative is largest. In logit space the ordering is the naive one: the extreme deciles are worst (decile 1 mean 0.0786, max 0.2618; decile 10 mean 0.0546, max 0.1919) against ~0.037-0.05 mid-range.

Pooled decile table (22 states per decile per seed; mean/median/max |err|, probability scale then logit scale):

| decile | true dw range | mean | median | max | logit mean | logit median | logit max |
|---|---|---|---|---|---|---|---|
| 1 | [0.003, 0.007] | 0.00035 | 0.00031 | 0.00107 | 0.0786 | 0.0656 | 0.2618 |
| 2 | [0.007, 0.016] | 0.00069 | 0.00057 | 0.00211 | 0.0616 | 0.0628 | 0.1952 |
| 3 | [0.017, 0.039] | 0.00125 | 0.00108 | 0.00382 | 0.0490 | 0.0490 | 0.1505 |
| 4 | [0.040, 0.089] | 0.00256 | 0.00194 | 0.00849 | 0.0448 | 0.0367 | 0.1582 |
| 5 | [0.092, 0.190] | 0.00528 | 0.00390 | 0.01300 | 0.0461 | 0.0375 | 0.1401 |
| 6 | [0.196, 0.361] | 0.00723 | 0.00663 | 0.01878 | 0.0377 | 0.0332 | 0.1077 |
| 7 | [0.370, 0.576] | 0.00908 | 0.00767 | 0.03872 | 0.0371 | 0.0313 | 0.1586 |
| 8 | [0.586, 0.766] | 0.01086 | 0.00939 | 0.03370 | 0.0505 | 0.0436 | 0.1502 |
| 9 | [0.773, 0.887] | 0.00640 | 0.00549 | 0.01742 | 0.0470 | 0.0350 | 0.1406 |
| 10 | [0.891, 0.950] | 0.00355 | 0.00262 | 0.00974 | 0.0546 | 0.0409 | 0.1919 |

## Runtime and memory

Wall clock per `estimate_dws` fit: 139.7s (seed 0) and 32.7s (seed 1); total script runtime 172.7s. The seed-0 fit is the same computation as seed 1 — the 4x gap is first-run overhead and machine load, not design size; a pre-study pilot of the identical fit measured 31-33s twice. `fit_pc` is ~99% of each fit. The halve-respondents contingency (rubric item 5: halve respondents, never states, if a fit projects past the 480s budget) is live code in the script — projection after seed 0 was ~282s for the 2-seed total, under the 480s budget, so it did not trigger; the scaling assumption it would use is fit time linear in PC rows, so halving respondents roughly halves the fit. Memory stays bounded: only one seed's data exists at a time (pc_df is ~26MB; the ~370MB float64 dense 210k x 219 design matrix is transient inside `fit_pc` and freed on return, and the study never rebuilds or duplicates it).
