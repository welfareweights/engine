# Coverage of bootstrap_dws intervals — Monte Carlo validation

**Verdict criterion (stated before results):** mean coverage across states within [0.88, 0.99] passes at this MC resolution. Nominal coverage is 0.95; the Monte Carlo binomial error at n_mc=60 is +/-5.5pp (1.96*sqrt(0.95*0.05/60)), so the band absorbs MC noise plus modest genuine undercoverage. Per-state coverage outside the band is flagged as a dip. The extreme states s000 and s007 are expected to be the worst performers: their PHE responses come back one-sided every rep, so estimate_dws drops them from the anchor stage and their dw is anchor-extrapolated rather than directly anchored.

**Design.** K=8 states with true dws evenly spaced in logit space over [0.005, 0.95] (simulate_true_dws defaults), held fixed across reps. Each MC rep simulates 600 PC respondents x 8 pairs (4800 rows, slope=-1.0, intercept=0.3) and 900 PHE respondents x 3 questions (2700 rows, sigma=0.8, all 8 states eligible), then runs bootstrap_dws with n_boot=100, level=0.95. Seeding: one master SeedSequence(20260716) spawned into 100 per-rep sequences, each spawned into independent pc/phe/boot streams — fully reproducible. Coverage of the percentile interval is the fraction of reps with lo <= true_dw <= hi; coverage of the normal interval uses dw +/- 1.9600*se from the bootstrap's own point estimate and replicate SD.

**Timing and n_mc decision (rubric #5).** The first rep (simulate + point estimate + 100-replicate bootstrap) took 21.3s, projecting 35.5 min for n_mc=100; the 12-minute trigger fired, so the study ran at n_mc=60 and the MC error widened to +/-5.5pp. Actual total wall clock: 10.9 min.

**Failure rates (rubric #4).** MC-rep-level failures (a whole rep raising in estimation or the bootstrap's own too-few-replicates guard): 0/60 (0.0%). Bootstrap-replicate-level failures inside successful reps (bootstrap_dws n_failed summed): 0/6000 (0.00%).

## Results (60 successful MC reps)

| state | true_dw | mean_point_dw | mean_boot_se | coverage_percentile | coverage_normal |
|---|---|---|---|---|---|
| s000 | 0.0050 | 0.0057 | 0.0015 | 0.883 | 0.917 |
| s001 | 0.0160 | 0.0176 | 0.0033 | 0.900 | 0.933 |
| s002 | 0.0502 | 0.0532 | 0.0062 | 0.917 | 0.933 |
| s003 | 0.1464 | 0.1511 | 0.0117 | 0.950 | 0.967 |
| s004 | 0.3575 | 0.3633 | 0.0205 | 0.883 | 0.883 |
| s005 | 0.6435 | 0.6377 | 0.0241 | 0.933 | 0.933 |
| s006 | 0.8542 | 0.8482 | 0.0182 | 0.950 | 0.967 |
| s007 | 0.9500 | 0.9470 | 0.0108 | 0.917 | 0.950 |
| **MEAN** | | | | **0.917** | **0.935** |

**Per-state dips.** Percentile interval: none outside [0.88, 0.99]. Normal interval: none outside [0.88, 0.99].

**Verdict.** Percentile interval: mean coverage 0.917 -> **PASS**. Normal interval (dw +/- 1.96se): mean coverage 0.935 -> **PASS**. Both are read against the [0.88, 0.99] band with +/-5.5pp MC error at n_mc=60.

Raw per-state numbers: `studies/results/coverage.csv`. Reproduce with `.venv/bin/python studies/coverage.py` (master seed 20260716).
