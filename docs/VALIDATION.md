# Validation evidence

No respondent-level disability-weight survey data is public, so this engine cannot yet be validated the obvious way — refitting published weights from their own raw responses (data requests are in progress; see `ROADMAP.md`). This document collects what we validate instead, in increasing order of ambition: that the code implements the model, that the estimates are invariant to arbitrary choices, that failures are loud, that inference is honest, and that the estimator degrades gracefully when respondents violate the model. Every number below is reproducible from the committed tests and `studies/` scripts with their recorded seeds.

## 1. The code recovers known truth (tests/test_recovery.py, tests/test_survey.py)

Simulating from the exact data-generating process the estimator assumes (Thurstone probit for paired comparisons, interval-censored logit-normal for population-health-equivalence responses) with known true weights, the full pipeline recovers them: correlation > 0.995 and max error < 0.04 at the tested sample sizes, through both the raw-table path and the full instrument-assignment/answer-parsing round trip. The stage-B probit reduction is verified to be the exact MLE against a direct minimization of the paper's (beta, sigma) likelihood, and the back-transform integral is verified against Monte Carlo.

## 2. Arbitrary choices don't move the estimates (tests/test_metamorphic.py, tests/test_recovery.py)

The choice of reference state, the labeling of states, which side of a comparison a state is shown on (with the response flipped), the common scale of the deaths/cases figures (only the ratio enters the model), and duplicating the dataset all leave the final weights unchanged, at tolerances between 1e-5 and 1e-9.

## 3. Failure modes are loud (tests/test_failure_modes.py)

Separation in either probit stage, non-convergence, one-sided anchor states, a degenerate or orientation-reversed anchor map, a disconnected comparison graph, contradictory deaths figures, and self-paired comparison rows all raise or warn — none passes garbage through silently. This matters because we demonstrated the silent version during review: an unguarded run with weakly identified anchor states produced weights with max error 0.208 while still showing rank correlation 0.99.

## 4. Results are not an artifact of the link function (tests/test_crosscheck.py)

Fitting the identical paired-comparison design with a logistic link (Bradley-Terry) instead of the probit gives Spearman rank correlation 1.000 and standardized-coefficient correlation 0.99998 — the normal-link assumption, the least defensible modeling choice a priori, is not driving the results.

## 5. The response model fits out of sample (src/welfareweights/validate.py)

Holding out 30% of respondents, the fitted model predicts their choices far better than base rates (held-out log-likelihood per response: paired comparisons -0.167 vs. -0.693 null; anchoring questions -0.321 vs. -0.690) with calibration slopes of 0.996 and 0.993 against the ideal 1.0. This is the one check that transfers unchanged to real human or LLM respondents.

## 6. Uncertainty intervals have their stated coverage (studies/RESULTS-coverage.md)

The respondent-cluster bootstrap's 95% percentile intervals achieve 0.917 mean coverage across states (normal intervals: 0.935) in a 60-replication Monte Carlo — inside the pre-stated [0.88, 0.99] acceptance band at that resolution, with zero failed replicates in 6,000 bootstrap fits.

## 7. Rankings survive every misspecification tested; levels survive all but one (studies/RESULTS-misspecification.md)

With respondents violating the model one way at a time — scale heterogeneity, within-respondent correlation, 5-15% random lapses on either module, position bias, a logistic error law — Spearman rank correlation stays at 1.000 and max level error stays within 0.012 of the 0.019 baseline, because these violations produce uniform attenuation or symmetric noise that the data-estimated anchor map absorbs. The exception is curvature in the comparison-utility link: at the stronger severity tested it produces level errors up to 0.161 while rank correlation stays at 0.9998 and anchor R² stays at 0.985 — high enough to evade the R² warning. This is the engine's documented blind spot: levels depend on a linearity assumption that current diagnostics do not certify. The mitigation (an explicit curvature diagnostic, and anchor states spread across the severity range) is `ROADMAP.md` item 2.

## 8. Precision scales as theory predicts, and the design margins are quantified (studies/RESULTS-power.md)

Mean error falls with respondent count at a fitted log-log slope of -0.517 (r² 0.984) — the textbook root-N rate. The cheapest precision margin is the anchor-set width, which cuts mean error 44% at zero additional responses; 500 respondents at 10 pairs each meet a 0.05 max-error bar, 2,000 meet 0.02. The cheap precision proxy agrees with the full bootstrap to 1% where both were run.

## 9. The published survey scale is comfortably sufficient (studies/RESULTS-gbd-scale.md)

At the GBD 2010 design size — 220 states, 14,000 respondents, 15 pairs each, 30 anchor states — the pipeline recovers rank correlation 0.9997 with mean error 0.0047 and max error 0.039, and the anchor extrapolation behaves as the logit geometry predicts (largest probability-space errors in the upper-middle deciles, not the extremes).

## What this does not show

Synthetic validation proves the code against the model and maps the model's robustness; it cannot prove the model against real respondents. That requires the original microdata (being requested) or new survey data (the project's larger goal). Until then, the two claims we make are deliberately limited: the implementation is correct, and the estimator's failure conditions are characterized and mostly guarded.
