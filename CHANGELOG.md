# Changelog

## Unreleased

- Fixed: `bootstrap_dws` published standard errors roughly 30% too small — and correspondingly too-narrow intervals — whenever the input tables carried duplicate index labels, e.g. batches combined with `pd.concat` without `ignore_index=True`; point estimates were unchanged and no warning fired, so the corruption was silent. Resampling now selects rows by position, immune to index labels; output on unique-index input is unchanged (bit-identical at a fixed seed).
- Added: per-state support floor in `bootstrap_dws`. A state informed by fewer than `min_state_respondents` (default 10) paired-comparison respondents, or estimable in fewer than half the bootstrap replicates (`min_state_reps`), previously published a spuriously tight interval with no warning (a 1-respondent state's interval could be tighter than every well-supported state's); it now publishes NaN se/lo/hi with a warning naming each state and the floor it failed, and keeps the point estimate. New `weights` columns: `n_resp_pc`, `n_resp_phe`, `supported`.

## 0.1.0 (2026-07-16)

First release. The three-stage GBD disability-weight estimator (Salomon et al. 2012 replication): probit on paired comparisons, interval regression on population-health-equivalence responses, logit-space anchoring with Gauss-Hermite back-transform. Identification hazards raise or warn instead of passing garbage (separation, non-convergence, one-sided anchors, degenerate or bent anchor maps). Respondent-cluster bootstrap for uncertainty (coverage-tested), multi-survey pooling with cross-survey tau, held-out fit diagnostics, survey instrument builder and LLM-robust answer parsing. Validation without microdata: known-truth recovery, metamorphic invariances, link-robustness, misspecification battery, power curves, and recovery at the published GBD 2010 design scale; see `docs/VALIDATION.md` and the methods guide (`docs/methods.pdf`).
