# Changelog

## 0.1.0 (2026-07-16)

First release. The three-stage GBD disability-weight estimator (Salomon et al. 2012 replication): probit on paired comparisons, interval regression on population-health-equivalence responses, logit-space anchoring with Gauss-Hermite back-transform. Identification hazards raise or warn instead of passing garbage (separation, non-convergence, one-sided anchors, degenerate or bent anchor maps). Respondent-cluster bootstrap for uncertainty (coverage-tested), multi-survey pooling with cross-survey tau, held-out fit diagnostics, survey instrument builder and LLM-robust answer parsing. Validation without microdata: known-truth recovery, metamorphic invariances, link-robustness, misspecification battery, power curves, and recovery at the published GBD 2010 design scale; see `docs/VALIDATION.md` and the methods guide (`docs/methods.pdf`).
