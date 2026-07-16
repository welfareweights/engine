# Roadmap

What this engine can and cannot claim today, and what comes next, in order.

## Done

- The three-stage Salomon et al. estimator (probit on paired comparisons, interval regression on population-health-equivalence responses, logit-space anchoring), with loud failure modes: separation, non-convergence, one-sided anchor states, and a degenerate anchor map all raise or warn instead of passing garbage through.
- Respondent-cluster bootstrap for uncertainty intervals, coverage-tested by Monte Carlo (`studies/RESULTS-coverage.md`).
- Validation without microdata: known-truth recovery tests, metamorphic invariance tests, a misspecification battery, link-robustness cross-checks, held-out fit diagnostics, and survey-design power curves. See `docs/VALIDATION.md` for the evidence in one place.

## Next

1. **Refit published weights from the original survey microdata.** The single most important step, and the one we cannot take alone: the GBD/GHE raw responses are not public. Data requests are in progress with the original study teams. Everything below is what we can do while we wait — and the synthetic validation above is designed so that, when microdata arrives, the only untested step is the data itself.
2. **Anchor-linearity diagnostic.** Our misspecification battery found the one violation that damages weight levels while evading every current diagnostic: curvature in the paired-comparison utility link. Planned: a curvature term in the anchor regression as a standard diagnostic, and PHE anchor states spread across the full severity range (the power study shows anchor width is also the cheapest precision margin).
3. **Multi-survey machinery.** The published pipeline pools multiple surveys with a cross-survey variance term in the back-transform; the engine currently runs single-survey (tau = 0). Becomes testable once more than one dataset exists.
4. **LLM-respondent surveys.** The instrument builder and parser already serve synthetic LLM respondents. Two separate uses, kept separate: stress-testing the estimation pipeline on realistic non-DGP responses, and studying how language models themselves value health states. Early pilots ran; a clean, larger re-run is queued behind the repo-facing work.
5. **New-state extension.** The point of the project: estimating weights for health and welfare states the current instruments miss. Requires the validated pipeline plus new survey data (see the companion platform at [welfareweights.com](https://welfareweights.com)).
6. **R companion.** A replication guide and port so the pipeline is usable from both ecosystems.

## Contributing

Issues and PRs welcome, especially: independent replication of the studies in `studies/`, holes in the validation argument, and pointers to accessible disability-weight microdata.
