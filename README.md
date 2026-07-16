# WelfareWeights Engine

[![tests](https://github.com/welfareweights/engine/actions/workflows/ci.yml/badge.svg)](https://github.com/welfareweights/engine/actions/workflows/ci.yml)

Open-source tooling to estimate disability weights from survey data. These are the valuations that turn health states into DALYs (disability-adjusted life years) for burden-of-disease work. Eventually we will extend that methodology to health and welfare states that are not yet well modeled.

This is the academic core of the [WelfareWeights](https://welfareweights.com) project. The companion survey platform (welfareweights.com) is managed separately; this repository is the freely reusable engine and its documentation.

## Why

DALYs compress an enormous range of human experience into a single number, and the disability weights underneath them are estimated from public surveys using a methodology that most researchers cannot easily reproduce or extend. The goal here is to democratize and open-source that methodology. I want transparent, reproducible, and open code, so weights can be replicated, scrutinized, and computed for new states. This will help people doing burden-of-disease accounting, policy analysis, and welfare measurement more broadly.

## Scope

1. **Replicate** the gold-standard approach (Salomon et al.) that derives disability weights from paired-comparison and valuation survey data.
2. **Open-source** it as a clean, well-documented library with user-friendly guides, so anyone can run survey data through it.
3. **Extend** it to estimate weights for states the current instruments miss.

## Validation

The original survey microdata is not public, so the estimator is validated the hard way instead: known-truth recovery from the assumed data-generating process, metamorphic invariance tests, loud-failure tests for every identification hazard we found in review, a link-robustness cross-check, held-out fit diagnostics, Monte Carlo coverage tests of the bootstrap intervals, a misspecification battery, and a full-scale run at the published GBD 2010 design size (220 states, 14,000 respondents: rank correlation 0.9997, mean error 0.005). The evidence, including the one documented blind spot, is collected in [docs/VALIDATION.md](docs/VALIDATION.md); the studies behind it are reproducible from [studies/](studies/) with recorded seeds.

## Status

Early and under active construction. The estimation pipeline is implemented and validated against synthetic data (see above) while we request the original survey data — the roadmap is in [ROADMAP.md](ROADMAP.md). Expect interfaces to change. Watch or star the repo to follow along.

## License

Two licenses, by artifact type:

- **Code** — [Apache License 2.0](LICENSE). Use it freely, including commercially; it carries an explicit patent grant.
- **Data, estimated weights, and documentation** — [Creative Commons Attribution 4.0](LICENSE-DATA) (CC-BY-4.0). Reuse freely with attribution.

Note that raw survey responses are governed by respondent consent and privacy rules, not by these licenses, and are not published here.

## Citation

If you use this work, please cite it via the "Cite this repository" button (see [CITATION.cff](CITATION.cff)). A versioned DOI will be added on first release.

