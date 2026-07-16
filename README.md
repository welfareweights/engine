# WelfareWeights Engine

Open-source tooling to estimate **disability weights** from survey data — the valuations that turn health states into DALYs (disability-adjusted life years) for burden-of-disease work — and to extend that methodology to health and welfare states that are not yet well modeled.

This is the academic core of the [WelfareWeights](https://welfareweights.com) project. The companion survey platform (welfareweights.com) is managed separately; this repository is the freely reusable engine and its documentation.

## Why

DALYs compress an enormous range of human experience into a single number, and the disability weights underneath them are estimated from public surveys using a methodology that most researchers cannot easily reproduce or extend. The goal here is to make that methodology transparent, reproducible, and open — so weights can be replicated, scrutinized, and computed for new states — with uses spanning burden-of-disease accounting, policy analysis, and welfare measurement more broadly.

## Scope

1. **Replicate** the gold-standard approach (Salomon et al.) that derives disability weights from paired-comparison and valuation survey data.
2. **Open-source** it as a clean, well-documented library with user-friendly guides, so anyone can run survey data through it.
3. **Extend** it to estimate weights for states the current instruments miss.

## Status

Early and under active construction — the methodology and the language choice (R and/or Python) are being set up now. Expect the structure and interfaces to change. Watch or star the repo to follow along.

## License

Two licenses, by artifact type:

- **Code** — [Apache License 2.0](LICENSE). Use it freely, including commercially; it carries an explicit patent grant.
- **Data, estimated weights, and documentation** — [Creative Commons Attribution 4.0](LICENSE-DATA) (CC-BY-4.0). Reuse freely with attribution.

Note that raw survey responses are governed by respondent consent and privacy rules, not by these licenses, and are not published here.

## Citation

If you use this work, please cite it via the "Cite this repository" button (see [CITATION.cff](CITATION.cff)). A versioned DOI will be added on first release.

## Links

- Project site: https://welfareweights.com
- Companion platform (managed): `welfareweights/platform`
