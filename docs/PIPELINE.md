# Estimation pipeline — code map

How the code implements the GBD disability-weight methodology (Salomon et al.
2012/2015). The methodology itself, with source pointers and the seven open
replication decisions, is in `research/notes/methodology-core.md`; this doc
maps that methodology onto modules so a reader can go from stage to code.

## Stages

The estimator is three stages. A and B are independent regressions; C fuses
them onto the 0–1 scale.

| Stage | What | Module | Key function |
|------|------|--------|--------------|
| A | Probit on paired comparisons → health values on an arbitrary linear scale | `probit.py`, `design.py` | `fit_pc` |
| B | Interval-censored normal regression on PHE responses → death-anchored logit-scale weights | `anchor.py` | `fit_phe` |
| C | Linear map A→B scale, then integrate through inverse-logit → weights on [0, 1] | `rescale.py` | `fit_anchor_map`, `expected_expit` |
| — | Chain A→B→C | `pipeline.py` | `estimate_dws` |

Two design choices worth knowing:

- **Stage A design coding** (`design.build_pc_design`): for k states, a k−1
  column indicator matrix, one state omitted as reference (β = 0). Per
  comparison, chosen state +1, non-chosen −1, others 0, so Xβ = β_chosen −
  β_other. This is the whole trick that turns the choice data into a probit.
- **Stage B as a probit reduction** (`anchor.fit_phe`): the interval-censored
  logit-normal likelihood divides through by σ to become an ordinary probit of
  the choice on `[threshold, state dummies]`, from which σ = 1/coef_threshold
  and β_s = −coef_s/σ. Globally concave, so Newton converges reliably.
  `anchor.nll` gives the direct (β, σ) likelihood the test suite uses to
  confirm the reduction is the MLE.

## Why it's trustworthy without real microdata

No respondent-level survey data is public yet (`research/notes/
data-availability.md`), so the estimator cannot yet be validated by refitting
the published GBD weights from their own raw responses. Two things stand in
for that:

1. **Monte Carlo recovery** (`tests/test_recovery.py`): `simulate.py` generates
   PC/PHE data from the exact DGP the estimator assumes, with known true
   weights; the tests require the pipeline to recover them (rank correlation
   > 0.995, max error < 0.04 at the tested N). This validates the *code against
   the model* — the referee's "does it recover truth", not just "does it run".
2. **Output validation** (`../data/published/`): the pipeline's estimates are
   compared to the published WHO GHE2021 / GBD weight tables. This validates
   *plausibility of level*, not identification.

The published tables and lay descriptions live in the sibling `../data/`
directory, not in the engine, because they carry clinical health-state text
that is kept out of the engine's review surface (see `../../data/README.md`).
The engine's own tests use synthetic state labels and read none of it.

The two are complementary and neither replaces refitting from real microdata,
which stays blocked pending a data grant (see top-level `HUMAN-TODO.md`).

## Synthetic survey (LLM respondents)

`survey.py` builds instruments (assign PC pairs + PHE questions per respondent,
enforcing a connected comparison graph) and parses responses into the long
tables the estimator consumes — the same code path for human web respondents
or synthetic LLM respondents. `scripts/ingest_haiku_survey.py` takes the output
of a Haiku respondent run and pushes it through the pipeline, comparing the
estimated weights to the curated published targets. This exercises the whole
loop end to end on non-DGP data, and — because the respondents are LLMs — is
itself a small study of how an LLM population values health states.

## Data extraction

`../data/tools/extract_ghe2021_annex.py` parses the WHO GHE2021 DALY-methods
PDF into `../data/published/` (weight tables, the validation target) and
`../data/instruments/` (lay descriptions, the question text). It lives under
`data/`, not the engine, because it names disease categories. See
`../../data/README.md` for provenance and the known Table B↔C join limitation.
