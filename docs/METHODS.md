# The method, from survey answers to weights

This is the auditing document: it explains what the estimator does and why, at the level of someone who took econometrics in grad school years ago, or someone fluent in math and code who never took it. It is deliberately not written for a journal referee — the referee-facing material is the test suite and `VALIDATION.md`. Code pointers throughout refer to `src/welfareweights/`; the stage-to-code map is `PIPELINE.md`.

Notation used throughout: Phi is the standard normal CDF; logit(p) = log(p/(1-p)) maps probabilities (0,1) onto the whole real line; expit is its inverse. "MLE" is maximum likelihood estimation: pick the parameter values under which the observed data was most probable.

## 1. The problem

A disability weight is a number between 0 and 1 that says how bad a year lived with a health state is: 0 is full health, 1 is as bad as being dead. Burden-of-disease accounting (DALYs) multiplies time lived with a condition by its weight, so every DALY figure you have ever seen has a table of these weights underneath it.

The weights come from surveys, and the surveys deliberately do NOT ask "rate this condition from 0 to 1" — people are bad at that. They ask two kinds of easier questions, and the estimation problem is to reassemble the 0-to-1 scale from the answers. This engine implements the reassembly used for the GBD weights (Salomon et al. 2012/2015), in three stages.

## 2. The two question types

**Paired comparison (PC).** The respondent reads lay descriptions of two health states and answers one thing (template in `survey.py::pc_question_text`):

> Two people have the same life expectancy. Assume each condition lasts for the rest of their life.
> Person A [description of state 1]
> Person B [description of state 2]
> Who is healthier overall — A or B?

Each respondent answers a handful of these with randomly assigned pairs. PC answers are informative about the ORDER and SPACING of states relative to each other, but they contain no information about where the whole scale sits — every answer would be identical if all states were twice as bad, or uniformly milder.

**Population health equivalence (PHE).** The question that pins the scale (template in `survey.py::phe_question_text`):

> Two health programs cost the same. Which produces the greater health benefit?
> Program 1 prevents 1000 people from dying.
> Program 2 prevents [c] people from having [state s] for the rest of their life.
> Which program — 1 or 2?

Here c varies (the GBD instrument uses 1500, 2000, 3000, 5000, 10000). If you value averting one death at 1 and averting a lifetime case of state s at its weight DW_s, program 1 is better exactly when 1000 > c·DW_s, i.e. when DW_s < 1000/c. So each answer reveals which side of a known threshold the respondent's valuation falls on — and because the trade is against DEATHS, the threshold is expressed on the 0-to-1 scale we ultimately want, with death = 1 built in.

One convention to swallow before going further: reading the choice as "1000 vs c·DW_s" treats averting a death and averting a lifetime condition as covering the same person-time. That is the DALY convention of the instrument this replicates, inherited here, not introduced here.

## 3. Stage A: paired comparisons → a relative scale (`probit.py`, `design.py`)

**The model.** Give every state i a latent "healthiness" number theta_i. When a respondent compares states i and j, they don't perceive theta exactly — they perceive it with noise: H_i ~ Normal(theta_i, sigma²), independently for each state in the pair, and they answer "i is healthier" iff H_i > H_j. Then

    P(answer "i healthier") = Phi( (theta_i − theta_j) / sqrt(2·sigma²) )

This is Thurstone's 1927 comparative-judgment model. Intuition: states far apart in theta produce near-unanimous answers; states close together produce closer to 50/50. The observed disagreement RATES are exactly what identifies the spacings.

**Two things the data cannot tell you,** which is what econometricians mean by an identification problem: adding a constant to every theta changes nothing (only differences enter), and multiplying all theta and sigma by the same factor changes nothing (only the ratio enters). So we normalize by convention: set 2·sigma² = 1, and fix one arbitrary reference state's theta to 0. Both choices are undone later — the test suite proves numerically that the final weights don't depend on which reference state you pick (`test_recovery.py::test_reference_state_invariance`), and stage C exists precisely to replace the arbitrary location/scale with a meaningful one.

**Why this is "just a probit".** A probit regression models P(y=1) = Phi(x'beta) and finds beta by MLE. Build one row per comparison: a vector that is +1 in the column of the first-listed state, −1 in the column of the second, 0 elsewhere, with the reference state's column dropped. Then x'beta = beta_first − beta_second, which is exactly the model above with beta = theta. So standard, well-understood probit machinery — with a globally concave log-likelihood, meaning Newton's method reliably finds THE maximum — estimates all the thetas at once from all comparisons pooled (`design.py::build_pc_design`, `probit.py::fit_pc`).

**What has to hold, and what the code does when it doesn't.** Think of states as nodes and comparisons as edges. You learn relative positions only within a connected graph — two islands of states never compared, even indirectly, have no estimable relative scale, so estimation refuses to run on a disconnected graph rather than returning something arbitrary. And a state that wins (or loses) every single comparison it appears in is like an undefeated team in a league table: the data only says "better than everything it met", the likelihood keeps improving as its theta grows without bound, and there is no finite estimate. The code detects this (separation) and raises instead of returning a huge garbage number, because that number would otherwise poison stage C.

## 4. Stage B: death-anchoring questions → absolute positions for some states (`anchor.py`)

**The model.** From section 2, a PHE answer reveals whether the respondent's valuation of state s is below or above 1000/c. Model the respondent's perceived weight on the logit scale: their draw is L = b_s + eps with eps ~ Normal(0, sigma²), where b_s = the population logit-weight of state s and eps is honest-to-goodness disagreement between respondents. Choosing program 1 (avert deaths) reveals L < t where t = logit(1000/c).

Each answer is therefore a "censored" observation: you never see L, only which side of a known threshold it fell on. The likelihood of an answer is Phi((t − b_s)/sigma) for one choice and its complement for the other. With five values of c the thresholds take five values, from logit(1/10) ≈ −2.20 up to logit(2/3) ≈ 0.69, and the observed choice frequencies across those thresholds trace out where each state's distribution sits — like locating a distribution by asking many people "is it below this line?" at a few different lines.

**A trick worth seeing once.** Divide through by sigma inside the Phi:

    P(choose deaths program) = Phi( t·(1/sigma) + Σ_s d_s·(−b_s/sigma) )

where d_s is a dummy for the state asked about. The right-hand side is linear in observables (the threshold value t and the state dummies), so this censored-regression problem IS an ordinary probit of the binary choice on [t, dummies] with no intercept. Fit that probit, then read off sigma = 1/coef_t and b_s = −coef_s·sigma. This is not an approximation — the test suite verifies it against brute-force minimization of the original likelihood (`test_recovery.py::test_probit_reduction_is_the_mle`). The payoff is the same as in stage A: a globally concave problem with dependable convergence.

**The limits of five thresholds.** A state so mild (or so severe) that its entire perceived distribution sits on one side of ALL the thresholds generates unanimous answers, and unanimity is the separation problem again: the estimate diverges. This is why the instrument only asks PHE questions about mid-severity states, and why the code raises on one-sided states (or, at the pipeline level, drops them from the anchoring with a warning). It's also why the ANCHOR SET design matters so much — see the power study.

## 5. Stage C: weld the two scales together (`rescale.py`)

Stage A produced beta for every state on an arbitrary linear scale. Stage B produced b_s = logit(weight) — an absolute, death-anchored position — but only for the anchor states. If the two measurements are consistent, they must be related by a line: beta ≈ slope·logit_dw + intercept, with slope negative (healthier = lower weight).

So: over the states measured by BOTH stages, run ordinary least squares of the stage-A betas on the stage-B logit-weights, then invert the fitted line to convert EVERY state's beta into a logit-weight, then expit back to (0,1). That's the whole stage. The R² of that regression is the single most informative diagnostic in the pipeline: it measures whether the relative scale and the absolute scale actually agree, and the pipeline warns when it's low.

Two subtleties:

The line is fit on estimates, not truth, so a wildly wrong anchor point (e.g. a nearly-one-sided state that slipped through) can lever the whole line and bias every weight at once. This is exactly the failure the review demonstrated (max error 0.208, silently) and why the gates in stage B exist. An optional variant weights the regression by each anchor's precision.

The back-transform averages correctly rather than plugging in: when the logit-scale estimate has spread tau around mu (as it does when pooling several surveys), the reported weight is E[expit(Normal(mu, tau²))], not expit(mu) — the mean of a nonlinear transform is not the transform of the mean (Jensen's inequality; the difference is largest near 0 and 1). The engine computes that expectation by Gauss–Hermite quadrature, a deterministic 64-point integral, rather than by simulation. With a single survey tau = 0 and this reduces to plain expit.

## 6. Uncertainty (`inference.py`)

Each respondent answers many questions, and their answers share that respondent's quirks, so treating every row as independent overstates how much information you have and makes the standard errors too small. The three-stage structure also makes textbook standard-error formulas awkward: errors from stage A and B both flow through the fitted anchor line and then through a nonlinear transform.

The engine uses the standard blunt instrument that handles all of this at once: the cluster bootstrap. Resample RESPONDENTS (not rows) with replacement — a drawn respondent brings all their PC and PHE answers with them — rerun the entire three-stage pipeline on each resample, and read the spread of the resulting weights. The 2.5th and 97.5th percentiles across resamples form the interval. Resamples that fail estimation (a resample can lose a state or go one-sided) are counted and reported, not silently retried; many failures mean the design is too fragile at that sample size for intervals to mean much.

"Does a procedure that claims 95% actually contain the truth 95% of the time?" is an empirical question, and we test it: simulating many surveys from known truth, the intervals cover at 0.92 (percentile) and 0.94 (normal-approximation) against a pre-stated acceptance band — `studies/RESULTS-coverage.md`.

## 7. What must be true for this to work

Assumptions, in decreasing order of how much they should worry you, with the evidence on each:

1. **The two scales are related by a straight line** (stage C's regression). This is the load-bearing wall. Our misspecification battery bent the true relationship and found the one blind spot in the pipeline: curvature damages weight LEVELS (errors up to 0.16 at the severity tested) while leaving rankings intact and keeping R² high enough to evade the warning. If you audit one thing, audit the anchor scatter for curvature. Mitigation is `ROADMAP.md` item 2.
2. **Respondents share a common valuation scale up to noise.** Violations we simulated — person-specific scales, within-person correlation, 5–15% of answers being coin flips, first-position bias — moved final weights barely or not at all, because they produce uniform attenuation or symmetric noise that the fitted anchor line absorbs. Rankings survived everything we threw at the estimator (`studies/RESULTS-misspecification.md`).
3. **The noise is normal.** Swapping in a logistic error law changed essentially nothing (`tests/test_crosscheck.py`: rank correlation 1.000 between probit and Bradley–Terry fits). Link choice is not driving results.
4. **The death-vs-lifetime-condition trade is understood as stated.** Untestable from inside the data; it is the instrument's convention (section 2). Real respondents may also just refuse to trade against death — our early LLM-respondent pilots suggest exactly that failure, which is a property of respondents, not of this estimator, but it caps what PHE-style anchoring can deliver.

## 8. How to audit this yourself

Start by running the worked example: `.venv/bin/python examples/quickstart.py` simulates a small survey with known true weights, runs the full pipeline plus bootstrap, and prints truth against estimates so you can see the machine work end to end in under a minute.

Then the fast checks, in rising order of effort:

- `pytest -q` runs the whole suite (~1 minute). What each file proves: `test_recovery.py` — the code recovers known truth from the assumed model, and the stage-B shortcut is exactly the MLE; `test_metamorphic.py` — relabeling states, swapping presentation order, rescaling deaths/cases, and duplicating data all leave weights unchanged; `test_failure_modes.py` — every identification hazard raises or warns rather than passing garbage; `test_crosscheck.py` — results aren't a normal-distribution artifact; `test_parsers.py` — the free-text answer parsing contract, including the prose traps; `test_validate.py` — the held-out fit machinery.
- The studies in `studies/` each rerun in minutes with recorded seeds; the reports state their designs so you can attack them (change a knob in `simulate.py`, add a violation we didn't think of, try to construct data that fools the gates — we document one construction that does).
- Read the estimator itself: the five stage modules total well under a thousand lines, and `PIPELINE.md` maps stage to file. The single most auditable property of this codebase is that the estimation logic is small.

The honest boundary of all of it: everything above validates the code against the model and probes the model's robustness. Whether real survey respondents follow anything like the model is a question only real microdata can answer — getting it is `ROADMAP.md` item 1 — and `validate.py`'s held-out checks are built so the same audit transfers to that data the day it arrives.

## References

Salomon et al. (2012), "Common values in assessing health outcomes from disease and injury: disability weights measurement study for the Global Burden of Disease Study 2010", The Lancet 380:2129-43, and its 2015 update in The Lancet Global Health. The estimation methodology this engine replicates is described in their supplementary appendices.
