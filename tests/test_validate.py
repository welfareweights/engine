"""Held-out model-fit evidence: does the fitted response model predict
choices from respondents it never saw?

test_recovery.py checks the estimator against a known DGP truth, evidence
that only exists because true disability weights are known by construction
in synthetic data. This file's evidence needs no such truth: it checks that
holdout_fit's train/test split, per-module discrimination (log-likelihood
over a null base-rate model) and calibration (predicted probability vs.
empirical frequency) behave correctly on data generated from the assumed
DGP — the one test design here that also runs, unchanged, on real LLM or
human survey data where no ground truth exists.
"""

import numpy as np
import pandas as pd
import pytest

from welfareweights.anchor import fit_phe
from welfareweights.probit import fit_pc
from welfareweights.simulate import make_states, simulate_pc, simulate_phe, simulate_true_dws
from welfareweights.validate import calibration_slope, eval_pc, eval_phe, holdout_fit

K = 20
STATES = make_states(K)
TRUE_DWS = simulate_true_dws(K)
# Same mid-range restriction as test_recovery.py: PHE thresholds logit(1000/c)
# only bracket the latent distribution for states in this range; extreme
# states go one-sided (quasi-separation), a design weakness, not a bug here.
ANCHOR_STATES = [s for s, dw in zip(STATES, TRUE_DWS) if 0.05 <= dw <= 0.70]


@pytest.fixture(scope="module")
def pc_df():
    return simulate_pc(TRUE_DWS, STATES, n_respondents=2000, slope=-1.0, intercept=0.3, rng=101)


@pytest.fixture(scope="module")
def phe_df():
    return simulate_phe(
        TRUE_DWS, STATES, n_respondents=2500, sigma=0.8, anchor_states=ANCHOR_STATES, rng=202
    )


@pytest.fixture(scope="module")
def result(pc_df, phe_df):
    return holdout_fit(pc_df, phe_df, test_frac=0.3, rng=7)


def test_holdout_splits_respondents_not_rows(pc_df, phe_df, result):
    # Every row a held-out respondent contributed must appear in n_eval +
    # n_skipped, and the split is over the ID union: with pc n_respondents
    # (2000) < phe n_respondents (2500), some test ids fall outside pc_df
    # entirely and contribute zero PC rows, so PC row counts undercount the
    # 30% test fraction while PHE's don't.
    assert result.n_train_respondents + result.n_test_respondents == 2500
    assert result.n_test_respondents == round(2500 * 0.3)
    assert result.pc.n_eval + result.pc.n_skipped <= result.n_test_respondents * 15
    assert result.phe.n_eval + result.phe.n_skipped == result.n_test_respondents * 3


def test_pc_model_beats_null_by_a_clear_margin(result):
    # Observed: model -0.167 nats/response vs. null -0.693 (= log 0.5, the
    # null's base rate given random state pairing is close to symmetric).
    # 0.2 nats of margin is roughly a third of the observed gap and far
    # outside anything sampling noise at n_eval=9000 held-out rows could
    # produce -- this is a floor for "clearly better," not a tight bound.
    assert result.pc.n_skipped == 0
    assert result.pc.mean_loglik > result.pc.mean_loglik_null + 0.2


def test_phe_model_beats_null_by_a_clear_margin(result):
    # Observed: model -0.321 vs. null -0.690, a 0.37-nat gap. Same reasoning
    # as the PC test: 0.15 nats is well inside the observed margin and far
    # above sampling noise at n_eval=2250.
    assert result.phe.n_skipped == 0
    assert result.phe.mean_loglik > result.phe.mean_loglik_null + 0.15


def test_calibration_slope_near_one(result):
    # Observed slopes: PC 0.996, PHE 0.992. 0.15 gives >10x headroom over the
    # observed deviation from 1 -- generous enough to absorb the sampling
    # noise of a different rng draw while still catching a miscalibrated
    # model (e.g. a sign or scale error in the Phi(...) formulas would push
    # the slope towards 0 or negative, not shave a few percent off it).
    for module in (result.pc, result.phe):
        slope = calibration_slope(module.calibration)
        assert abs(slope - 1.0) < 0.15


def test_calibration_table_counts_partition_eval_rows(result):
    for module in (result.pc, result.phe):
        assert module.calibration["n"].sum() == module.n_eval


def test_pc_holdout_rows_naming_an_unknown_state_are_skipped():
    # Deterministic, isolated from the random respondent split: fit stage A
    # on 5 states only, then score 3 held-out rows, 2 of which name a state
    # the fit never saw.
    small_states = make_states(5)
    small_true = simulate_true_dws(5)
    small_pc = simulate_pc(small_true, small_states, n_respondents=200, rng=3)
    fit = fit_pc(small_pc)

    test_rows = pd.DataFrame(
        {
            "respondent_id": [999, 999, 999],
            "state_1": [small_states[0], "unknown_state", small_states[1]],
            "state_2": [small_states[1], small_states[2], "unknown_state"],
            "y": [1, 0, 1],
        }
    )
    scored = eval_pc(fit, test_rows, null_p=0.5)
    assert scored.n_skipped == 2
    assert scored.n_eval == 1


def test_phe_holdout_rows_naming_an_unknown_state_are_skipped():
    small_states = make_states(8)
    small_true = simulate_true_dws(8)
    small_anchor = [s for s, dw in zip(small_states, small_true) if 0.05 <= dw <= 0.70]
    small_phe = simulate_phe(
        small_true, small_states, n_respondents=400, sigma=0.8, anchor_states=small_anchor, rng=5
    )
    fit = fit_phe(small_phe)

    test_rows = pd.DataFrame(
        {
            "respondent_id": [1, 1, 1],
            "state": [small_anchor[0], "unknown_state", small_anchor[1]],
            "n_cases": [2000, 2000, 2000],
            "deaths": [1000, 1000, 1000],
            "y": [1, 0, 1],
        }
    )
    scored = eval_phe(fit, test_rows, null_p=0.5)
    assert scored.n_skipped == 1
    assert scored.n_eval == 2


def test_calibration_slope_needs_at_least_two_populated_bins():
    single_bin = pd.DataFrame(
        {"bin_lo": [0.0], "bin_hi": [0.1], "n": [5], "mean_predicted": [0.05], "empirical_freq": [0.04]}
    )
    with pytest.raises(ValueError, match="populated calibration bin"):
        calibration_slope(single_bin)
