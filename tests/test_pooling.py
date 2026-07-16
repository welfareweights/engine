"""Multi-survey pooling: reduction to the single-survey estimator, recovery
across heterogeneous surveys, a positive tau when surveys genuinely disagree,
order invariance, and no silent dropping of partially covered states.

Each survey gets its own slope/intercept/sigma because the affine link and
the response noise are survey-specific in the real design; the pooled scale
must come out right anyway, since stage C re-anchors each survey before
pooling.
"""

import warnings

import numpy as np
import pytest
from scipy.special import expit, logit

from welfareweights.pipeline import estimate_dws
from welfareweights.pooling import estimate_dws_pooled
from welfareweights.simulate import make_states, simulate_pc, simulate_phe, simulate_true_dws

K = 12
STATES = make_states(K)
TRUE_DWS = simulate_true_dws(K, low=0.01, high=0.90)
ANCHOR_STATES = [s for s, dw in zip(STATES, TRUE_DWS) if 0.05 <= dw <= 0.70]
# Survey-specific DGP parameters: different affine links and PHE noise.
SURVEY_PARAMS = [
    {"slope": -0.8, "intercept": 0.0, "sigma": 0.7},
    {"slope": -1.0, "intercept": 0.3, "sigma": 0.8},
    {"slope": -1.3, "intercept": -0.2, "sigma": 1.0},
]


def _survey(true_dws, states, params, seed, n_pc=800, n_phe=1200):
    anchors = [s for s, dw in zip(states, true_dws) if 0.05 <= dw <= 0.70]
    pc = simulate_pc(
        true_dws, states, n_pc, slope=params["slope"], intercept=params["intercept"], rng=seed
    )
    phe = simulate_phe(
        true_dws, states, n_phe, sigma=params["sigma"], anchor_states=anchors, rng=seed + 500
    )
    return pc, phe


def test_single_survey_reduces_to_estimate_dws():
    sv = _survey(TRUE_DWS, STATES, SURVEY_PARAMS[1], seed=11)
    pooled, diag = estimate_dws_pooled([sv])
    single, _ = estimate_dws(*sv)
    np.testing.assert_allclose(
        pooled["dw"].reindex(STATES).to_numpy(),
        single["dw"].reindex(STATES).to_numpy(),
        atol=1e-12,  # exact reduction: n_surveys=1 forces tau=0, expected_expit(mu,0)=expit(mu)
    )
    assert (pooled["tau"] == 0.0).all()
    assert (pooled["n_surveys"] == 1).all()


def test_multi_survey_recovery_beats_median_single_survey():
    surveys = [_survey(TRUE_DWS, STATES, p, seed=100 + i) for i, p in enumerate(SURVEY_PARAMS)]
    pooled, diag = estimate_dws_pooled(surveys)
    pooled_err = np.max(np.abs(pooled["dw"].reindex(STATES).to_numpy() - TRUE_DWS))
    single_errs = [
        np.max(np.abs(w["dw"].reindex(STATES).to_numpy() - TRUE_DWS))
        for w, _ in diag["per_survey"]
    ]
    # Averaging three independent anchored estimates should not do worse than
    # the middle one; small epsilon tolerates the nonlinearity of max().
    assert pooled_err <= np.median(single_errs) + 0.005
    assert np.corrcoef(pooled["dw"].reindex(STATES).to_numpy(), TRUE_DWS)[0, 1] > 0.99


def test_tau_positive_when_surveys_genuinely_disagree():
    """Surveys sampling populations with genuinely different valuations
    (true logit weights shifted +/-0.4) must yield a positive cross-survey
    tau, and dw must then differ from expit(mean) in the direction Jensen
    predicts (toward 1/2)."""
    shifts = (-0.4, 0.0, 0.4)
    surveys = [
        _survey(expit(logit(TRUE_DWS) + sh), STATES, SURVEY_PARAMS[i], seed=200 + i)
        for i, sh in enumerate(shifts)
    ]
    pooled, _ = estimate_dws_pooled(surveys)
    assert pooled["tau"].median() > 0.15  # disagreement is real and detected
    plugin = expit(pooled["logit_dw"].to_numpy())
    toward_half = np.where(plugin < 0.5, pooled["dw"] >= plugin, pooled["dw"] <= plugin)
    assert toward_half.all()


def test_survey_order_invariance():
    surveys = [_survey(TRUE_DWS, STATES, p, seed=300 + i) for i, p in enumerate(SURVEY_PARAMS)]
    a, _ = estimate_dws_pooled(surveys)
    b, _ = estimate_dws_pooled(list(reversed(surveys)))
    np.testing.assert_allclose(a["dw"].to_numpy(), b["dw"].to_numpy(), atol=1e-12)
    np.testing.assert_allclose(a["tau"].to_numpy(), b["tau"].to_numpy(), atol=1e-12)


def test_partial_state_coverage_pooled_not_dropped():
    full = _survey(TRUE_DWS, STATES, SURVEY_PARAMS[0], seed=400)
    sub_states, sub_true = STATES[:8], TRUE_DWS[:8]
    partial = _survey(sub_true, sub_states, SURVEY_PARAMS[1], seed=401)
    pooled, _ = estimate_dws_pooled([full, partial])
    assert list(pooled.index) == STATES  # union, nothing dropped
    assert (pooled.loc[sub_states, "n_surveys"] == 2).all()
    assert (pooled.loc[STATES[8:], "n_surveys"] == 1).all()
    assert (pooled.loc[STATES[8:], "tau"] == 0.0).all()


def test_empty_list_and_tau_passthrough_raise():
    with pytest.raises(ValueError, match="at least one survey"):
        estimate_dws_pooled([])
    sv = _survey(TRUE_DWS, STATES, SURVEY_PARAMS[0], seed=500)
    with pytest.raises(ValueError, match="estimated across surveys"):
        estimate_dws_pooled([sv], tau=0.5)
