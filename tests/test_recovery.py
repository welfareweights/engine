"""Monte Carlo recovery tests: simulate from the assumed DGP with known true
disability weights, fit, and require the truth back.

These are the engine's core correctness evidence while no survey microdata is
public: they validate the code against the model. Tolerances are set for the
sample sizes used (sampling error shrinks as N grows, so failures at these
tolerances indicate coding errors, not noise).
"""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize
from scipy.special import logit

from welfareweights.anchor import fit_phe, nll, thresholds
from welfareweights.pipeline import estimate_dws
from welfareweights.probit import fit_pc
from welfareweights.simulate import make_states, simulate_pc, simulate_phe, simulate_true_dws

K = 20
SLOPE, INTERCEPT = -1.0, 0.3
STATES = make_states(K)
TRUE_DWS = simulate_true_dws(K)
# Anchor subset: mid-range weights, where the PHE thresholds logit(1000/c) actually
# bracket the latent distribution. Extreme states give one-sided responses
# (quasi-separation) — a weakness of the real design, not of this test.
ANCHOR_STATES = [s for s, dw in zip(STATES, TRUE_DWS) if 0.05 <= dw <= 0.70]


@pytest.fixture(scope="module")
def pc_df():
    return simulate_pc(
        TRUE_DWS, STATES, n_respondents=3000, slope=SLOPE, intercept=INTERCEPT, rng=7
    )


@pytest.fixture(scope="module")
def phe_df():
    return simulate_phe(
        TRUE_DWS, STATES, n_respondents=4000, sigma=0.8, anchor_states=ANCHOR_STATES, rng=11
    )


def test_probit_recovers_theta_up_to_reference_shift(pc_df):
    fit = fit_pc(pc_df)
    theta = SLOPE * logit(TRUE_DWS) + INTERCEPT
    expected = theta - theta[STATES.index(fit.ref_state)]  # beta_ref = 0 normalization
    err = fit.beta.to_numpy() - expected
    # The reference state's own sampling error shifts ALL betas by a common
    # constant (beta = theta - theta_ref), and stage C's intercept absorbs any
    # location shift — so test the relative errors, demeaned.
    assert np.corrcoef(fit.beta.to_numpy(), expected)[0, 1] > 0.999
    assert np.max(np.abs(err - err.mean())) < 0.15  # ~2x the per-state SE at this N


def test_interval_regression_recovers_logit_dw_and_sigma(phe_df):
    fit = fit_phe(phe_df)
    true_logit = pd.Series(logit(TRUE_DWS), index=STATES)[fit.states]
    err = fit.logit_dw - true_logit
    assert np.max(np.abs(err.to_numpy())) < 0.25
    assert abs(fit.sigma - 0.8) < 0.08


def test_probit_reduction_is_the_mle(phe_df):
    """The production estimator (probit reduction) must match a direct
    minimization of the paper's (beta, log sigma) likelihood."""
    fit = fit_phe(phe_df)
    state_idx = phe_df["state"].map({s: i for i, s in enumerate(fit.states)}).to_numpy()
    t = thresholds(phe_df["n_cases"].to_numpy(), 1000)
    y = phe_df["y"].to_numpy()

    def objective(params):
        return nll(params[:-1], params[-1], state_idx, t, y)

    x0 = np.append(np.zeros(len(fit.states)), 0.0)
    direct = minimize(objective, x0, method="BFGS")
    # BFGS can flag "precision loss" at an already-converged point; the
    # convergence criterion that matters is a vanished gradient.
    assert np.max(np.abs(direct.jac)) < 0.1
    np.testing.assert_allclose(direct.x[:-1], fit.logit_dw.to_numpy(), atol=1e-3)
    np.testing.assert_allclose(np.exp(direct.x[-1]), fit.sigma, atol=1e-3)
    # And the reduction's solution must be at least as good a minimizer.
    reduction = np.append(fit.logit_dw.to_numpy(), np.log(fit.sigma))
    assert objective(reduction) <= direct.fun + 1e-4


def test_end_to_end_recovery(pc_df, phe_df):
    weights, diag = estimate_dws(pc_df, phe_df)
    est = weights["dw"].reindex(STATES).to_numpy()
    assert diag["anchor_map"].r_squared > 0.98
    assert np.corrcoef(est, TRUE_DWS)[0, 1] > 0.995
    assert np.max(np.abs(est - TRUE_DWS)) < 0.04


def test_reference_state_invariance(pc_df, phe_df):
    """Replication decision 2, verified numerically: the arbitrary choice of
    omitted reference state must not move the final weights."""
    w_a, _ = estimate_dws(pc_df, phe_df, ref_state=STATES[0])
    w_b, _ = estimate_dws(pc_df, phe_df, ref_state=STATES[7])
    np.testing.assert_allclose(w_a["dw"].to_numpy(), w_b["dw"].to_numpy(), atol=1e-5)


def test_disconnected_comparison_graph_is_refused():
    df = pd.DataFrame(
        {
            "respondent_id": [0, 0],
            "state_1": ["a", "c"],
            "state_2": ["b", "d"],  # {a,b} and {c,d} never compared
            "y": [1, 0],
        }
    )
    with pytest.raises(ValueError, match="disconnected"):
        fit_pc(df)


def test_expected_expit_matches_monte_carlo():
    from welfareweights.rescale import expected_expit
    from scipy.special import expit

    rng = np.random.default_rng(3)
    mu, tau = np.array([-2.0, 0.0, 1.5]), 0.7
    mc = expit(mu[:, None] + tau * rng.normal(size=(3, 2_000_000))).mean(axis=1)
    np.testing.assert_allclose(expected_expit(mu, tau), mc, atol=2e-3)
    np.testing.assert_allclose(expected_expit(mu, 0.0), expit(mu), atol=1e-12)
