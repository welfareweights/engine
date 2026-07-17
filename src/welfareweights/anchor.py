"""Stage B: interval regression on PHE responses -> logit-scale disability weights.

Model (Salomon 2012 supplement): the respondent's perceived logit-weight of
state s is L = beta_s + eps, eps ~ N(0, sigma^2). Choosing Program 1
(prevent `deaths` fatal cases) over Program 2 (prevent c cases of state s)
reveals deaths > c * DW_perceived, i.e. L below the threshold
t = logit(deaths / c); choosing Program 2 reveals L above t. Each response
is therefore interval-censored and the likelihood is

    y=1 (deaths program):   Phi((t - beta_s) / sigma)
    y=0 (nonfatal program): 1 - Phi((t - beta_s) / sigma)

Estimation detail: dividing through by sigma shows this IS a plain probit
of y on [t, state dummies] with no constant —
P(y=1) = Phi(t * (1/sigma) + sum_s d_s * (-beta_s / sigma)) — so the exact
MLE for (beta, sigma) is recovered from that probit's coefficients:
sigma = 1 / coef_t and beta_s = -coef_s / coef_t. The probit reduction is
the estimator here (globally concave likelihood, reliable Newton
convergence); `nll` exposes the direct (beta, log sigma) negative
log-likelihood so tests can verify the reduction against a direct
minimization.

Identification: with binary censoring only, a state whose weight puts every
threshold on the same side of its distribution yields one-sided responses
and a divergent beta_s (quasi-separation). fit_phe refuses to run on
one-sided states (their MLE diverges) and on a non-converged probit;
estimate_dws pre-drops one-sided states from the anchor stage with a
warning, matching the design intent that PHE anchors only states the
thresholds actually bracket.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import logit
from scipy.stats import norm

from welfareweights.checks import check_phe_df


@dataclass
class PHEFit:
    """Stage-B output: death-anchored disability weights in logit space."""

    states: list[str]
    logit_dw: pd.Series  # beta_s, indexed by state
    # Delta-method SE of beta_s = -coef_s / coef_t from the probit-reduction
    # covariance. Model-based (independent responses); used to precision-weight
    # the anchor map, not for final inference — that is inference.bootstrap_dws.
    logit_dw_se: pd.Series
    sigma: float
    response_counts: pd.DataFrame  # per state: n_deaths_chosen, n_cases_chosen
    n_obs: int
    result: object  # statsmodels probit-reduction result


def thresholds(n_cases: np.ndarray, deaths: int | np.ndarray) -> np.ndarray:
    """Censoring threshold t = logit(deaths / c) per response.

    deaths may be a scalar or a per-response array (broadcast against n_cases).
    """
    ratio = np.asarray(deaths, dtype=float) / np.asarray(n_cases, dtype=float)
    # All-based guard, deliberately (audit finding F4): np.all is False
    # whenever any element is NaN, so missing n_cases/deaths fail HERE, in the
    # function written to validate the ratio, instead of sailing through an
    # any-based check into statsmodels (fit path) or into a silently-NaN
    # mean_loglik (validate.eval_phe path). Empty input passes vacuously —
    # correct, since emptiness is caught earlier by the front door
    # (welfareweights.checks) with a sharper message.
    if not np.all((ratio > 0.0) & (ratio < 1.0)):
        raise ValueError(
            "threshold ratio deaths/n_cases must be finite and lie strictly in (0, 1) for "
            "every response (n_cases must exceed deaths); check n_cases and deaths for "
            "missing, non-positive, or contradictory values"
        )
    return logit(ratio)


def fit_phe(phe_df: pd.DataFrame, deaths: int | None = None) -> PHEFit:
    """Fit stage B by the exact probit reduction (see module docstring).

    deaths: fatal cases prevented by Program 1. If phe_df carries a per-row
    `deaths` column (the assignment and simulator write one), that column is
    used and a contradictory argument raises — the threshold must be computed
    from the deaths figure the respondent actually saw. Without the column,
    the argument is used, defaulting to the GBD instrument's 1000.

    phe_df is validated up front (welfareweights.checks): malformed input
    raises ValueError naming the frame, column, and cause.
    """
    check_phe_df(phe_df)
    states = sorted(set(phe_df["state"]))

    if "deaths" in phe_df.columns:
        d = phe_df["deaths"].to_numpy()
        if deaths is not None and not np.all(d == deaths):
            raise ValueError(
                "deaths argument contradicts the per-row deaths column; "
                "thresholds must use the deaths figure shown to respondents"
            )
    else:
        d = 1000 if deaths is None else deaths

    y_by_state = phe_df.groupby("state")["y"]
    n1, ntot = y_by_state.sum(), y_by_state.size()
    one_sided = sorted(n1.index[(n1 == 0) | (n1 == ntot)])
    if one_sided:
        raise ValueError(
            f"one-sided PHE state(s) {one_sided}: every response favors the same program, "
            "so their logit weights diverge — drop them (estimate_dws does) or collect more data"
        )

    t = thresholds(phe_df["n_cases"].to_numpy(), d)
    dummies = pd.get_dummies(phe_df["state"], dtype=float)[states]
    X = pd.concat([pd.Series(t, name="_threshold", index=phe_df.index), dummies], axis=1)
    y = phe_df["y"].astype(int)
    res = sm.Probit(y, X).fit(disp=0, method="newton")
    if not res.mle_retvals.get("converged", False):
        raise ValueError(
            "stage-B probit did not converge; likely near-separation in the PHE responses"
        )

    inv_sigma = res.params["_threshold"]
    if inv_sigma <= 0:
        raise ValueError(
            "estimated 1/sigma is non-positive; responses do not follow the censoring model"
        )
    sigma = 1.0 / inv_sigma
    logit_dw = pd.Series(
        -res.params[states].to_numpy() * sigma, index=pd.Index(states, name="state")
    )

    # Delta method for beta_s = -p_s / p_t: gradient (-1/p_t, p_s/p_t^2) wrt (p_s, p_t).
    V = res.cov_params()
    p_t = inv_sigma
    se = {}
    for s in states:
        p_s = res.params[s]
        g_s, g_t = -1.0 / p_t, p_s / p_t**2
        var = (
            g_s**2 * V.loc[s, s]
            + 2.0 * g_s * g_t * V.loc[s, "_threshold"]
            + g_t**2 * V.loc["_threshold", "_threshold"]
        )
        se[s] = float(np.sqrt(var))
    logit_dw_se = pd.Series(se, name="state").reindex(states)

    counts = (
        phe_df.assign(deaths_chosen=phe_df["y"] == 1)
        .groupby("state")["deaths_chosen"]
        .agg(n_deaths_chosen="sum", n_cases_chosen=lambda g: int((~g).sum()))
    )
    return PHEFit(
        states=states,
        logit_dw=logit_dw,
        logit_dw_se=logit_dw_se,
        sigma=float(sigma),
        response_counts=counts,
        n_obs=len(phe_df),
        result=res,
    )


def nll(
    beta: np.ndarray,
    log_sigma: float,
    state_idx: np.ndarray,
    t: np.ndarray,
    y: np.ndarray,
) -> float:
    """Direct negative log-likelihood in the paper's (beta, sigma) parametrization.

    Used by the test suite to verify the probit reduction is the MLE; not the
    production estimator.
    """
    sigma = np.exp(log_sigma)
    z = (t - beta[state_idx]) / sigma
    return -float(np.sum(np.where(y == 1, norm.logcdf(z), norm.logcdf(-z))))
