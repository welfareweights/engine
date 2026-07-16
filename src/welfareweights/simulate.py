"""Synthetic survey data from the exact DGP the estimator assumes.

The estimator (probit on paired comparisons, interval-censored normal
regression on PHE responses) is derived from a specific data-generating
process. Simulating from that process with known true disability weights
and checking recovery is the package's core correctness test: it validates
the code against the model, separately from the (harder) question of
whether the model fits real respondents.

Both simulators also expose misspecification knobs (all defaulting to the
assumed DGP, with the default draw sequence unchanged) so the robustness
studies can measure how estimation degrades when respondents violate the
model. The estimator never sees these knobs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm

# PHE design constants from the GBD 2010 instrument (Salomon 2012 supplement):
# Program 1 prevents DEATHS fatal cases; Program 2 prevents c cases of the
# nonfatal state, c drawn uniformly from C_CHOICES.
DEATHS = 1000
C_CHOICES = (1500, 2000, 3000, 5000, 10000)


def make_states(k: int) -> list[str]:
    """Synthetic state labels s000, s001, ..."""
    return [f"s{i:03d}" for i in range(k)]


def simulate_true_dws(k: int, low: float = 0.005, high: float = 0.95) -> np.ndarray:
    """True disability weights, evenly spaced in logit space over [low, high]."""
    return expit(np.linspace(logit(low), logit(high), k))


def simulate_pc(
    true_dws: np.ndarray,
    states: list[str],
    n_respondents: int,
    pairs_per_respondent: int = 15,
    slope: float = -1.0,
    intercept: float = 0.0,
    curvature: float = 0.0,
    scale_sd: float = 0.0,
    position_bias: float = 0.0,
    lapse_rate: float = 0.0,
    error_dist: str = "normal",
    rng: np.random.Generator | int | None = None,
) -> pd.DataFrame:
    """Paired-comparison long table from the Thurstone probit DGP.

    Latent health value of state i is theta_i = slope * logit(dw_i) + intercept,
    with slope < 0 (heavier weight = less healthy). The affine link is exactly
    the relationship stage C's rescaling assumes, so end-to-end recovery tests
    the whole pipeline. A respondent judges state_1 healthier with probability
    Phi(theta_1 - theta_2) — stage A's model under its 2*sigma^2 = 1
    normalization. Pairs are assigned uniformly at random, mirroring the
    survey's random pair assignment.

    Misspecification knobs (defaults = the assumed DGP):
      curvature: adds curvature * logit(dw)^2 to theta, bending the affine
        A<->B link stage C assumes.
      scale_sd: respondent-specific comparison scale exp(N(0, scale_sd^2))
        multiplying the index — scale heterogeneity across respondents.
      position_bias: added to the index in favor of the first-listed state.
      lapse_rate: fraction of responses replaced by a fair coin (inattention).
      error_dist: "normal" (assumed) or "logistic" (slope-matched via
        expit(1.702 z)) — link misspecification.

    Returns a DataFrame [respondent_id, state_1, state_2, y],
    y = 1 iff state_1 was judged healthier.
    """
    rng = np.random.default_rng(rng)
    k = len(true_dws)
    if len(states) != k:
        raise ValueError("states and true_dws must align")
    lg = logit(np.asarray(true_dws, dtype=float))
    theta = slope * lg + intercept + curvature * lg**2
    n = n_respondents * pairs_per_respondent
    rid = np.repeat(np.arange(n_respondents), pairs_per_respondent)
    s1 = rng.integers(0, k, size=n)
    s2 = (s1 + rng.integers(1, k, size=n)) % k  # uniform over distinct ordered pairs
    z = theta[s1] - theta[s2] + position_bias
    if scale_sd > 0:
        z = z * np.exp(rng.normal(0.0, scale_sd, n_respondents))[rid]
    if error_dist == "normal":
        p = norm.cdf(z)
    elif error_dist == "logistic":
        p = expit(1.702 * z)
    else:
        raise ValueError(f"unknown error_dist {error_dist!r}")
    y = (rng.random(n) < p).astype(int)
    if lapse_rate > 0:
        lapse = rng.random(n) < lapse_rate
        y[lapse] = rng.integers(0, 2, size=int(lapse.sum()))
    st = np.asarray(states, dtype=object)
    return pd.DataFrame(
        {
            "respondent_id": rid,
            "state_1": st[s1],
            "state_2": st[s2],
            "y": y,
        }
    )


def simulate_phe(
    true_dws: np.ndarray,
    states: list[str],
    n_respondents: int,
    questions_per_respondent: int = 3,
    sigma: float = 0.8,
    re_sd: float = 0.0,
    lapse_rate: float = 0.0,
    deaths: int = DEATHS,
    c_choices: tuple[int, ...] = C_CHOICES,
    anchor_states: list[str] | None = None,
    rng: np.random.Generator | int | None = None,
) -> pd.DataFrame:
    """PHE long table from the interval-censored logit-normal DGP.

    The respondent's perceived logit-weight of state s is
    L = logit(dw_s) + eps, eps ~ N(0, sigma^2). Program 1 (prevent `deaths`
    fatal cases) is chosen iff deaths > c * expit(L), i.e. iff
    L < logit(deaths / c) — exactly stage B's censoring model.

    anchor_states restricts questions to a subset of states (in the real
    design PHE questions cover fewer states than the PC module, and states
    with extreme weights are weakly identified by binary censoring alone).

    Misspecification knobs (defaults = the assumed DGP):
      re_sd: respondent random intercept N(0, re_sd^2) added to L — within-
        respondent correlation the estimator ignores.
      lapse_rate: fraction of responses replaced by a fair coin.

    Returns a DataFrame [respondent_id, state, n_cases, deaths, y],
    y = 1 iff Program 1 (averting deaths) was chosen.
    """
    rng = np.random.default_rng(rng)
    true_dws = np.asarray(true_dws, dtype=float)
    if anchor_states is None:
        anchor_states = list(states)
    dw_by_state = dict(zip(states, true_dws))
    anchor_dws = np.array([dw_by_state[s] for s in anchor_states])

    n = n_respondents * questions_per_respondent
    rid = np.repeat(np.arange(n_respondents), questions_per_respondent)
    idx = rng.integers(0, len(anchor_states), size=n)
    c = rng.choice(np.asarray(c_choices), size=n)
    latent = logit(anchor_dws[idx]) + rng.normal(0.0, sigma, size=n)
    if re_sd > 0:
        latent = latent + rng.normal(0.0, re_sd, n_respondents)[rid]
    y = (latent < logit(deaths / c)).astype(int)
    if lapse_rate > 0:
        lapse = rng.random(n) < lapse_rate
        y[lapse] = rng.integers(0, 2, size=int(lapse.sum()))
    st = np.asarray(anchor_states, dtype=object)
    return pd.DataFrame(
        {
            "respondent_id": rid,
            "state": st[idx],
            "n_cases": c,
            "deaths": deaths,
            "y": y,
        }
    )
