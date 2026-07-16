"""Stage C: map stage-A coefficients onto the death-anchored scale, back to [0, 1].

Two steps (Salomon 2012 supplement):

1. Across the states appearing in both stages, OLS of the stage-A probit
   coefficients on the stage-B logit-weights: beta = slope * logit_dw +
   intercept. Inverting the fitted line maps ANY probit coefficient onto the
   logit-weight scale — this resolves stage A's affine indeterminacy using
   the only death-anchored information available.

   Regression-direction decision: Salomon regresses the stage-A coefficients
   on the stage-B logit weights and inverts the fitted line (inverse
   prediction); the replication keeps that. Regressing the other way, or an
   errors-in-variables fit (Deming), differs by roughly a factor R^2 in the
   slope — second-order when the anchor fit is tight, which the pipeline's
   R^2 warning enforces. Measured empirically: classical attenuation from
   stage-B sampling noise moved the slope < 1% even at an 800-respondent PHE
   sample; the mechanism that actually bites is a quasi-separated state's
   divergent logit weight acting as a high-leverage point, which fit_phe and
   estimate_dws now refuse/drop upstream.

2. Back-transform by integration, not by plugging in: the reported weight is
   E[expit(N(mu, tau^2))] with mu the rescaled coefficient, because the mean
   of inverse-logit draws differs from the inverse-logit of the mean. Salomon
   computes the expectation by Monte Carlo simulation with tau^2 the variance
   of coefficients across survey-specific estimates; `expected_expit` computes
   the same integral by Gauss-Hermite quadrature (deterministic, no simulation
   noise). With a single survey there is no cross-survey variance and tau
   defaults to 0, reducing to a plain inverse logit (replication decisions
   4 and 6 track the multi-survey tau).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit

from welfareweights.anchor import PHEFit
from welfareweights.probit import PCFit


@dataclass
class AnchorMap:
    """Fitted line beta = slope * logit_dw + intercept over shared states."""

    slope: float
    intercept: float
    n_shared: int
    r_squared: float

    def to_logit_dw(self, beta: pd.Series) -> pd.Series:
        return (beta - self.intercept) / self.slope


def fit_anchor_map(
    pc_fit: PCFit, phe_fit: PHEFit, weight_by_precision: bool = False
) -> AnchorMap:
    """Fit the anchor line over the states present in both stages.

    weight_by_precision=False reproduces Salomon's unweighted OLS; True
    weights each shared state by the inverse of its stage-B delta-method SE,
    downweighting weakly identified anchors.
    """
    shared = [s for s in pc_fit.states if s in set(phe_fit.states)]
    if len(shared) < 3:
        raise ValueError(
            "need at least three states in both stages: two determine the anchor line "
            "exactly, leaving no way to assess its fit"
        )
    if len(shared) < 5:
        warnings.warn(
            f"anchor map fitted on only {len(shared)} shared states; "
            "R^2 is a weak diagnostic at this size"
        )
    x = phe_fit.logit_dw[shared].to_numpy()
    y = pc_fit.beta[shared].to_numpy()
    w = 1.0 / phe_fit.logit_dw_se[shared].to_numpy() if weight_by_precision else None
    slope, intercept = np.polyfit(x, y, 1, w=w)
    if slope >= 0:
        raise ValueError(
            "anchor-map slope is non-negative: stage-A healthiness must decrease in the "
            "stage-B logit weight, so the two stages disagree on orientation"
        )
    resid = y - (slope * x + intercept)
    r2 = 1.0 - resid.var() / y.var()
    return AnchorMap(
        slope=float(slope), intercept=float(intercept), n_shared=len(shared), r_squared=float(r2)
    )


def expected_expit(mu: np.ndarray, tau: float | np.ndarray, n_nodes: int = 64) -> np.ndarray:
    """E[expit(X)], X ~ N(mu, tau^2), by Gauss-Hermite quadrature.

    tau may be scalar or per-element. tau = 0 returns expit(mu) exactly.
    """
    mu = np.asarray(mu, dtype=float)
    tau_arr = np.broadcast_to(np.asarray(tau, dtype=float), mu.shape)
    nodes, weights = np.polynomial.hermite.hermgauss(n_nodes)
    vals = expit(mu[..., None] + tau_arr[..., None] * np.sqrt(2.0) * nodes)
    return vals @ weights / np.sqrt(np.pi)
