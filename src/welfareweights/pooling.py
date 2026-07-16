"""Pool several surveys: per-survey estimation, cross-survey tau in the back-transform.

Why pooling runs the full pipeline per survey first: each survey's stage-A
coefficients sit on their own arbitrary affine scale, and the anchor map
that resolves the indeterminacy is fitted per survey. The only scale the
surveys share is the death-anchored logit-weight scale, so that is where
estimates combine. estimate_dws_pooled calls the existing estimate_dws once
per survey and never reaches into stage internals.

This module implements the multi-survey tau that rescale.py's docstring
flags as replication decisions 4 and 6. The judgment calls, each with its
reason (described, not numbered — a global decision numbering already
exists and is not extended here):

Tau construction: per-state, matching Salomon's "variance of coefficients
across survey-specific estimates" (one variance per coefficient, not one
scalar shared across states with very different cross-survey agreement);
the sample SD (ddof=1) of the per-survey logit weights, because the
population formula would mechanically shrink tau exactly when few surveys
exist, the regime where honest disagreement matters most; and tau = 0 for
a state seen by fewer than two surveys, which is what makes a
single-survey list reduce exactly to estimate_dws.

Equal weighting across surveys, for both the mean and tau: estimate_dws
carries no per-state precision on the anchored scale (the stage
covariances are pre-anchor, and the anchor OLS does not propagate them),
so a weighting scheme here would be invented machinery; and precision
weighting would let a large survey dominate the mean while shrinking tau,
understating disagreement when surveys differ because they sample
different populations rather than because one is noisier.

Missing states: a state is pooled over the surveys that report it, with
the count in n_surveys; nothing is silently dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from welfareweights.pipeline import estimate_dws
from welfareweights.rescale import expected_expit


def estimate_dws_pooled(
    surveys: list[tuple[pd.DataFrame, pd.DataFrame]],
    **estimate_kwargs,
) -> tuple[pd.DataFrame, dict]:
    """Estimate pooled disability weights from several surveys.

    surveys: list of (pc_df, phe_df) pairs, one per survey, in the formats
    estimate_dws documents. estimate_kwargs pass through to each per-survey
    estimate_dws call (deaths, ref_state, min_anchor_r2, ...); tau is NOT
    accepted here because the cross-survey tau is what this function
    estimates.

    Returns (weights, diagnostics): weights indexed by state with columns
    logit_dw (cross-survey mean), tau (cross-survey SD; 0 when the state is
    seen by fewer than two surveys), n_surveys, and dw = E[expit(N(mu,
    tau^2))]; diagnostics carries the per-survey (weights, diagnostics)
    pairs under "per_survey".
    """
    if not surveys:
        raise ValueError("estimate_dws_pooled needs at least one survey")
    if "tau" in estimate_kwargs:
        raise ValueError("tau is estimated across surveys here; it cannot be passed through")
    per_survey = [estimate_dws(pc_df, phe_df, **estimate_kwargs) for pc_df, phe_df in surveys]

    logit_by_state: dict[str, list[float]] = {}
    for w, _ in per_survey:
        for state, val in w["logit_dw"].items():
            logit_by_state.setdefault(state, []).append(float(val))

    states = sorted(logit_by_state)
    n_surveys = np.array([len(logit_by_state[s]) for s in states])
    mu = np.array([np.mean(logit_by_state[s]) for s in states])
    tau = np.array(
        [np.std(logit_by_state[s], ddof=1) if n >= 2 else 0.0 for s, n in zip(states, n_surveys)]
    )
    weights = pd.DataFrame(
        {"logit_dw": mu, "tau": tau, "n_surveys": n_surveys, "dw": expected_expit(mu, tau)},
        index=pd.Index(states, name="state"),
    )
    return weights, {"per_survey": per_survey}
