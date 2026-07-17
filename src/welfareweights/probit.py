"""Stage A: probit on paired comparisons -> health values on an arbitrary linear scale.

Model (Salomon 2012 supplement): latent health of state i is
H_i ~ N(theta_i, sigma^2); state 1 is judged healthier iff H_1 > H_2, so
P(y=1) = Phi(theta_1 - theta_2) under the 2*sigma^2 = 1 normalization.
With the +1/-1/0 design coding this is a plain binary probit; the fitted
beta_i recover theta_i up to an affine transform (beta_ref = 0 and the
variance normalization fix location and scale arbitrarily). Stage C
resolves the affine indeterminacy.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm

from welfareweights.checks import check_pc_df
from welfareweights.design import build_pc_design, connected_components, infer_states


@dataclass
class PCFit:
    """Stage-A output: probit coefficients on the arbitrary linear scale."""

    states: list[str]
    ref_state: str
    beta: pd.Series  # indexed by ALL states; beta[ref_state] = 0.0
    # Model-based covariance of the k-1 estimated coefficients. It treats
    # responses as independent; respondents answer many pairs each, so use
    # inference.bootstrap_dws for uncertainty, not this.
    cov: pd.DataFrame
    n_obs: int
    result: object  # statsmodels result, for diagnostics


def fit_pc(
    pc_df: pd.DataFrame,
    states: list[str] | None = None,
    ref_state: str | None = None,
) -> PCFit:
    """Fit the stage-A probit.

    ref_state defaults to the first state in sorted order. The choice is an
    arbitrary normalization: final disability weights must be invariant to it
    (verified numerically in the test suite — replication decision 2).

    pc_df is validated up front (welfareweights.checks): malformed input
    raises ValueError naming the frame, column, and cause.
    """
    check_pc_df(pc_df)
    if states is None:
        states = infer_states(pc_df)
    comps = connected_components(pc_df)
    if len(comps) > 1:
        sizes = sorted((len(c) for c in comps), reverse=True)
        raise ValueError(
            f"comparison graph is disconnected ({len(comps)} components, sizes {sizes}); "
            "relative values across components are unidentified"
        )
    if ref_state is None:
        ref_state = states[0]

    # Separation pre-check: a state judged healthier in all (or none) of its
    # comparisons has a divergent MLE, like an unbeaten team in Bradley-Terry.
    wins = pd.concat(
        [pc_df.loc[pc_df["y"] == 1, "state_1"], pc_df.loc[pc_df["y"] == 0, "state_2"]]
    ).value_counts()
    appearances = pd.concat([pc_df["state_1"], pc_df["state_2"]]).value_counts()
    one_sided = [s for s in states if wins.get(s, 0) in (0, appearances.get(s, 0))]
    if one_sided:
        raise ValueError(
            f"state(s) {one_sided} judged healthier in all or none of their comparisons; "
            "their coefficients diverge (separation) — drop them or collect more data"
        )

    X, y = build_pc_design(pc_df, states, ref_state)
    res = sm.Probit(y, X).fit(disp=0, method="newton")
    if not res.mle_retvals.get("converged", False):
        raise ValueError(
            "stage-A probit did not converge; likely near-separation in the comparison data"
        )
    beta = pd.Series(0.0, index=pd.Index(states, name="state"))
    beta[res.params.index] = res.params.to_numpy()
    return PCFit(
        states=list(states),
        ref_state=ref_state,
        beta=beta,
        cov=res.cov_params(),
        n_obs=len(pc_df),
        result=res,
    )
