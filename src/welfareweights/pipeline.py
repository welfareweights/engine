"""End-to-end estimation: PC + PHE long tables in, disability weights out."""

from __future__ import annotations

import warnings

import pandas as pd

from welfareweights.anchor import fit_phe
from welfareweights.probit import fit_pc
from welfareweights.rescale import expected_expit, fit_anchor_map


def estimate_dws(
    pc_df: pd.DataFrame,
    phe_df: pd.DataFrame,
    deaths: int | None = None,
    ref_state: str | None = None,
    tau: float = 0.0,
    min_anchor_r2: float = 0.9,
    weight_by_precision: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Run the full stage A -> B -> C pipeline.

    pc_df:  [respondent_id, state_1, state_2, y], y = 1 iff state_1 judged healthier.
    phe_df: [respondent_id, state, n_cases, y] plus an optional per-row deaths
            column, y = 1 iff the deaths-averting program was chosen.
    deaths: passed to fit_phe; None uses phe_df's deaths column (or 1000).
    tau:    SD applied in the logit-space back-transform (stage C step 2);
            0 with a single survey — see rescale module docstring.
    min_anchor_r2: warn when the anchor line fits the shared states worse than
            this — weight LEVELS are suspect below it even where ranking holds.
    weight_by_precision: precision-weight the anchor OLS (see fit_anchor_map).

    One-sided PHE states (every response favoring the same program) are dropped
    from the anchor stage with a warning: their logit weights are unidentified
    and would otherwise enter the anchor OLS as high-leverage garbage. They
    still receive final weights through stage A and the anchor map.

    Returns (weights, diagnostics): weights is a DataFrame indexed by state
    with columns beta (stage-A scale), logit_dw (anchored scale), dw (final,
    on [0, 1]), covering the stage-A states — states appearing only in phe_df
    are listed in diagnostics["phe_only_states"], not in weights. diagnostics
    also carries the fitted stage objects, the anchor map, and
    "one_sided_dropped".
    """
    y_by_state = phe_df.groupby("state")["y"]
    n1, ntot = y_by_state.sum(), y_by_state.size()
    one_sided = sorted(n1.index[(n1 == 0) | (n1 == ntot)])
    if one_sided:
        warnings.warn(
            f"dropping {len(one_sided)} one-sided PHE state(s) from the anchor stage "
            f"(logit weights unidentified): {one_sided}"
        )
        phe_df = phe_df[~phe_df["state"].isin(one_sided)]

    pc_fit = fit_pc(pc_df, ref_state=ref_state)
    phe_fit = fit_phe(phe_df, deaths=deaths)
    amap = fit_anchor_map(pc_fit, phe_fit, weight_by_precision=weight_by_precision)
    if amap.r_squared < min_anchor_r2:
        warnings.warn(
            f"anchor-map R^2 = {amap.r_squared:.3f} < {min_anchor_r2}: the affine map "
            "from the comparison scale to the death-anchored scale fits poorly, so "
            "weight levels are suspect"
        )

    logit_dw = amap.to_logit_dw(pc_fit.beta)
    dw = expected_expit(logit_dw.to_numpy(), tau)
    weights = pd.DataFrame(
        {"beta": pc_fit.beta, "logit_dw": logit_dw, "dw": dw}, index=pc_fit.beta.index
    )
    pc_states = set(pc_fit.states)
    diagnostics = {
        "pc_fit": pc_fit,
        "phe_fit": phe_fit,
        "anchor_map": amap,
        "one_sided_dropped": one_sided,
        "phe_only_states": [s for s in phe_fit.states if s not in pc_states],
    }
    return weights, diagnostics
