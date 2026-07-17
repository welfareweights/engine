"""End-to-end estimation: PC + PHE long tables in, disability weights out."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from welfareweights.anchor import fit_phe
from welfareweights.checks import check_pc_df, check_phe_df
from welfareweights.probit import fit_pc
from welfareweights.rescale import expected_expit, fit_anchor_map


def estimate_dws(
    pc_df: pd.DataFrame,
    phe_df: pd.DataFrame,
    deaths: int | None = None,
    ref_state: str | None = None,
    tau: float = 0.0,
    min_anchor_r2: float = 0.9,
    max_curvature_impact: float = 0.025,
    max_extrapolation: float = 0.5,
    weight_by_precision: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Run the full stage A -> B -> C pipeline.

    pc_df:  [respondent_id, state_1, state_2, y], y = 1 iff state_1 judged healthier.
    phe_df: [respondent_id, state, n_cases, y] plus an optional per-row deaths
            column, y = 1 iff the deaths-averting program was chosen.
    Both frames are validated up front (welfareweights.checks): malformed
    input — missing columns, empty tables, NaN or out-of-domain y, NaN state
    labels, non-finite case counts — raises ValueError naming the frame,
    column, and cause before any stage code runs.
    deaths: passed to fit_phe; None uses phe_df's deaths column (or 1000).
    tau:    SD applied in the logit-space back-transform (stage C step 2);
            0 with a single survey — see rescale module docstring.
    min_anchor_r2: warn when the anchor line fits the shared states worse than
            this — weight LEVELS are suspect below it even where ranking holds.
    max_curvature_impact: warn when the anchor scatter bends by more than this
            (in weight units, see AnchorMap.curvature_impact) AND the bend is
            statistically distinguishable from noise (quadratic-term p < 0.05).
            Curvature is the one violation that damages levels while evading
            R^2; this gate detects severe bends, while mild ones sit below the
            noise floor at typical anchor counts (calibration in
            tests/test_curvature_diagnostic.py).
    max_extrapolation: warn when any state's mapped position sits further
            outside the PHE anchor range than this, measured in anchor-span
            units — the always-published `extrapolation` weights column. The
            curvature gate above measures misfit only INSIDE the anchor
            range, so a bend it reads as clean is amplified without bound
            where the line is extrapolated; this diagnostic covers exactly
            that blind spot.

            Why 0.5 spans is the default. The alternative model is the same
            quadratic the curvature gate tests (a different alternative
            would make the two diagnostics incoherent as a pair). For
            anchors roughly uniform on a span W and a bend c*x^2, the
            least-squares line's in-range residual is c(u^2 - W^2/12), so
            the largest misfit the gate can SEE is cW^2/6 at the range
            edges, while at delta spans past an edge the line-inversion
            error is c((1/2 + delta)^2 - 1/12)W^2 — an amplification over
            the gate's view of rho(delta) = 6(1/2 + delta)^2 - 1/2:
            rho(0) = 1, rho(0.25) ~ 2.9, rho(0.5) = 5.5, rho(1) = 13.
            Calibration: a bend at the gate's measured quiet level (impact
            ~0.0045 in the audit's reproduction of this configuration)
            reaches 5.5 x 0.0045 ~ 0.025 — the pipeline's OWN declared
            damage threshold, max_curvature_impact — at exactly delta = 0.5,
            and a bend sitting just under the gate's firing threshold
            reaches ~0.14 there, five times tolerance, invisible to every
            in-range gate. Deliberate non-goals: (a) extrapolation below
            half a span, where the same model bounds silently-amplified
            error under ~5.5x a quiet-gate bend — the designed configuration
            (PHE anchors fewer states than PC) extrapolates a little on
            every healthy run, and a threshold that always fires trains the
            reader to ignore the channel; (b) harm from links outside the
            quadratic family — the honest statement out there is "no anchor
            data exists", which is why the column publishes unconditionally
            rather than only when the warning fires; (c) a truly linear
            link, where extrapolating the correct line is harmless — hence
            warn-only, never raise, and dw is never blanked (blanking would
            silently drop extreme states from published output, the F6
            defect class). Caveats: the uniform-anchor residual is a design
            calculation, not an estimate — heavily clustered anchors weaken
            the rho(delta) mapping; and rho is an x-space ratio while expit
            compresses x-errors hardest at the extreme states being
            extrapolated, so weight-space harm is generally SMALLER than
            rho suggests — the threshold errs loud, the right direction for
            an unattended publisher. An analyst who intends deep
            extrapolation raises max_extrapolation explicitly in code,
            where the choice is visible and priced.
    weight_by_precision: precision-weight the anchor OLS (see fit_anchor_map).

    One-sided PHE states (every response favoring the same program) are dropped
    from the anchor stage with a warning: their logit weights are unidentified
    and would otherwise enter the anchor OLS as high-leverage garbage. They
    still receive final weights through stage A and the anchor map. When EVERY
    PHE state is one-sided — the empirically observed regime when respondents
    nearly always avert death — no identifiable anchor states remain and
    estimation raises a ValueError saying so instead of proceeding.

    Returns (weights, diagnostics): weights is a DataFrame indexed by state
    with columns beta (stage-A scale), logit_dw (anchored scale), dw (final,
    on [0, 1]), and extrapolation (distance of the state's mapped position
    outside the PHE anchor range, in anchor-span units; 0.0 inside the
    range), covering the stage-A states — states appearing only in phe_df
    are absent from weights (no stage-A coefficient exists to map), listed
    in diagnostics["phe_only_states"], and warned about. diagnostics also
    carries the fitted stage objects, the anchor map, and "one_sided_dropped".
    """
    check_pc_df(pc_df)
    check_phe_df(phe_df)

    y_by_state = phe_df.groupby("state")["y"]
    n1, ntot = y_by_state.sum(), y_by_state.size()
    one_sided = sorted(n1.index[(n1 == 0) | (n1 == ntot)])
    if one_sided:
        if len(one_sided) == len(ntot):
            raise ValueError(
                f"all {len(one_sided)} PHE state(s) are one-sided (every response favors "
                "the same program), so no identifiable PHE states remain for the anchor "
                "stage; the death anchor cannot be estimated — collect more anchor "
                "responses or widen the case-count range"
            )
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
    # NaN compares False on both conditions: no warning below 5 shared states.
    if amap.curvature_impact > max_curvature_impact and amap.curvature_pvalue < 0.05:
        warnings.warn(
            f"anchor scatter shows curvature (impact ~{amap.curvature_impact:.3f} in weight "
            f"units on the anchor range, p={amap.curvature_pvalue:.3f}): the straight-line "
            "map stage C assumes is bent, so weight LEVELS are suspect; the damage is worse "
            "outside the anchor range, where the bend is extrapolated. Rankings are "
            "unaffected. Widen the PHE anchor range and inspect the anchor scatter."
        )

    logit_dw = amap.to_logit_dw(pc_fit.beta)
    x = logit_dw.to_numpy()
    lo, hi = amap.anchor_range
    span = hi - lo
    outside = np.maximum(np.maximum(lo - x, x - hi), 0.0)
    if span > 0:
        extrapolation = outside / span
    else:
        # Degenerate span (exact ties across all shared anchors): a zero-span
        # anchor set supports no off-anchor state, so any distance is infinite
        # extrapolation and always warns.
        extrapolation = np.where(outside > 0, np.inf, 0.0)
    far = pd.Series(extrapolation, index=logit_dw.index)
    far = far[far > max_extrapolation]
    if len(far):
        listing = ", ".join(
            f"{s} ({d:.2f} spans)" for s, d in far.sort_values(ascending=False).items()
        )
        warnings.warn(
            f"{len(far)} state(s) sit more than {max_extrapolation} anchor-spans outside "
            f"the PHE anchor range [{lo:.3f}, {hi:.3f}] and their weights rely on the "
            "fitted line where no anchor data exists — the in-range bend gate above "
            "cannot see a bend amplified out here (see the extrapolation column): "
            f"{listing}; widen the PHE anchor range toward the extreme states or treat "
            "these levels as provisional. Rankings are unaffected."
        )

    dw = expected_expit(x, tau)
    weights = pd.DataFrame(
        {"beta": pc_fit.beta, "logit_dw": logit_dw, "dw": dw, "extrapolation": extrapolation},
        index=pc_fit.beta.index,
    )
    pc_states = set(pc_fit.states)
    phe_only = [s for s in phe_fit.states if s not in pc_states]
    if phe_only:
        warnings.warn(
            f"{len(phe_only)} PHE state(s) have no paired-comparison data and are absent "
            f"from the published weights (no stage-A coefficient exists to map): "
            f"{sorted(phe_only)}; their responses still inform the stage-B fit"
        )
    diagnostics = {
        "pc_fit": pc_fit,
        "phe_fit": phe_fit,
        "anchor_map": amap,
        "one_sided_dropped": one_sided,
        "phe_only_states": phe_only,
    }
    return weights, diagnostics
