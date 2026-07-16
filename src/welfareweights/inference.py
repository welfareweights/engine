"""Respondent-cluster bootstrap: uncertainty intervals for the full pipeline.

Why this and not the stage covariance matrices: each respondent contributes
several PC rows and several PHE rows, so responses are dependent within
respondent and the model-based covariances (which assume independence)
understate uncertainty; and the anchor map plus the nonlinear expit
back-transform have no delta method worth trusting near the [0, 1] boundary.
Salomon 2012 likewise reports simulation-based uncertainty intervals.
Resampling whole respondents — jointly across the PC and PHE tables, since a
respondent answers both modules — and re-running all three stages propagates
every source at once: stage-A and stage-B sampling error, the anchor-map fit,
and the back-transform.

Replicates that fail estimation (a resample can lose a state, go one-sided,
or fail to converge) are dropped and counted in n_failed rather than
retried; a large n_failed is itself a diagnostic that the design is fragile
at this sample size.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from welfareweights.pipeline import estimate_dws


@dataclass
class BootstrapDWs:
    # weights per state: dw (full-sample point estimate), se (SD across
    # replicates), lo/hi (percentile interval), n_reps (replicates in which
    # the state was estimable), n_resp_pc / n_resp_phe (distinct respondents
    # informing the state in each module), supported (False where the
    # per-state support floor blanked se/lo/hi — see bootstrap_dws).
    weights: pd.DataFrame
    n_boot: int
    n_failed: int
    level: float
    diagnostics: dict  # full-sample estimate_dws diagnostics


def _take(df: pd.DataFrame, rows_by_id: dict, draw: np.ndarray) -> pd.DataFrame:
    # Resamples by row POSITION: rows_by_id maps respondent -> array of row
    # positions (GroupBy.indices) and iloc fetches them, so index labels are
    # ignored. Label lookup (df.loc) fans out under duplicate labels — e.g.
    # batches combined with pd.concat without ignore_index=True — silently
    # mixing rows across respondents in every replicate (audit finding B1).
    chunks = [rows_by_id[i] for i in draw if i in rows_by_id]
    idx = np.concatenate(chunks) if chunks else np.array([], dtype=int)
    return df.iloc[idx].reset_index(drop=True)


def bootstrap_dws(
    pc_df: pd.DataFrame,
    phe_df: pd.DataFrame,
    n_boot: int = 200,
    level: float = 0.95,
    rng: np.random.Generator | int | None = None,
    min_state_respondents: int = 10,
    min_state_reps: int | None = None,
    **estimate_kwargs,
) -> BootstrapDWs:
    """Percentile bootstrap over respondents for estimate_dws.

    min_state_respondents and min_state_reps set the per-state support floor
    below (they are NOT passed through); every other keyword argument goes to
    estimate_dws (deaths, ref_state, tau, min_anchor_r2, weight_by_precision).

    Per-state support floor (audit finding B2). A state's interval rests on
    the respondents who actually inform it, not on total N: with R distinct
    PC respondents on a state, each one's copy count per resample is
    ~Poisson(1), so the state's per-replicate support is ~Poisson(R). At
    R = 1 the state appears in ~63% of replicates carrying the same single
    respondent's answers every time — the spread across replicates is other
    states' resampling noise leaking through the shared probit and anchor
    map, not sampling uncertainty in the state, and the published interval
    is spuriously tight (measured: half-width ~5x smaller than the effect of
    flipping one of the respondent's answers) with no warning, which would
    let an automated CI-width promotion gate promote exactly the states it
    must hold back. States failing either floor therefore publish NaN
    se/lo/hi (fail-safe: a NaN width can never pass a `<` threshold),
    supported=False, and one loud warning naming each state and the floor(s)
    it failed. dw is kept: the full-sample point estimate's identification
    is enforced by fit_pc's separation and connectivity checks, so only the
    uncertainty statement is structurally meaningless — blanking dw would
    silently drop the state from published weights.

    The two floors, and why these values:
      min_state_respondents (default 10): distinct PC respondents whose
        pairs involve the state — the cluster count the respondent bootstrap
        actually resamples for it. Below ~10 clusters, percentile intervals
        sit in the few-clusters regime the cluster-inference literature
        treats as unreliable (Cameron, Gelbach & Miller 2008); at R = 10 the
        per-replicate support varies with SD ~ sqrt(10) (real respondent-
        sampling signal, not one respondent echoed) and P(state absent from
        a replicate) = e^-10, killing the estimability-selection effect the
        rep floor guards. The value sits deliberately at the BOTTOM of the
        defensible range: the floor's job is to make structurally
        meaningless intervals unpublishable, not to certify calibration
        (moderate-support coverage is studies/coverage.py's mandate); a
        higher floor would blank the thin-but-honest provisional intervals
        the promotion design wants visible.
      min_state_reps (default None = max(20, n_boot // 2), deliberately
        mirroring the whole-run fragility gate — same rationale per state):
        a state estimable in only a minority of replicates has percentile
        endpoints computed over a SELECTED subset — the resamples in which a
        fragile state survived are systematically the better-behaved ones,
        biasing the interval narrow — and quantiles of fewer than ~20 points
        are order statistics of a tiny sample. This leg alone would NOT
        catch the 1-respondent case (n_reps ~ 0.63 * n_boot sails past
        half); the respondent leg is primary.

    PHE support (n_resp_phe) is reported but never gated: a thin PHE anchor
    state distorts the anchor map globally through its OLS leverage, so
    blanking that state's own row would neither contain nor signal the
    damage — the global anchor gates (min_anchor_r2, curvature) own that
    risk. Setting both floors to 0 disables the gate entirely (research-use
    escape hatch, e.g. for simulation studies of the ungated estimator).
    """
    rng = np.random.default_rng(rng)
    point, diag = estimate_dws(pc_df, phe_df, **estimate_kwargs)

    ids = np.union1d(pc_df["respondent_id"].unique(), phe_df["respondent_id"].unique())
    # GroupBy.indices: respondent -> row POSITIONS, not index labels (see _take).
    pc_rows = pc_df.groupby("respondent_id").indices
    phe_rows = phe_df.groupby("respondent_id").indices

    reps: list[pd.Series] = []
    n_failed = 0
    for _ in range(n_boot):
        draw = rng.choice(ids, size=len(ids), replace=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                w, _ = estimate_dws(
                    _take(pc_df, pc_rows, draw), _take(phe_df, phe_rows, draw), **estimate_kwargs
                )
            reps.append(w["dw"])
        except (ValueError, np.linalg.LinAlgError):
            n_failed += 1
    # F9 cross-reference: the per-state rep floor below reuses this same
    # max(20, n_boot // 2) expression on purpose (one policy, applied whole-run
    # and per-state); the F9 fix (failure-fraction gate) must change both.
    if len(reps) < max(20, n_boot // 2):
        raise ValueError(
            f"only {len(reps)}/{n_boot} bootstrap replicates were estimable; the design "
            "is too fragile at this sample size for interval estimates to mean anything"
        )

    mat = pd.concat(reps, axis=1)
    alpha = (1.0 - level) / 2.0
    weights = pd.DataFrame(
        {
            "dw": point["dw"],
            "se": mat.std(axis=1, ddof=1),
            "lo": mat.quantile(alpha, axis=1),
            "hi": mat.quantile(1.0 - alpha, axis=1),
            "n_reps": mat.notna().sum(axis=1),
        }
    ).reindex(point.index)
    # A state absent from every replicate must read n_reps 0, not NaN: NaN
    # would silently pass the `<` floor below (NaN < x is False).
    weights["n_reps"] = weights["n_reps"].fillna(0).astype(int)

    # Per-state support (rationale in the docstring): n_resp_pc counts the
    # distinct respondents whose pairs involve the state on either side — the
    # cluster count the respondent bootstrap actually resamples for it.
    pc_long = pd.concat(
        [
            pc_df[["respondent_id", "state_1"]].rename(columns={"state_1": "state"}),
            pc_df[["respondent_id", "state_2"]].rename(columns={"state_2": "state"}),
        ]
    )
    n_resp_pc = pc_long.groupby("state")["respondent_id"].nunique()
    n_resp_phe = phe_df.groupby("state")["respondent_id"].nunique()
    weights["n_resp_pc"] = n_resp_pc.reindex(point.index).fillna(0).astype(int)
    weights["n_resp_phe"] = n_resp_phe.reindex(point.index).fillna(0).astype(int)

    # F9 cross-reference: same expression as the whole-run gate above — the
    # F9 failure-fraction fix must change both together.
    rep_floor = max(20, n_boot // 2) if min_state_reps is None else min_state_reps
    gated = (weights["n_resp_pc"] < min_state_respondents) | (weights["n_reps"] < rep_floor)
    weights["supported"] = ~gated
    if gated.any():
        details = []
        for s in weights.index[gated]:
            fails = []
            if weights.at[s, "n_resp_pc"] < min_state_respondents:
                fails.append(
                    f"pc respondents {weights.at[s, 'n_resp_pc']} < {min_state_respondents}"
                )
            if weights.at[s, "n_reps"] < rep_floor:
                fails.append(f"n_reps {weights.at[s, 'n_reps']} < {rep_floor}")
            details.append(f"{s} ({', '.join(fails)})")
        warnings.warn(
            f"{int(gated.sum())} state(s) below the per-state support floor — se/lo/hi "
            f"blanked (NaN): {'; '.join(details)}; a respondent-cluster bootstrap cannot "
            "express uncertainty for a state this thin, and the interval it would publish "
            "is spuriously tight; point estimates (dw) are retained"
        )
        weights.loc[gated, ["se", "lo", "hi"]] = np.nan
    return BootstrapDWs(
        weights=weights, n_boot=n_boot, n_failed=n_failed, level=level, diagnostics=diag
    )
