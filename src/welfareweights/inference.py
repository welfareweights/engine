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
    # the state was estimable).
    weights: pd.DataFrame
    n_boot: int
    n_failed: int
    level: float
    diagnostics: dict  # full-sample estimate_dws diagnostics


def _take(df: pd.DataFrame, rows_by_id: dict, draw: np.ndarray) -> pd.DataFrame:
    chunks = [rows_by_id[i] for i in draw if i in rows_by_id]
    idx = np.concatenate(chunks) if chunks else np.array([], dtype=int)
    return df.loc[idx].reset_index(drop=True)


def bootstrap_dws(
    pc_df: pd.DataFrame,
    phe_df: pd.DataFrame,
    n_boot: int = 200,
    level: float = 0.95,
    rng: np.random.Generator | int | None = None,
    **estimate_kwargs,
) -> BootstrapDWs:
    """Percentile bootstrap over respondents for estimate_dws.

    estimate_kwargs are passed through to estimate_dws (deaths, ref_state,
    tau, min_anchor_r2, weight_by_precision).
    """
    rng = np.random.default_rng(rng)
    point, diag = estimate_dws(pc_df, phe_df, **estimate_kwargs)

    ids = np.union1d(pc_df["respondent_id"].unique(), phe_df["respondent_id"].unique())
    pc_rows = {i: g.index.to_numpy() for i, g in pc_df.groupby("respondent_id")}
    phe_rows = {i: g.index.to_numpy() for i, g in phe_df.groupby("respondent_id")}

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
    return BootstrapDWs(
        weights=weights, n_boot=n_boot, n_failed=n_failed, level=level, diagnostics=diag
    )
