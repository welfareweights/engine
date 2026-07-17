"""Input-validation front door: the checks every public entry point runs
before any stage code touches a frame (audit finding F4, composing F1's y
domain, F7's missing-data attribution, and F8's empty-input legs).

One module owns the LOGIC; the public entry points (pipeline.estimate_dws,
inference.bootstrap_dws, validate.holdout_fit, probit.fit_pc,
anchor.fit_phe) all CALL it, so no path reaches stage code with an
unvalidated frame and the diagnoses cannot drift apart. The checkers
validate and never transform: downstream code sees exactly the frames the
caller passed. Every failure raises ValueError naming the frame, the
column, and the cause — the exception class bootstrap_dws already counts as
a replicate failure, which gives a degenerate resample hitting this front
door inside the bootstrap loop correct semantics for free.

Check order is diagnostic sharpness: missing data fires before any domain
or unanimity logic, so NaN responses can never be misdiagnosed as one-sided
states (F7) or die in integer coercion naming neither state nor cause. The
y domain check is value-based (isin {0, 1}), never dtype-based: float
0.0/1.0, bool, and object-int columns are all previously-correct inputs and
stay accepted, while fractional values (silently truncated by .astype(int)
to a FLIPPED answer, F1), mis-codings like {1, 2}, and string-typed y
('0' != 0, which disabled the one-sided and separation guards, F4) all fail
loudly. n_cases/deaths are validated through pd.to_numeric, consistent with
downstream consumption (thresholds() reads them via np.asarray(..., float),
which tolerates numeric strings); the deaths-vs-n_cases RELATION is a model
precondition checked in anchor.thresholds, the only place that sees the
scalar deaths argument.

All checks are vectorized O(n) boolean scans — microseconds next to the two
Newton probit fits every estimation already pays for — so re-validation at
nested entry points (estimate_dws then its stage fits, every bootstrap
replicate) is the cheap price of single-sourced coverage, with no bypass
flag to go stale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_PC_COLUMNS = ("state_1", "state_2", "y")
_PHE_COLUMNS = ("state", "n_cases", "y")


def _check_columns(df: pd.DataFrame, name: str, required: list) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing required column(s) {missing}; expected columns {required}"
        )


def _check_empty(df: pd.DataFrame, name: str, what: str) -> None:
    if len(df) == 0:
        raise ValueError(f"{name} is empty: no {what} responses to fit")


def _check_state_labels(df: pd.DataFrame, name: str, cols: tuple) -> None:
    # A NaN state label would otherwise vanish silently from groupby counts
    # (pandas drops NaN groups) and crash sorted() in design.infer_states with
    # a cryptic mixed-type TypeError — the same silent class as F4/F7.
    for col in cols:
        n_nan = int(df[col].isna().sum())
        if n_nan:
            raise ValueError(
                f"{name}[{col!r}] contains {n_nan} missing state label(s) (NaN); "
                "rows with missing states must be resolved upstream"
            )


def _check_y(df: pd.DataFrame, name: str) -> None:
    # Missing-data check FIRST (F7): it must outrank the domain check so an
    # all-NaN state is reported as missing data, never as unanimity.
    n_nan = int(df["y"].isna().sum())
    if n_nan:
        raise ValueError(
            f"{name}['y'] contains {n_nan} missing value(s) (NaN); missing responses "
            "cannot be coerced or treated as one-sided — resolve them upstream"
        )
    y = df["y"]
    if y.dtype.kind in "iufb":
        # Numeric fast path (the common case, and the per-replicate hot path
        # inside bootstrap_dws): plain elementwise equality, identical
        # semantics to isin for numeric/bool values.
        arr = y.to_numpy()
        ok = (arr == 0) | (arr == 1)
    else:
        ok = y.isin([0, 1]).to_numpy()
    if not ok.all():
        bad = y[~ok].drop_duplicates().tolist()
        shown = "[" + ", ".join(repr(v) for v in bad[:5]) + "]"
        if any(isinstance(v, str) for v in bad):
            hint = "string-typed responses must be cast to integers"
        else:
            hint = (
                "fractional or mis-coded responses would otherwise be silently "
                "truncated by integer coercion"
            )
        raise ValueError(f"{name}['y'] contains values other than 0 and 1: {shown} — {hint}")


def _check_counts(df: pd.DataFrame, name: str, col: str) -> None:
    n_nan = int(df[col].isna().sum())
    if n_nan:
        raise ValueError(
            f"{name}[{col!r}] contains {n_nan} missing value(s) (NaN); every response "
            f"needs a finite, positive {col}"
        )
    if df[col].dtype.kind in "iuf":
        # Numeric fast path: already numeric, nothing to coerce.
        arr = df[col].to_numpy(dtype=float)
    else:
        coerced = pd.to_numeric(df[col], errors="coerce")
        noncoercible = coerced.isna()
        if noncoercible.any():
            bad = df.loc[noncoercible, col].drop_duplicates().tolist()
            shown = "[" + ", ".join(repr(v) for v in bad[:5]) + "]"
            raise ValueError(f"{name}[{col!r}] contains non-numeric value(s): {shown}")
        arr = coerced.to_numpy(dtype=float)
    good = np.isfinite(arr) & (arr > 0)
    if not good.all():
        shown = "[" + ", ".join(str(v) for v in sorted(set(arr[~good]))[:5]) + "]"
        raise ValueError(f"{name}[{col!r}] must be finite and positive; found {shown}")


def _check_respondent_id(df: pd.DataFrame, name: str) -> None:
    n_nan = int(df["respondent_id"].isna().sum())
    if n_nan:
        raise ValueError(
            f"{name}['respondent_id'] contains {n_nan} missing value(s) (NaN); every "
            "row must carry the respondent it came from"
        )


def check_pc_df(pc_df: pd.DataFrame, *, require_respondent_id: bool = False) -> None:
    """Validate a paired-comparison table; raise ValueError naming frame,
    column, and cause. Checks, in diagnostic-sharpness order (module
    docstring): required columns, non-emptiness, state labels, missing y
    BEFORE the y domain, y strictly in {0, 1}, and respondent_id
    completeness when required. Validation only — never transforms."""
    required = list(_PC_COLUMNS) + (["respondent_id"] if require_respondent_id else [])
    _check_columns(pc_df, "pc_df", required)
    _check_empty(pc_df, "pc_df", "paired-comparison")
    _check_state_labels(pc_df, "pc_df", ("state_1", "state_2"))
    _check_y(pc_df, "pc_df")
    if require_respondent_id:
        _check_respondent_id(pc_df, "pc_df")


def check_phe_df(phe_df: pd.DataFrame, *, require_respondent_id: bool = False) -> None:
    """Validate a population-health-equivalence table; raise ValueError
    naming frame, column, and cause. Same order as check_pc_df, plus
    n_cases and — when the column exists — deaths (no NaN, numeric, finite,
    positive). The deaths-vs-n_cases relation is checked in
    anchor.thresholds (module docstring). Validation only."""
    required = list(_PHE_COLUMNS) + (["respondent_id"] if require_respondent_id else [])
    _check_columns(phe_df, "phe_df", required)
    _check_empty(phe_df, "phe_df", "population-health-equivalence")
    _check_state_labels(phe_df, "phe_df", ("state",))
    _check_y(phe_df, "phe_df")
    _check_counts(phe_df, "phe_df", "n_cases")
    if "deaths" in phe_df.columns:
        _check_counts(phe_df, "phe_df", "deaths")
    if require_respondent_id:
        _check_respondent_id(phe_df, "phe_df")
