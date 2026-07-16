"""Held-out model fit: does the fitted response model predict choices from
respondents it never saw?

test_recovery.py is the engine's correctness evidence: it simulates from a
known DGP with known true disability weights and checks the estimator
recovers them. That evidence only exists because the true weights are known
by construction — it says nothing about how well the response model fits
respondents whose "true" weights nobody knows, which is exactly the
situation with real LLM or human survey data. holdout_fit needs no ground
truth: it splits respondents into train/test, fits stage A (paired
comparisons) and stage B (population-health-equivalence responses) on train
only, and scores each stage's own generative model against the actual
choices made by the held-out respondents —

    PC:  P(state_1 judged healthier) = Phi(beta_1 - beta_2)
    PHE: P(deaths program chosen)    = Phi((t - beta_s) / sigma)

with beta, sigma from the train fit and t computed from each held-out row's
own deaths/n_cases. That is out-of-sample fit of the response model itself
— discrimination (log-likelihood over a null base-rate model) and
calibration (do predicted probabilities match empirical frequencies) — and
because it only ever compares a predicted probability to an observed binary
choice, it is the one validity check here that transfers unchanged from
this file's synthetic tests to real survey data later.

Stage C (the anchor map and expit back-transform) is not evaluated here: it
maps between the two stages' arbitrary scales rather than predicting a
response, so it has no choice probability to score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from welfareweights.anchor import PHEFit, thresholds
from welfareweights.pipeline import estimate_dws
from welfareweights.probit import PCFit

_N_CALIB_BINS = 10
_EPS = 1e-12


@dataclass
class ModuleFit:
    """Held-out fit for one module (PC or PHE); see module docstring."""

    mean_loglik: float  # mean held-out log-likelihood per response, fitted model
    mean_loglik_null: float  # same, null model (constant train base rate)
    n_eval: int  # held-out rows scored
    n_skipped: int  # held-out rows naming a state absent from the train fit
    calibration: pd.DataFrame  # columns: bin_lo, bin_hi, n, mean_predicted, empirical_freq


@dataclass
class HoldoutFit:
    """Held-out evaluation of the pipeline's response models; see module docstring."""

    pc: ModuleFit
    phe: ModuleFit
    n_train_respondents: int
    n_test_respondents: int


def _loglik(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return float(np.mean(np.where(y == 1, np.log(p), np.log(1.0 - p))))


def _calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = _N_CALIB_BINS) -> pd.DataFrame:
    """Fixed-width bins of predicted probability vs. empirical frequency, with counts."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        rows.append(
            {
                "bin_lo": edges[b],
                "bin_hi": edges[b + 1],
                "n": n,
                "mean_predicted": float(p[mask].mean()) if n else float("nan"),
                "empirical_freq": float(y[mask].mean()) if n else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _module_result(y: np.ndarray, p: np.ndarray, null_p: float, n_skipped: int) -> ModuleFit:
    n_eval = int(len(y))
    if n_eval == 0:
        empty = pd.DataFrame(columns=["bin_lo", "bin_hi", "n", "mean_predicted", "empirical_freq"])
        return ModuleFit(float("nan"), float("nan"), 0, n_skipped, empty)
    return ModuleFit(
        mean_loglik=_loglik(y, p),
        mean_loglik_null=_loglik(y, np.full(n_eval, null_p)),
        n_eval=n_eval,
        n_skipped=n_skipped,
        calibration=_calibration_table(y, p),
    )


def eval_pc(pc_fit: PCFit, pc_test: pd.DataFrame, null_p: float) -> ModuleFit:
    """Score held-out PC rows against the train stage-A fit (module docstring).

    Rows naming a state absent from pc_fit.beta (i.e. absent from the train
    comparison data) are skipped and counted in the result's n_skipped.
    """
    known = set(pc_fit.beta.index)
    keep = pc_test["state_1"].isin(known) & pc_test["state_2"].isin(known)
    n_skipped = int((~keep).sum())
    df = pc_test.loc[keep]
    p = norm.cdf(
        pc_fit.beta.loc[df["state_1"]].to_numpy() - pc_fit.beta.loc[df["state_2"]].to_numpy()
    )
    return _module_result(df["y"].to_numpy(), p, null_p, n_skipped)


def eval_phe(
    phe_fit: PHEFit, phe_test: pd.DataFrame, null_p: float, deaths: int | None = None
) -> ModuleFit:
    """Score held-out PHE rows against the train stage-B fit (module docstring).

    Rows naming a state absent from phe_fit.logit_dw are skipped and counted
    in the result's n_skipped. deaths mirrors anchor.fit_phe: a per-row
    `deaths` column on phe_test is used when present (thresholds must use
    the deaths figure the respondent actually saw); otherwise the deaths
    argument is used, defaulting to 1000.
    """
    known = set(phe_fit.logit_dw.index)
    keep = phe_test["state"].isin(known)
    n_skipped = int((~keep).sum())
    df = phe_test.loc[keep]
    d = df["deaths"].to_numpy() if "deaths" in df.columns else (1000 if deaths is None else deaths)
    t = thresholds(df["n_cases"].to_numpy(), d)
    beta_s = phe_fit.logit_dw.loc[df["state"]].to_numpy()
    p = norm.cdf((t - beta_s) / phe_fit.sigma)
    return _module_result(df["y"].to_numpy(), p, null_p, n_skipped)


def calibration_slope(calibration: pd.DataFrame) -> float:
    """Weighted least squares slope of empirical frequency on predicted probability.

    Bins are weighted by their count; empty bins (n=0) are dropped. A
    well-calibrated model has slope ~= 1: predicted probabilities track
    observed frequencies bin-for-bin. This is the calibration half of model
    fit, complementing the discrimination mean_loglik already scores.
    """
    used = calibration[calibration["n"] > 0]
    if len(used) < 2:
        raise ValueError(
            f"only {len(used)} populated calibration bin(s); need at least two to fit a slope"
        )
    slope, _intercept = np.polyfit(
        used["mean_predicted"].to_numpy(),
        used["empirical_freq"].to_numpy(),
        1,
        w=used["n"].to_numpy(),
    )
    return float(slope)


def holdout_fit(
    pc_df: pd.DataFrame,
    phe_df: pd.DataFrame,
    test_frac: float = 0.3,
    rng: np.random.Generator | int | None = None,
    **estimate_kwargs,
) -> HoldoutFit:
    """Split respondents into train/test, fit the pipeline on train, score test.

    The split is over the union of respondent ids in pc_df and phe_df, not
    over rows: a respondent's PC and PHE rows move together, mirroring
    inference.bootstrap_dws's respondent-level resampling (a real respondent
    answers both modules, so held-out evaluation must hold out the whole
    respondent, not just some of their rows). test_frac is the fraction of
    that respondent union assigned to test.

    estimate_kwargs pass through to pipeline.estimate_dws, fit on the train
    partition only (deaths, ref_state, tau, min_anchor_r2,
    weight_by_precision); `deaths` also supplies the PHE evaluation fallback
    when a held-out row lacks its own deaths column (see eval_phe).

    Returns a HoldoutFit: per-module (pc, phe) held-out log-likelihood for
    the fitted model and for a null model (the train base rate), a
    calibration table, and counts of skipped rows (module docstring) plus
    train/test respondent counts.
    """
    rng = np.random.default_rng(rng)
    ids = np.union1d(pc_df["respondent_id"].unique(), phe_df["respondent_id"].unique())
    n_test = max(1, round(len(ids) * test_frac)) if len(ids) else 0
    test_ids = set(rng.choice(ids, size=n_test, replace=False)) if n_test else set()
    train_ids = set(ids) - test_ids

    pc_train = pc_df[pc_df["respondent_id"].isin(train_ids)]
    pc_test = pc_df[pc_df["respondent_id"].isin(test_ids)]
    phe_train = phe_df[phe_df["respondent_id"].isin(train_ids)]
    phe_test = phe_df[phe_df["respondent_id"].isin(test_ids)]

    _, diag = estimate_dws(pc_train, phe_train, **estimate_kwargs)

    pc_module = eval_pc(diag["pc_fit"], pc_test, float(pc_train["y"].mean()))
    phe_module = eval_phe(
        diag["phe_fit"],
        phe_test,
        float(phe_train["y"].mean()),
        deaths=estimate_kwargs.get("deaths"),
    )
    return HoldoutFit(
        pc=pc_module,
        phe=phe_module,
        n_train_respondents=len(train_ids),
        n_test_respondents=len(test_ids),
    )
