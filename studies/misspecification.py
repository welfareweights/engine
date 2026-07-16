"""Misspecification battery: how estimation degrades when respondents violate the model.

One knob at a time, simulate from a DGP that violates an estimator assumption
(simulate.py's misspecification knobs), run the full estimate_dws pipeline, and
measure degradation against the baseline exact-DGP cell. Every cell shares one
base configuration (slope=-1.0, intercept=0.3, sigma=0.8, anchor_states =
mid-range subset — identical to tests/test_recovery.py) built from single
shared kwargs dicts, with exactly the cell's target knob overridden on top, so
"only one thing changes per cell" holds structurally, not by transcription.

Failures (ValueError from any pipeline stage) are counted per cell and never
retried with a new seed; each rep's seeds are recorded in the CSV so any row is
reproducible. Output: studies/results/misspecification.csv (one row per
cell x rep); the per-cell summary is printed to the console and transcribed
into studies/RESULTS-misspecification.md, not persisted as a separate file.
"""

import sys

sys.path.insert(0, "/home/richard/projects/WelfareWeights/engine/src")

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from welfareweights.pipeline import estimate_dws
from welfareweights.simulate import make_states, simulate_pc, simulate_phe, simulate_true_dws

RESULTS_DIR = Path("/home/richard/projects/WelfareWeights/engine/studies/results")

# Design constants — identical to tests/test_recovery.py's fixture setup.
K = 20
STATES = make_states(K)
TRUE_DWS = simulate_true_dws(K)
# Mid-range anchor subset: extreme states give one-sided PHE responses
# (quasi-separation), a weakness of the real design, not a misspecification
# effect — excluding them everywhere (baseline included) keeps that constant
# out of the knob comparisons.
ANCHOR_STATES = [s for s, dw in zip(STATES, TRUE_DWS) if 0.05 <= dw <= 0.70]

N_REPS = 15

# One shared base config for ALL cells (baseline and misspecified alike).
# simulate_pc's own default intercept is 0.0, not 0.3 — passing these
# explicitly everywhere is what keeps non-baseline cells from differing from
# baseline in two ways at once.
BASE_PC_KWARGS = dict(n_respondents=1500, slope=-1.0, intercept=0.3)
BASE_PHE_KWARGS = dict(n_respondents=2500, sigma=0.8, anchor_states=ANCHOR_STATES)

# Grid: (cell_label, knob, severity, module, kwarg overrides for that module).
# One knob per cell; severity is the printed value ("-" for baseline).
CELLS = [
    ("baseline", "baseline", 0.0, None, {}),
    ("pc_scale_sd=0.3", "pc_scale_sd", 0.3, "pc", {"scale_sd": 0.3}),
    ("pc_scale_sd=0.6", "pc_scale_sd", 0.6, "pc", {"scale_sd": 0.6}),
    ("phe_re_sd=0.5", "phe_re_sd", 0.5, "phe", {"re_sd": 0.5}),
    ("phe_re_sd=1.0", "phe_re_sd", 1.0, "phe", {"re_sd": 1.0}),
    ("pc_lapse=0.05", "pc_lapse_rate", 0.05, "pc", {"lapse_rate": 0.05}),
    ("pc_lapse=0.15", "pc_lapse_rate", 0.15, "pc", {"lapse_rate": 0.15}),
    ("phe_lapse=0.05", "phe_lapse_rate", 0.05, "phe", {"lapse_rate": 0.05}),
    ("phe_lapse=0.15", "phe_lapse_rate", 0.15, "phe", {"lapse_rate": 0.15}),
    ("pc_position_bias=0.1", "pc_position_bias", 0.1, "pc", {"position_bias": 0.1}),
    ("pc_position_bias=0.3", "pc_position_bias", 0.3, "pc", {"position_bias": 0.3}),
    ("pc_curvature=0.05", "pc_curvature", 0.05, "pc", {"curvature": 0.05}),
    ("pc_curvature=0.15", "pc_curvature", 0.15, "pc", {"curvature": 0.15}),
    ("pc_error_dist=logistic", "pc_error_dist", np.nan, "pc", {"error_dist": "logistic"}),
]


def run_rep(cell_idx: int, rep: int, module: str | None, overrides: dict) -> dict:
    """Simulate one rep of one cell, fit, and return the metrics row."""
    pc_kwargs = dict(BASE_PC_KWARGS)
    phe_kwargs = dict(BASE_PHE_KWARGS)
    if module == "pc":
        pc_kwargs.update(overrides)
    elif module == "phe":
        phe_kwargs.update(overrides)
    # Deterministic, distinct seeds recorded in the CSV: no rep is ever
    # silently re-seeded, and any row is exactly reproducible.
    pc_seed = cell_idx * 10000 + 2 * rep
    phe_seed = pc_seed + 1
    pc_df = simulate_pc(TRUE_DWS, STATES, rng=pc_seed, **pc_kwargs)
    phe_df = simulate_phe(TRUE_DWS, STATES, rng=phe_seed, **phe_kwargs)
    row = dict(rep=rep, pc_seed=pc_seed, phe_seed=phe_seed, failed=False)
    metrics = ["spearman", "max_abs_err", "mean_abs_err", "anchor_slope", "anchor_r2"]
    try:
        with warnings.catch_warnings():
            # estimate_dws's low-R^2 / one-sided-state warnings are expected
            # under misspecification and show up in the metrics themselves;
            # the failure signal here is ValueError, not a warning.
            warnings.simplefilter("ignore")
            weights, diag = estimate_dws(pc_df, phe_df)
    except ValueError:
        row["failed"] = True
        row.update({m: np.nan for m in metrics})
        return row
    est = weights["dw"].reindex(STATES).to_numpy()
    err = np.abs(est - TRUE_DWS)
    row["spearman"] = spearmanr(est, TRUE_DWS).statistic
    row["max_abs_err"] = err.max()
    row["mean_abs_err"] = err.mean()
    row["anchor_slope"] = diag["anchor_map"].slope
    row["anchor_r2"] = diag["anchor_map"].r_squared
    return row


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    t_start = time.perf_counter()
    for cell_idx, (label, knob, severity, module, overrides) in enumerate(CELLS):
        t_cell = time.perf_counter()
        for rep in range(N_REPS):
            row = run_rep(cell_idx, rep, module, overrides)
            row.update(knob=knob, severity=severity, cell_label=label)
            rows.append(row)
        print(
            f"[{time.perf_counter() - t_start:6.1f}s] {label}: "
            f"{time.perf_counter() - t_cell:.1f}s for {N_REPS} reps",
            flush=True,
        )
    df = pd.DataFrame(rows)[
        [
            "knob",
            "severity",
            "cell_label",
            "rep",
            "pc_seed",
            "phe_seed",
            "failed",
            "spearman",
            "max_abs_err",
            "mean_abs_err",
            "anchor_slope",
            "anchor_r2",
        ]
    ]
    csv_path = RESULTS_DIR / "misspecification.csv"
    df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path} ({len(df)} rows)")

    # Per-cell summary (mean and sd over successful reps; failures counted).
    metrics = ["spearman", "max_abs_err", "mean_abs_err", "anchor_slope", "anchor_r2"]
    g = df.groupby("cell_label", sort=False)
    summary = pd.concat(
        [g[metrics].mean().add_suffix("_mean"), g[metrics].std().add_suffix("_sd")], axis=1
    )
    summary["n_failed"] = g["failed"].sum().astype(int)
    order = [m + s for m in metrics for s in ("_mean", "_sd")] + ["n_failed"]
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format):
        print(summary[order].to_string())
    print(f"total runtime: {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
