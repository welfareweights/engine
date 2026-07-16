"""Estimator precision at the published GBD 2010 design scale, on synthetic known-truth data.

This is a design study, not a correctness test. It characterizes how precisely the
pipeline recovers 220 known disability weights when fed a survey of the size Salomon
2012 actually fielded per instrument arm: 14,000 paired-comparison respondents at 15
pairs each (210,000 PC rows) and ~4,000 PHE respondents at 3 questions each over 30
mid-range anchor states with the GBD c-grid. Everything is simulated from the assumed
DGP, so the numbers measure design-scale sampling precision only — comparison against
the published GBD uncertainty intervals is left to the reader (those tables sit outside
this repo's engine surface by policy).

Outputs:
  studies/results/gbd_scale.csv — one row per (seed, state), per-seed scalars broadcast
  numbers quoted in RESULTS-gbd-scale.md come from stdout of this run
"""

import sys

sys.path.insert(0, "/home/richard/projects/WelfareWeights/engine/src")

import os
import time

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import spearmanr

from welfareweights.pipeline import estimate_dws
from welfareweights.simulate import make_states, simulate_pc, simulate_phe, simulate_true_dws

# Published GBD 2010 web-survey scale (Salomon 2012): ~14,000 PC respondents x 15
# pairs, ~4,000 PHE respondents x 3 questions, K=220 states. true dw range
# [0.003, 0.95] per the study spec — note simulate_true_dws's own default low is
# 0.005, so low is passed explicitly.
K = 220
STATES = make_states(K)
TRUE_DWS = simulate_true_dws(K, low=0.003, high=0.95)
TRUE_LOGIT = logit(TRUE_DWS)

N_PC = 14_000
PAIRS = 15
N_PHE = 4_000
QUESTIONS = 3

# DGP constants match tests/test_recovery.py (slope=-1.0, intercept=0.3, sigma=0.8):
# a chosen convention for internal consistency, not a rubric mandate.
SLOPE, INTERCEPT, SIGMA = -1.0, 0.3, 0.8

# Anchor subset: the rubric asks for exactly 30 states with true dw in [0.05, 0.70]
# (mid-range, where the PHE thresholds logit(1000/c) actually bracket the latent
# distribution — same band as test_recovery.py's ANCHOR_STATES). 95 of the 220
# states fall in the band; the 30 are taken evenly spaced by rank within it, an
# explicit design choice since the rubric does not pin the within-band rule.
ELIGIBLE = [i for i, dw in enumerate(TRUE_DWS) if 0.05 <= dw <= 0.70]
ANCHOR_IDX = sorted({ELIGIBLE[j] for j in np.linspace(0, len(ELIGIBLE) - 1, 30).round().astype(int)})
ANCHOR_STATES = [STATES[i] for i in ANCHOR_IDX]
assert len(ANCHOR_STATES) == 30

SEEDS = (0, 1)
# Runtime contingency (rubric item 5): if the first seed's fit projects the total
# past the budget, halve RESPONDENTS (not states) for the remaining seeds. Budget
# is set below the hard ~10-minute cap to leave headroom for simulation + I/O.
BUDGET_S = 480.0

# Deciles of the true weight, 22 states per bin, labeled 1 (lightest) .. 10 (heaviest).
TRUE_DECILE = pd.qcut(TRUE_DWS, 10, labels=False) + 1


def run_seed(seed: int, n_pc: int, n_phe: int) -> tuple[pd.DataFrame, dict]:
    """Simulate one replicate at the given respondent counts, fit, return per-state rows.

    pc rng=seed and phe rng=seed+1000 decorrelates the two modules' draws within a
    replicate (same deliberate-separation pattern as test_recovery.py's rng=7 vs 11).
    Only this seed's data frames exist at a time; the ~370MB dense design matrix is
    transient inside fit_pc and freed on return — nothing here rebuilds it.
    """
    pc_df = simulate_pc(
        TRUE_DWS, STATES, n_respondents=n_pc, pairs_per_respondent=PAIRS,
        slope=SLOPE, intercept=INTERCEPT, rng=seed,
    )
    phe_df = simulate_phe(
        TRUE_DWS, STATES, n_respondents=n_phe, questions_per_respondent=QUESTIONS,
        sigma=SIGMA, anchor_states=ANCHOR_STATES, rng=seed + 1000,
    )
    t0 = time.perf_counter()
    weights, diag = estimate_dws(pc_df, phe_df)
    wall = time.perf_counter() - t0

    est = weights["dw"].reindex(STATES).to_numpy()
    est_logit = weights["logit_dw"].reindex(STATES).to_numpy()  # tau=0: logit(dw) exactly
    err = est - TRUE_DWS
    logit_err = est_logit - TRUE_LOGIT
    amap = diag["anchor_map"]
    rows = pd.DataFrame(
        {
            "seed": seed,
            "state": STATES,
            "true_dw": TRUE_DWS,
            "true_decile": TRUE_DECILE,
            "est_dw": est,
            "err": err,
            "abs_err": np.abs(err),
            "logit_err": logit_err,
            "logit_abserr": np.abs(logit_err),
            "in_anchor_set": [s in set(ANCHOR_STATES) for s in STATES],
            "pc_respondents": n_pc,
            "phe_respondents": n_phe,
            "anchor_slope": amap.slope,
            "anchor_r2": amap.r_squared,
            "n_shared": amap.n_shared,
            "wall_clock_s": round(wall, 2),
        }
    )
    summary = {
        "seed": seed,
        "rank_corr": float(spearmanr(est, TRUE_DWS).statistic),
        "mean_abs_err": float(np.mean(np.abs(err))),
        "max_abs_err": float(np.max(np.abs(err))),
        "anchor_slope": amap.slope,
        "anchor_r2": amap.r_squared,
        "n_shared": amap.n_shared,
        "one_sided_dropped": len(diag["one_sided_dropped"]),
        "wall_clock_s": wall,
    }
    return rows, summary


def main() -> None:
    t_start = time.perf_counter()
    print(f"K={K} states, true dw in [{TRUE_DWS[0]:.3f}, {TRUE_DWS[-1]:.2f}] (logit-spaced)")
    print(f"anchors: {len(ANCHOR_STATES)} of {len(ELIGIBLE)} eligible (dw in [0.05, 0.70]), "
          f"dw range [{TRUE_DWS[ANCHOR_IDX[0]]:.3f}, {TRUE_DWS[ANCHOR_IDX[-1]]:.3f}]")

    n_pc, n_phe = N_PC, N_PHE
    halved = False
    all_rows, summaries = [], []
    for i, seed in enumerate(SEEDS):
        rows, s = run_seed(seed, n_pc, n_phe)
        all_rows.append(rows)
        summaries.append(s)
        print(
            f"seed {seed} (pc={n_pc}x{PAIRS}, phe={n_phe}x{QUESTIONS}): "
            f"rank_corr={s['rank_corr']:.5f} mean|err|={s['mean_abs_err']:.5f} "
            f"max|err|={s['max_abs_err']:.5f} anchor_slope={s['anchor_slope']:.4f} "
            f"anchor_r2={s['anchor_r2']:.4f} n_shared={s['n_shared']} "
            f"one_sided_dropped={s['one_sided_dropped']} fit={s['wall_clock_s']:.1f}s"
        )
        # Contingency check after the first fit: project remaining fits at the
        # measured per-fit cost (fit time scales ~linearly in PC rows, the design
        # matrix build and probit iterations both being O(rows) — the scaling
        # assumption; halving respondents halves rows and roughly halves the fit).
        elapsed = time.perf_counter() - t_start
        remaining = len(SEEDS) - (i + 1)
        projected = elapsed + remaining * (elapsed / (i + 1))
        if remaining and projected > BUDGET_S and not halved:
            n_pc, n_phe = n_pc // 2, n_phe // 2
            halved = True
            print(
                f"projected total {projected:.0f}s > budget {BUDGET_S:.0f}s: halving "
                f"respondents (not states) to pc={n_pc}, phe={n_phe} for remaining seeds; "
                f"assumes fit time linear in PC rows"
            )
    if not halved:
        print(f"runtime contingency not triggered (budget {BUDGET_S:.0f}s)")

    out = pd.concat(all_rows, ignore_index=True)
    results_dir = "/home/richard/projects/WelfareWeights/engine/studies/results"
    os.makedirs(results_dir, exist_ok=True)
    out.to_csv(os.path.join(results_dir, "gbd_scale.csv"), index=False)
    print(f"wrote {len(out)} rows to {results_dir}/gbd_scale.csv")

    # Pooled headline metrics (both seeds).
    pooled_rank = float(spearmanr(out["est_dw"], out["true_dw"]).statistic)
    print(
        f"pooled: rank_corr={pooled_rank:.5f} mean|err|={out['abs_err'].mean():.5f} "
        f"max|err|={out['abs_err'].max():.5f}"
    )

    # Error by true-weight decile, pooled over seeds, in BOTH scales: the affine
    # anchor map is fit on 30 mid-range states and extrapolated linearly in logit
    # space; expit's shrinking derivative near 0/1 compresses any logit-space error
    # toward zero in probability space, so the two scales rank the deciles oppositely.
    dec = out.groupby("true_decile").agg(
        dw_lo=("true_dw", "min"), dw_hi=("true_dw", "max"),
        mean_abs=("abs_err", "mean"), med_abs=("abs_err", "median"), max_abs=("abs_err", "max"),
        mean_labs=("logit_abserr", "mean"), med_labs=("logit_abserr", "median"),
        max_labs=("logit_abserr", "max"),
    )
    print("\nabs error by true-weight decile (pooled over seeds; prob scale | logit scale):")
    print(f"{'dec':>3} {'dw range':>15} {'mean':>8} {'med':>8} {'max':>8} | {'mean':>7} {'med':>7} {'max':>7}")
    for d, r in dec.iterrows():
        print(
            f"{d:>3} [{r.dw_lo:.3f}, {r.dw_hi:.3f}] {r.mean_abs:8.5f} {r.med_abs:8.5f} "
            f"{r.max_abs:8.5f} | {r.mean_labs:7.4f} {r.med_labs:7.4f} {r.max_labs:7.4f}"
        )

    print(f"\ntotal runtime: {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
