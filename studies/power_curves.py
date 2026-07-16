"""Power curves for survey planning: estimation error vs sample size and design choices.

This is a design study, not a correctness test. It provides the evidence a survey
planner needs to size the instrument: how max and mean absolute error in the
recovered disability weights fall with the number of respondents, with pairs per
respondent, and with the width of the PHE anchor set, plus a bootstrap cross-check
that the cheap across-rep precision proxy tracks the inference module's intervals.

Outputs:
  studies/results/power_curves.csv  — one row per (axis, design cell)
  numbers quoted in RESULTS-power.md come from stdout of this run
"""

import sys

sys.path.insert(0, "/home/richard/projects/WelfareWeights/engine/src")

import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import linregress

from welfareweights.inference import bootstrap_dws
from welfareweights.pipeline import estimate_dws
from welfareweights.simulate import make_states, simulate_pc, simulate_phe, simulate_true_dws

# K=20 matches test_recovery.py's convention and, decisively, keeps both anchor
# widths above fit_anchor_map's hard 3-shared-state minimum: the narrow set
# dw in [0.10, 0.40] holds 4 states and the mid set dw in [0.05, 0.70] holds 9.
# K=12 would leave only 2 narrow states — the narrow cell would raise
# "need at least three states" instead of measuring a precision tradeoff.
K = 20
STATES = make_states(K)
TRUE_DWS = simulate_true_dws(K)

ANCHOR_MID = [s for s, dw in zip(STATES, TRUE_DWS) if 0.05 <= dw <= 0.70]
ANCHOR_NARROW = [s for s, dw in zip(STATES, TRUE_DWS) if 0.10 <= dw <= 0.40]

# R=20 seeded reps per cell — well above the >=6 floor. Chosen because a single
# pipeline fit costs ~0.06-0.5 s at these designs, so 8 cells x 20 reps stays
# around a minute; more reps tighten the across-rep SD proxy at negligible cost.
N_REPS = 20

# PHE questions per respondent stays at simulate_phe's default (3) throughout:
# it is the GBD instrument's fixed format, not a design margin under study.
BASELINE = dict(n_respondents=1000, pairs=10, anchor_width="mid")

ANCHORS = {"mid": ANCHOR_MID, "narrow": ANCHOR_NARROW}


def run_cell(cell_id: int, n_respondents: int, pairs: int, anchor_width: str) -> dict:
    """Simulate + estimate N_REPS times at one design point; return cell metrics."""
    anchor_states = ANCHORS[anchor_width]
    max_errs, mean_errs, est_rows = [], [], []
    n_failed = 0
    t0 = time.perf_counter()
    for rep in range(N_REPS):
        # Distinct seeds per cell, per rep, and per module (pc vs phe), same
        # deliberate-separation pattern as test_recovery.py's rng=7 vs rng=11.
        seed_pc = 10_000 * cell_id + rep
        seed_phe = seed_pc + 5_000
        pc = simulate_pc(
            TRUE_DWS, STATES, n_respondents=n_respondents, pairs_per_respondent=pairs, rng=seed_pc
        )
        phe = simulate_phe(
            TRUE_DWS, STATES, n_respondents=n_respondents, anchor_states=anchor_states, rng=seed_phe
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # house pattern, inference.py
                weights, _ = estimate_dws(pc, phe)
        except (ValueError, np.linalg.LinAlgError):
            n_failed += 1  # same failure accounting as bootstrap_dws
            continue
        est = weights["dw"].reindex(STATES).to_numpy()
        err = est - TRUE_DWS
        max_errs.append(np.max(np.abs(err)))
        mean_errs.append(np.mean(np.abs(err)))
        est_rows.append(est)
    wall = time.perf_counter() - t0
    est_mat = np.vstack(est_rows)  # (reps_ok, K)
    mean_max_err = float(np.mean(max_errs))
    mean_mean_err = float(np.mean(mean_errs))
    return {
        "n_respondents": n_respondents,
        "pairs_per_respondent": pairs,
        "anchor_width": anchor_width,
        "k_states": K,
        "n_reps": N_REPS,
        "n_failed": n_failed,
        "mean_max_err": mean_max_err,
        "mean_mean_err": mean_mean_err,
        "err_ratio": mean_max_err / mean_mean_err,
        # Cheap precision proxy: per-state SD of the dw estimate across reps,
        # averaged over states — stands in for a bootstrap SE at 1/n_boot the cost.
        "mean_across_rep_sd": float(est_mat.std(axis=0, ddof=1).mean()),
        "total_pc_responses": n_respondents * pairs,
        "total_phe_responses": n_respondents * 3,
        "wall_time_s": round(wall, 2),
    }


def main() -> None:
    t_start = time.perf_counter()

    # 8 distinct design cells; the baseline cell (n=1000, pairs=10, mid) is
    # shared by all three axes and computed exactly once.
    cells = [
        (0, 250, 10, "mid"),
        (1, 500, 10, "mid"),
        (2, 1000, 10, "mid"),  # baseline, reused by all three axes
        (3, 2000, 10, "mid"),
        (4, 4000, 10, "mid"),
        (5, 1000, 5, "mid"),
        (6, 1000, 15, "mid"),
        (7, 1000, 10, "narrow"),
    ]
    results = {}
    for cell_id, n, pairs, width in cells:
        results[(n, pairs, width)] = run_cell(cell_id, n, pairs, width)
        r = results[(n, pairs, width)]
        print(
            f"cell n={n} pairs={pairs} anchor={width}: "
            f"mean_max_err={r['mean_max_err']:.4f} mean_mean_err={r['mean_mean_err']:.4f} "
            f"sd={r['mean_across_rep_sd']:.4f} failed={r['n_failed']} ({r['wall_time_s']}s)"
        )

    # One row per axis membership so each axis plots independently; the
    # baseline row appears under all three axis tags.
    rows = []
    for n in (250, 500, 1000, 2000, 4000):
        rows.append({"axis": "n_respondents", **results[(n, 10, "mid")]})
    for pairs in (5, 10, 15):
        rows.append({"axis": "pairs_per_respondent", **results[(1000, pairs, "mid")]})
    for width in ("narrow", "mid"):
        rows.append({"axis": "anchor_width", **results[(1000, 10, width)]})
    out = pd.DataFrame(rows)
    out.to_csv("/home/richard/projects/WelfareWeights/engine/studies/results/power_curves.csv", index=False)

    # Consistency check: root-N convergence implies slope -0.5 in log-log.
    ns = np.array([250, 500, 1000, 2000, 4000], dtype=float)
    errs = np.array([results[(int(n), 10, "mid")]["mean_mean_err"] for n in ns])
    fit = linregress(np.log(ns), np.log(errs))
    print(f"\nlog-log slope of mean_mean_err on n_respondents: {fit.slope:.3f} (r^2={fit.rvalue**2:.3f})")

    # Bootstrap cross-check at the baseline point. n_boot=80: above the >=60
    # floor for a steadier mean interval width, still ~20 s. Seeds 999_000/999_001
    # sit outside the 10_000*cell_id + rep grid, so no draw is shared with any cell.
    pc = simulate_pc(TRUE_DWS, STATES, n_respondents=1000, pairs_per_respondent=10, rng=999_000)
    phe = simulate_phe(TRUE_DWS, STATES, n_respondents=1000, anchor_states=ANCHOR_MID, rng=999_001)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        boot = bootstrap_dws(pc, phe, n_boot=80, rng=999_002)
    boot_width = float((boot.weights["hi"] - boot.weights["lo"]).mean())
    proxy_sd = results[(1000, 10, "mid")]["mean_across_rep_sd"]
    proxy_width = 2 * 1.96 * proxy_sd
    print(
        f"bootstrap mean 95% CI width = {boot_width:.4f} (n_boot=80, n_failed={boot.n_failed}); "
        f"proxy-implied width 2*1.96*SD = {proxy_width:.4f}; ratio boot/proxy = {boot_width / proxy_width:.2f}"
    )

    # Design guidance inputs: smallest n meeting the planning bar. 0.05 is the
    # bar because GBD-scale weights are reported to ~2 decimals and adjacent
    # states in the published tables differ by ~0.03-0.10; max error above 0.05
    # would reorder neighbors.
    for n in (250, 500, 1000, 2000, 4000):
        r = results[(n, 10, "mid")]
        flag = "PASS" if r["mean_max_err"] < 0.05 else "fail"
        print(f"n={n}: mean_max_err={r['mean_max_err']:.4f} [{flag}] total_responses={n * 10 + n * 3}")

    print(f"\ntotal runtime: {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
