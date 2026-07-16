"""Monte Carlo coverage validation of inference.bootstrap_dws percentile intervals.

Simulate n_mc surveys from the exact DGP the estimator assumes (K=8 states,
600 PC respondents x 8 pairs, 900 PHE respondents x 3 questions), run the
respondent-cluster bootstrap (n_boot=100, level=0.95) on each, and count how
often the percentile interval [lo, hi] and the normal interval dw +/- z*se
contain the known true dw. Nominal coverage is 0.95; the Monte Carlo binomial
error at n_mc reps is 1.96*sqrt(0.95*0.05/n_mc) (~+/-4.3pp at n_mc=100,
~+/-5.5pp at n_mc=60).

Verdict criterion, fixed before looking at results: mean coverage across
states within [0.88, 0.99] passes at this MC resolution; per-state dips
outside that band are flagged individually. The extreme states (s000, s007)
are expected to be worst: their PHE responses come back one-sided every rep,
so their dw is anchor-extrapolated rather than directly anchored.

Timing rule: the first MC rep is timed in full; if the projection for
n_mc=100 exceeds 12 minutes, the study drops to n_mc=60 and widens the
stated MC error accordingly.

Writes studies/results/coverage.csv and studies/RESULTS-coverage.md.
"""

import sys

sys.path.insert(0, "/home/richard/projects/WelfareWeights/engine/src")

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from welfareweights.inference import bootstrap_dws
from welfareweights.simulate import make_states, simulate_pc, simulate_phe, simulate_true_dws

ENGINE = Path("/home/richard/projects/WelfareWeights/engine")
RESULTS_DIR = ENGINE / "studies" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

K = 8
STATES = make_states(K)
TRUE_DWS = simulate_true_dws(K)  # module-level constant: same truth in every rep
N_PC, PAIRS = 600, 8
N_PHE, QUESTIONS = 900, 3
SLOPE, INTERCEPT = -1.0, 0.3
SIGMA = 0.8
N_BOOT = 100
LEVEL = 0.95
N_MC_FULL, N_MC_REDUCED = 100, 60
TIME_BUDGET_S = 12 * 60  # projection past this triggers the n_mc=60 fallback
MASTER_SEED = 20260716
Z = norm.ppf(0.975)  # normal-interval multiplier, not hardcoded 1.96


def run_rep(seed_seq: np.random.SeedSequence):
    """One MC rep: simulate both modules, bootstrap, return per-state hits.

    Returns (hit_pct, hit_norm, dw, se, boot_n_failed) as Series/int, or None
    if the rep itself fails estimation (counted separately from bootstrap-
    replicate failures inside a successful rep).
    """
    pc_seed, phe_seed, boot_seed = seed_seq.spawn(3)
    pc_df = simulate_pc(
        TRUE_DWS,
        STATES,
        n_respondents=N_PC,
        pairs_per_respondent=PAIRS,
        slope=SLOPE,
        intercept=INTERCEPT,
        rng=np.random.default_rng(pc_seed),
    )
    phe_df = simulate_phe(
        TRUE_DWS,
        STATES,
        n_respondents=N_PHE,
        questions_per_respondent=QUESTIONS,
        sigma=SIGMA,
        rng=np.random.default_rng(phe_seed),
    )
    try:
        with warnings.catch_warnings():
            # One-sided-drop and anchor-R^2 warnings fire by design every rep.
            warnings.simplefilter("ignore")
            boot = bootstrap_dws(
                pc_df,
                phe_df,
                n_boot=N_BOOT,
                level=LEVEL,
                rng=np.random.default_rng(boot_seed),
            )
    except (ValueError, np.linalg.LinAlgError):
        return None
    w = boot.weights.reindex(STATES)
    true = pd.Series(TRUE_DWS, index=STATES)
    hit_pct = (w["lo"] <= true) & (true <= w["hi"])
    hit_norm = (w["dw"] - Z * w["se"] <= true) & (true <= w["dw"] + Z * w["se"])
    return hit_pct, hit_norm, w["dw"], w["se"], boot.n_failed


def main():
    master = np.random.SeedSequence(MASTER_SEED)
    rep_seeds = master.spawn(N_MC_FULL)  # spawn the full set; use a prefix if reduced

    t_total0 = time.perf_counter()

    # Time rep 0 in full to decide n_mc (rubric #5).
    t0 = time.perf_counter()
    first = run_rep(rep_seeds[0])
    rep0_s = time.perf_counter() - t0
    projected_full_s = rep0_s * N_MC_FULL
    if projected_full_s > TIME_BUDGET_S:
        n_mc = N_MC_REDUCED
    else:
        n_mc = N_MC_FULL
    mc_err = Z * np.sqrt(LEVEL * (1 - LEVEL) / n_mc)
    print(
        f"rep 0: {rep0_s:.1f}s; projected n_mc={N_MC_FULL} total "
        f"{projected_full_s/60:.1f} min -> using n_mc={n_mc} "
        f"(MC binomial error +/-{100*mc_err:.1f}pp)"
    )

    hits_pct, hits_norm, dws, ses = [], [], [], []
    boot_failed_total = 0
    n_mc_failed = 0

    def collect(res):
        nonlocal boot_failed_total, n_mc_failed
        if res is None:
            n_mc_failed += 1
            return
        hp, hn, dw, se, bf = res
        hits_pct.append(hp)
        hits_norm.append(hn)
        dws.append(dw)
        ses.append(se)
        boot_failed_total += bf

    collect(first)
    for i in range(1, n_mc):
        collect(run_rep(rep_seeds[i]))
        if (i + 1) % 10 == 0:
            elapsed = time.perf_counter() - t_total0
            print(f"  rep {i + 1}/{n_mc} done, {elapsed/60:.1f} min elapsed", flush=True)

    total_s = time.perf_counter() - t_total0
    n_ok = len(hits_pct)

    cov_pct = pd.concat(hits_pct, axis=1).mean(axis=1)
    cov_norm = pd.concat(hits_norm, axis=1).mean(axis=1)
    mean_dw = pd.concat(dws, axis=1).mean(axis=1)
    mean_se = pd.concat(ses, axis=1).mean(axis=1)

    out = pd.DataFrame(
        {
            "state": STATES,
            "true_dw": TRUE_DWS,
            "mean_point_dw": mean_dw.to_numpy(),
            "mean_boot_se": mean_se.to_numpy(),
            "coverage_percentile": cov_pct.to_numpy(),
            "coverage_normal": cov_norm.to_numpy(),
            "n_mc_used": n_ok,
        }
    )
    mean_row = pd.DataFrame(
        {
            "state": ["MEAN"],
            "true_dw": [np.nan],
            "mean_point_dw": [np.nan],
            "mean_boot_se": [np.nan],
            "coverage_percentile": [cov_pct.mean()],
            "coverage_normal": [cov_norm.mean()],
            "n_mc_used": [n_ok],
        }
    )
    out = pd.concat([out, mean_row], ignore_index=True)
    csv_path = RESULTS_DIR / "coverage.csv"
    out.to_csv(csv_path, index=False)

    boot_fail_rate = boot_failed_total / (n_ok * N_BOOT) if n_ok else np.nan
    band_lo, band_hi = 0.88, 0.99
    verdict_pct = "PASS" if band_lo <= cov_pct.mean() <= band_hi else "FAIL"
    verdict_norm = "PASS" if band_lo <= cov_norm.mean() <= band_hi else "FAIL"
    dips_pct = [s for s, c in zip(STATES, cov_pct) if not (band_lo <= c <= band_hi)]
    dips_norm = [s for s, c in zip(STATES, cov_norm) if not (band_lo <= c <= band_hi)]

    rows = "\n".join(
        f"| {s} | {t:.4f} | {d:.4f} | {e:.4f} | {cp:.3f} | {cn:.3f} |"
        for s, t, d, e, cp, cn in zip(STATES, TRUE_DWS, mean_dw, mean_se, cov_pct, cov_norm)
    )
    md = f"""# Coverage of bootstrap_dws intervals — Monte Carlo validation

**Verdict criterion (stated before results):** mean coverage across states within [0.88, 0.99] passes at this MC resolution. Nominal coverage is 0.95; the Monte Carlo binomial error at n_mc={n_mc} is +/-{100*mc_err:.1f}pp (1.96*sqrt(0.95*0.05/{n_mc})), so the band absorbs MC noise plus modest genuine undercoverage. Per-state coverage outside the band is flagged as a dip. The extreme states s000 and s007 are expected to be the worst performers: their PHE responses come back one-sided every rep, so estimate_dws drops them from the anchor stage and their dw is anchor-extrapolated rather than directly anchored.

**Design.** K=8 states with true dws evenly spaced in logit space over [0.005, 0.95] (simulate_true_dws defaults), held fixed across reps. Each MC rep simulates {N_PC} PC respondents x {PAIRS} pairs ({N_PC*PAIRS} rows, slope=-1.0, intercept=0.3) and {N_PHE} PHE respondents x {QUESTIONS} questions ({N_PHE*QUESTIONS} rows, sigma=0.8, all 8 states eligible), then runs bootstrap_dws with n_boot={N_BOOT}, level={LEVEL}. Seeding: one master SeedSequence({MASTER_SEED}) spawned into {N_MC_FULL} per-rep sequences, each spawned into independent pc/phe/boot streams — fully reproducible. Coverage of the percentile interval is the fraction of reps with lo <= true_dw <= hi; coverage of the normal interval uses dw +/- {Z:.4f}*se from the bootstrap's own point estimate and replicate SD.

**Timing and n_mc decision (rubric #5).** The first rep (simulate + point estimate + {N_BOOT}-replicate bootstrap) took {rep0_s:.1f}s, projecting {projected_full_s/60:.1f} min for n_mc={N_MC_FULL}; the 12-minute trigger {"fired, so the study ran at n_mc=" + str(N_MC_REDUCED) + " and the MC error widened to +/-" + f"{100*mc_err:.1f}" + "pp" if n_mc == N_MC_REDUCED else "did not fire, so the study ran at the full n_mc=" + str(N_MC_FULL)}. Actual total wall clock: {total_s/60:.1f} min.

**Failure rates (rubric #4).** MC-rep-level failures (a whole rep raising in estimation or the bootstrap's own too-few-replicates guard): {n_mc_failed}/{n_mc} ({100*n_mc_failed/n_mc:.1f}%). Bootstrap-replicate-level failures inside successful reps (bootstrap_dws n_failed summed): {boot_failed_total}/{n_ok*N_BOOT} ({100*boot_fail_rate:.2f}%).

## Results ({n_ok} successful MC reps)

| state | true_dw | mean_point_dw | mean_boot_se | coverage_percentile | coverage_normal |
|---|---|---|---|---|---|
{rows}
| **MEAN** | | | | **{cov_pct.mean():.3f}** | **{cov_norm.mean():.3f}** |

**Per-state dips.** Percentile interval: {", ".join(dips_pct) if dips_pct else "none outside [0.88, 0.99]"}. Normal interval: {", ".join(dips_norm) if dips_norm else "none outside [0.88, 0.99]"}.

**Verdict.** Percentile interval: mean coverage {cov_pct.mean():.3f} -> **{verdict_pct}**. Normal interval (dw +/- {Z:.2f}se): mean coverage {cov_norm.mean():.3f} -> **{verdict_norm}**. Both are read against the [0.88, 0.99] band with +/-{100*mc_err:.1f}pp MC error at n_mc={n_mc}.

Raw per-state numbers: `studies/results/coverage.csv`. Reproduce with `.venv/bin/python studies/coverage.py` (master seed {MASTER_SEED}).
"""
    md_path = ENGINE / "studies" / "RESULTS-coverage.md"
    md_path.write_text(md)

    print(f"\nwrote {csv_path}")
    print(f"wrote {md_path}")
    print(f"\nmean coverage: percentile {cov_pct.mean():.3f}, normal {cov_norm.mean():.3f}")
    print(f"per-state percentile: {dict(zip(STATES, cov_pct.round(3)))}")
    print(f"per-state normal:     {dict(zip(STATES, cov_norm.round(3)))}")
    print(f"n_mc_failed={n_mc_failed}, boot replicate failure rate={100*boot_fail_rate:.2f}%")
    print(f"verdicts: percentile {verdict_pct}, normal {verdict_norm}")
    print(f"total wall clock: {total_s/60:.1f} min")


if __name__ == "__main__":
    main()
