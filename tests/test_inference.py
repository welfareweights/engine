"""Smoke test for the respondent-cluster bootstrap; the full coverage Monte
Carlo lives in studies/coverage.py (too slow for the unit suite)."""

import numpy as np

from welfareweights.inference import bootstrap_dws
from welfareweights.simulate import make_states, simulate_pc, simulate_phe, simulate_true_dws

K = 8
STATES = make_states(K)
TRUE_DWS = simulate_true_dws(K, low=0.03, high=0.75)


def test_bootstrap_smoke():
    pc = simulate_pc(TRUE_DWS, STATES, n_respondents=800, pairs_per_respondent=8, rng=3)
    phe = simulate_phe(TRUE_DWS, STATES, n_respondents=1200, rng=4)
    res = bootstrap_dws(pc, phe, n_boot=60, rng=5)

    w = res.weights.reindex(STATES)
    assert res.n_failed <= 6  # occasional one-sided resample is tolerable, not the norm
    assert (w["se"] > 0).all()
    assert (w["lo"] < w["hi"]).all()
    assert ((w["dw"] >= w["lo"]) & (w["dw"] <= w["hi"])).all()
    # Intervals should usually contain truth (exact coverage is studies/coverage.py's job).
    covered = ((TRUE_DWS >= w["lo"]) & (TRUE_DWS <= w["hi"])).mean()
    assert covered >= 0.75
