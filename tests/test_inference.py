"""Smoke test for the respondent-cluster bootstrap; the full coverage Monte
Carlo lives in studies/coverage.py (too slow for the unit suite)."""

import functools
import warnings

import numpy as np
import pandas as pd
import pytest

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
    # A well-supported design clears the per-state support floor everywhere.
    assert res.weights["supported"].all()
    assert res.weights[["se", "lo", "hi"]].notna().all().all()


# --- regression tests for the audit's blocks-nightly findings (B1, B2) ------
# Smaller design than the smoke test so several bootstrap runs stay cheap.

K6 = 6
STATES6 = make_states(K6)
# True dws in ~[0.05, 0.7]: keeps one-sided PHE states improbable at this n.
TRUE6 = simulate_true_dws(K6, low=0.05, high=0.7)
PC6 = simulate_pc(TRUE6, STATES6, n_respondents=150, pairs_per_respondent=8, rng=11)
PHE6 = simulate_phe(TRUE6, STATES6, n_respondents=200, rng=12)
SEED = 7
N_BOOT = 30


def _two_batches(df: pd.DataFrame, ignore_index: bool) -> pd.DataFrame:
    """The same rows as two concatenated halves — the nightly paging shape.

    ignore_index=False leaves labels 0..m-1 appearing twice (the B1 hazard);
    ignore_index=True restores a clean RangeIndex over identical rows.
    """
    m = len(df) // 2
    a = df.iloc[:m].reset_index(drop=True)
    b = df.iloc[m:].reset_index(drop=True)
    return pd.concat([a, b], ignore_index=ignore_index)


@functools.cache
def _clean_run():
    """Clean-index baseline run, reused across tests; warnings recorded."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = bootstrap_dws(
            _two_batches(PC6, True), _two_batches(PHE6, True), n_boot=N_BOOT, rng=SEED
        )
    return res, caught


def test_bootstrap_label_immunity():
    """B1 regression: bootstrap output must not depend on index labels.

    The defect (audit B1): _take fetched respondent rows with df.loc on
    labels harvested from the caller's index; under duplicate labels
    (pd.concat of batches WITHOUT ignore_index=True — exactly how the
    nightly job pages responses out of Postgres) .loc fans out, so every
    replicate silently mixes rows across respondents: identical point
    estimates, n_failed=0, standard errors shrunk ~30%. The dup leg below
    catches that; the shifted-label leg (unique but non-range index) passed
    even pre-fix and pins the positional resampling mechanism against any
    future reintroduction of label sensitivity.

    Verified to fail against the pre-fix _take (run before the fix was
    applied): assert_frame_equal on the dup leg reported 'se values are
    different (100.0 %)', with every dup se shrunk to 0.67-0.73 of the clean
    run's (s000 0.005475 vs 0.007972, ..., s005 0.015803 vs 0.021584), at
    identical dw and n_failed=0 — the silent-corruption signature.
    """
    res_clean, _ = _clean_run()
    res_dup = bootstrap_dws(
        _two_batches(PC6, False), _two_batches(PHE6, False), n_boot=N_BOOT, rng=SEED
    )
    res_shift = bootstrap_dws(
        PC6.set_axis(PC6.index + 1000), PHE6.set_axis(PHE6.index + 1000), n_boot=N_BOOT, rng=SEED
    )
    pd.testing.assert_frame_equal(res_dup.weights, res_clean.weights, check_exact=True)
    pd.testing.assert_frame_equal(res_shift.weights, res_clean.weights, check_exact=True)
    assert res_dup.n_failed == res_clean.n_failed == res_shift.n_failed


def _with_thin_state(pc: pd.DataFrame) -> pd.DataFrame:
    """Append state 'sthin' supported by exactly ONE respondent: 3 PC rows
    against three distinct well-supported states, y mixed (1 win of 3) so
    neither the full sample nor any resample can go separated on sthin."""
    extra = pd.DataFrame(
        {
            "respondent_id": [10_000, 10_000, 10_000],
            "state_1": ["sthin", "sthin", "sthin"],
            "state_2": ["s000", "s001", "s002"],
            "y": [1, 0, 0],
        }
    )
    return pd.concat([pc, extra], ignore_index=True)


def test_bootstrap_thin_state_blanked():
    """B2 regression: a 1-respondent state must not publish an interval.

    The defect (audit B2): the whole-run fragility gate passes while a state
    supported by a single respondent publishes a spuriously tight interval
    (the audit measured half-width ~5x smaller than the effect of flipping
    one of the respondent's 3 answers) with no warning — defeating an
    automated CI-width promotion gate. Two-leg point: at n_boot=40 the thin
    state is estimable in ~63% of replicates (n_reps 22 here, ABOVE the rep
    floor of 20), so this test also proves the respondent leg fires where an
    n_reps-only gate would stay silent — the audit's core B2 observation.

    Verified to fail against the pre-fix code (run before the fix was
    applied): pytest.warns reported DID NOT WARN (no warnings emitted at
    all), and the pre-fix run published finite se = 0.01316 at sthin with
    n_reps 22 — tighter than the se of all four heavier well-supported
    states (0.0176-0.0299) despite resting on a single respondent.
    """
    pc = _with_thin_state(PC6)
    with pytest.warns(UserWarning, match="support floor"):
        res = bootstrap_dws(pc, PHE6, n_boot=40, rng=SEED)
    w = res.weights
    assert w.loc["sthin", ["se", "lo", "hi"]].isna().all()
    assert np.isfinite(w.loc["sthin", "dw"])  # point estimate retained
    assert not w.loc["sthin", "supported"]
    assert w.loc["sthin", "n_resp_pc"] == 1
    # The respondent leg fired, not the rep leg: n_reps clears its floor.
    assert w.loc["sthin", "n_reps"] >= max(20, 40 // 2)
    # The gate changed nothing at the well-supported states.
    base = w.drop(index="sthin")
    assert base["supported"].all()
    assert base[["se", "lo", "hi"]].notna().all().all()
    assert (base["se"] > 0).all()


def test_bootstrap_well_supported_stays_silent():
    """B2: on a well-supported design the support gate must not fire —
    no warning, every state supported, no blanked cells."""
    res, caught = _clean_run()
    assert not any("support floor" in str(w.message) for w in caught)
    assert res.weights["supported"].all()
    assert res.weights[["se", "lo", "hi"]].notna().all().all()


def test_bootstrap_rep_floor_wiring():
    """B2, second leg: the n_reps floor is independently wired.

    A floor no state can meet (min_state_reps=26 at n_boot=25) must warn and
    blank every state even with the respondent floor disabled. Constructing
    an organic high-respondents/low-n_reps state is contrived; the leg's
    statistical content (selection bias of percentile endpoints computed
    over the minority of replicates in which a fragile state happened to be
    estimable) is justified in the bootstrap_dws docstring — its wiring is
    what needs a test.

    Verified to fail against the pre-fix code (run before the fix was
    applied): TypeError — bootstrap_dws passed the then-unknown keywords
    through **estimate_kwargs and estimate_dws raised "unexpected keyword
    argument 'min_state_respondents'".
    """
    with pytest.warns(UserWarning, match="support floor"):
        res = bootstrap_dws(
            PC6, PHE6, n_boot=25, rng=SEED, min_state_respondents=0, min_state_reps=26
        )
    w = res.weights
    assert not w["supported"].any()
    assert w[["se", "lo", "hi"]].isna().all().all()
    assert (w["n_reps"] < 26).all()  # the leg that fired
    assert np.isfinite(w["dw"]).all()  # point estimates retained everywhere
