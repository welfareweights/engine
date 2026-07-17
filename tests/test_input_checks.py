"""The input-validation front door (audit findings F4, F1, F3, F7, F8; rider P6).

Every public entry point must reject malformed frames BEFORE any stage code
runs, with a ValueError naming the frame, column, and cause. The validation
LOGIC lives once, in welfareweights.checks; the parametrized test below feeds
each documented-bad input through BOTH estimate_dws and bootstrap_dws and
requires character-identical messages from the two entries — the observable
proof the checks are single-sourced rather than scattered per call site.

Pre-fix behavior is recorded in the docstrings of the silent-wrong-output
tests (F1) and the misattribution tests (F7, F8, F3), each measured by
running the test's exact trigger against the unfixed tree before the fix
was applied.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import welfareweights
from welfareweights import bootstrap_dws, estimate_dws  # top-level imports ARE rider P6
from welfareweights.anchor import fit_phe, thresholds
from welfareweights.checks import check_pc_df, check_phe_df
from welfareweights.probit import fit_pc
from welfareweights.simulate import make_states, simulate_pc, simulate_phe, simulate_true_dws
from welfareweights.validate import eval_phe

K = 6
STATES = make_states(K)
# Mid-range true dws (same regime as test_inference's PC6/PHE6): no one-sided
# PHE states at these n, so every failure below is the injected corruption.
TRUE_DWS = simulate_true_dws(K, low=0.05, high=0.7)
PC = simulate_pc(TRUE_DWS, STATES, n_respondents=100, pairs_per_respondent=6, rng=21)
PHE = simulate_phe(TRUE_DWS, STATES, n_respondents=120, rng=22)


def _with_y(df, pos, value):
    out = df.copy()
    out["y"] = out["y"].astype(float)
    out.iloc[pos, out.columns.get_loc("y")] = value
    return out


def _with_col(df, col, pos, value, dtype):
    out = df.copy()
    out[col] = out[col].astype(dtype)
    out.iloc[pos, out.columns.get_loc(col)] = value
    return out


# (case id, corruption of clean (pc, phe) copies, expected message substring).
CASES = [
    (
        "pc_missing_state_1",
        lambda pc, phe: (pc.drop(columns=["state_1"]), phe),
        "pc_df is missing required column(s) ['state_1']",
    ),
    (
        "phe_missing_n_cases",
        lambda pc, phe: (pc, phe.drop(columns=["n_cases"])),
        "phe_df is missing required column(s) ['n_cases']",
    ),
    (
        "pc_empty",
        lambda pc, phe: (pc.iloc[0:0], phe),
        "pc_df is empty: no paired-comparison responses",
    ),
    (
        "phe_empty",
        lambda pc, phe: (pc, phe.iloc[0:0]),
        "phe_df is empty: no population-health-equivalence responses",
    ),
    (
        "pc_nan_y",
        lambda pc, phe: (_with_y(pc, 3, np.nan), phe),
        "pc_df['y'] contains 1 missing value(s) (NaN)",
    ),
    (
        "phe_nan_y",
        lambda pc, phe: (pc, _with_y(phe, 3, np.nan)),
        "phe_df['y'] contains 1 missing value(s) (NaN)",
    ),
    (
        "pc_fractional_y",
        lambda pc, phe: (_with_y(pc, 3, 0.9), phe),
        "pc_df['y'] contains values other than 0 and 1: [0.9]",
    ),
    (
        "phe_fractional_y",
        lambda pc, phe: (pc, _with_y(phe, 3, 0.7)),
        "phe_df['y'] contains values other than 0 and 1: [0.7]",
    ),
    (
        "pc_string_y",
        lambda pc, phe: (pc.assign(y=pc["y"].astype(str)), phe),
        "string-typed responses must be cast to integers",
    ),
    (
        "phe_string_y",
        lambda pc, phe: (pc, phe.assign(y=phe["y"].astype(str))),
        "string-typed responses must be cast to integers",
    ),
    (
        "phe_y_coded_1_2",
        lambda pc, phe: (pc, phe.assign(y=phe["y"] + 1)),
        "phe_df['y'] contains values other than 0 and 1: [2]",
    ),
    (
        "phe_nan_n_cases",
        lambda pc, phe: (pc, _with_col(phe, "n_cases", 0, np.nan, float)),
        "phe_df['n_cases'] contains 1 missing value(s) (NaN)",
    ),
    (
        "phe_inf_n_cases",
        lambda pc, phe: (pc, _with_col(phe, "n_cases", 0, np.inf, float)),
        "phe_df['n_cases'] must be finite and positive",
    ),
    (
        "phe_nonnumeric_n_cases",
        lambda pc, phe: (pc, _with_col(phe, "n_cases", 0, "many", object)),
        "phe_df['n_cases'] contains non-numeric value(s): ['many']",
    ),
    (
        "phe_nan_deaths",
        lambda pc, phe: (pc, _with_col(phe, "deaths", 0, np.nan, float)),
        "phe_df['deaths'] contains 1 missing value(s) (NaN)",
    ),
    (
        "pc_nan_state",
        lambda pc, phe: (_with_col(pc, "state_1", 2, np.nan, object), phe),
        "pc_df['state_1'] contains 1 missing state label(s)",
    ),
]


@pytest.mark.parametrize("case_id,corrupt,substring", CASES, ids=[c[0] for c in CASES])
def test_front_door_rejects_bad_inputs_at_both_entry_points(case_id, corrupt, substring):
    """Rubric 2: no public entry point reaches stage code with an unvalidated
    frame. Each documented-bad input (the audit's F1/F4/F7/F8 triggers plus
    the designed NaN-state-label extension of the same silent class) must
    raise ValueError with the same cause-naming message from BOTH
    estimate_dws and bootstrap_dws; message identity across the two entries
    is the observable proof the validation logic is single-sourced. (The two
    missing-column cases are the one designed exception to full-message
    identity: bootstrap_dws legitimately expects respondent_id on top of the
    shared columns, so only the shared missing-column clause is compared.)"""
    messages = {}
    for entry_name, entry in [
        ("estimate_dws", lambda pc, phe: estimate_dws(pc, phe)),
        ("bootstrap_dws", lambda pc, phe: bootstrap_dws(pc, phe, n_boot=5, rng=0)),
    ]:
        pc, phe = corrupt(PC.copy(), PHE.copy())
        with pytest.raises(ValueError) as ei:
            entry(pc, phe)
        assert substring in str(ei.value), f"{entry_name} raised: {ei.value}"
        messages[entry_name] = str(ei.value)
    if "missing_" not in case_id:
        assert messages["estimate_dws"] == messages["bootstrap_dws"]


def test_front_door_requires_respondent_id_for_bootstrap():
    """bootstrap_dws resamples respondents, so it (unlike estimate_dws, which
    never reads the column) requires respondent_id in both frames and says so
    by name instead of failing after the full point-estimation run."""
    with pytest.raises(ValueError) as ei:
        bootstrap_dws(PC.drop(columns=["respondent_id"]), PHE, n_boot=5, rng=0)
    assert "pc_df is missing required column(s) ['respondent_id']" in str(ei.value)
    with pytest.raises(ValueError) as ei:
        bootstrap_dws(PC, PHE.drop(columns=["respondent_id"]), n_boot=5, rng=0)
    assert "phe_df is missing required column(s) ['respondent_id']" in str(ei.value)
    # estimate_dws is indifferent: same frame, no respondent_id, runs clean.
    estimate_dws(PC.drop(columns=["respondent_id"]), PHE)


def test_front_door_covers_direct_stage_calls():
    """F1 at the stage surface: fit_pc and fit_phe are public module-level API
    and the audit verified F1 by calling them directly, so the front door
    must cover them, not only the pipeline entries.

    Observed pre-fix behavior (this exact trigger run against the unfixed
    tree): fit_pc on 4500 rows with 200 at y=0.9 completed with NO warning
    and returned betas bit-identical to a fit with those 200 answers
    truncated to 0 — i.e. 200 FLIPPED answers — shifting betas by up to
    3.632 vs the clean fit; fit_phe with 200 rows at y=0.7 completed
    silently with logit weights shifted by up to 1.230 vs the clean fit
    (the audit measured a 0.465 shift on its own design)."""
    k10 = make_states(10)
    true10 = simulate_true_dws(10)
    pc = simulate_pc(true10, k10, n_respondents=300, pairs_per_respondent=15, rng=42)
    assert len(pc) == 4500
    pc["y"] = pc["y"].astype(float)
    pc.iloc[:200, pc.columns.get_loc("y")] = 0.9
    with pytest.raises(ValueError) as ei:
        fit_pc(pc)
    assert "pc_df['y'] contains values other than 0 and 1: [0.9]" in str(ei.value)

    mid = [s for s, d in zip(k10, true10) if 0.05 <= d <= 0.90]
    anchors = sorted(mid, key=lambda s: abs(true10[k10.index(s)] - 0.5))[:6]
    phe = simulate_phe(true10, k10, n_respondents=600, sigma=0.8, anchor_states=anchors, rng=102)
    phe["y"] = phe["y"].astype(float)
    phe.iloc[:200, phe.columns.get_loc("y")] = 0.7
    with pytest.raises(ValueError) as ei:
        fit_phe(phe)
    assert "phe_df['y'] contains values other than 0 and 1: [0.7]" in str(ei.value)


def test_front_door_clean_inputs_silent():
    """Clean frames pass both checkers with no exception and no warning, and
    the domain check is value-based, not dtype-based: bool y and float
    0.0/1.0 y are previously-correct inputs and stay accepted (rubric 5's
    clean-input-unchanged contract, pinned at the checker level)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_pc_df(PC)
        check_phe_df(PHE)
        check_pc_df(PC, require_respondent_id=True)
        check_phe_df(PHE, require_respondent_id=True)
        check_pc_df(PC.assign(y=PC["y"].astype(bool)))
        check_pc_df(PC.assign(y=PC["y"].astype(float)))
        check_phe_df(PHE.assign(y=PHE["y"].astype(bool)))
        check_phe_df(PHE.assign(y=PHE["y"].astype(float)))
        # object-dtype y holding true ints was previously correct and stays so.
        check_phe_df(PHE.assign(y=PHE["y"].astype(object)))


def test_thresholds_rejects_nonfinite_ratio():
    """F4's numerics leg, closed at its root so BOTH thresholds() callers
    (anchor.fit_phe and validate.eval_phe) are covered by the one guard.

    Observed pre-fix behavior (unfixed tree): thresholds(np.array([2000.0,
    nan, 3000.0]), 1000) returned [0.0, nan, -0.693...] — NaN sailed through
    the any-based guard in the one function written to validate the ratio —
    and eval_phe fed a held-out frame with a single NaN n_cases returned
    mean_loglik=nan with no exception, silently defeating every downstream
    `<` comparison (nan < x is False)."""
    with pytest.raises(ValueError, match=r"strictly in \(0, 1\)"):
        thresholds(np.array([2000.0, np.nan, 3000.0]), 1000)

    fit = fit_phe(PHE)
    bad_test = PHE.copy()
    bad_test["n_cases"] = bad_test["n_cases"].astype(float)
    bad_test.iloc[0, bad_test.columns.get_loc("n_cases")] = np.nan
    with pytest.raises(ValueError, match=r"strictly in \(0, 1\)"):
        eval_phe(fit, bad_test, null_p=0.5)


def test_level_validation():
    """F3: level and n_boot are validated at entry, before any work, naming
    the parameter. The clean leg (level=0.95) is exercised by the whole
    existing suite.

    Observed pre-fix behavior (unfixed tree): bootstrap_dws(level=95) burned
    all 30 replicate refits (~1.1 s on the PC/PHE fixtures here) before
    dying inside mat.quantile with 'percentiles should all be in the
    interval [0, 1]' — never naming level; level=1.0 completed silently and
    published min/max extreme-order-statistic intervals (lo < hi everywhere)
    as if a 100% interval were legitimate; level=0.0 completed silently and
    published zero-width lo == hi intervals."""
    for bad in (95, 1.0, 0.0):
        with pytest.raises(ValueError, match="level must be a fraction strictly between"):
            bootstrap_dws(PC, PHE, n_boot=5, level=bad, rng=0)
    with pytest.raises(ValueError, match="n_boot must be a positive integer"):
        bootstrap_dws(PC, PHE, n_boot=0, rng=0)


def test_nan_y_misattribution_fixed():
    """F7: missing responses must fail as missing data, never be misdiagnosed
    as unanimity or die in integer coercion naming neither state nor cause.

    Observed pre-fix behavior (these exact triggers, unfixed tree): (a) with
    every s003 response set to NaN, estimate_dws warned \"dropping 1
    one-sided PHE state(s) from the anchor stage (logit weights
    unidentified): ['s003']\" — missing data diagnosed as unanimity — and
    published weights for all PC states under that wrong-cause warning;
    (b) with s003 nearly one-sided (all 1 except one 0) plus a single NaN,
    the one-sided guard passed (sum skips NaN, size counts it) and the run
    died in astype(int) with IntCastingNaNError 'Cannot convert non-finite
    values (NA or inf) to integer...' naming neither the state nor the
    missing data."""
    phe_all_nan = PHE.copy()
    phe_all_nan["y"] = phe_all_nan["y"].astype(float)
    phe_all_nan.loc[phe_all_nan["state"] == "s003", "y"] = np.nan
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError) as ei:
            estimate_dws(PC, phe_all_nan)
    assert "phe_df['y'] contains" in str(ei.value)
    assert "missing value(s) (NaN)" in str(ei.value)
    # The unanimity misdiagnosis is gone: no drop warning, no "every response
    # favors the same program" verdict anywhere.
    assert "favors the same program" not in str(ei.value)
    assert not [w for w in caught if "dropping" in str(w.message)]

    phe_near = PHE.copy()
    phe_near["y"] = phe_near["y"].astype(float)
    idx = phe_near.index[phe_near["state"] == "s003"]
    phe_near.loc[idx, "y"] = 1.0
    phe_near.loc[idx[0], "y"] = 0.0
    phe_near.loc[idx[1], "y"] = np.nan
    with pytest.raises(ValueError) as ei:
        estimate_dws(PC, phe_near)
    assert "missing value(s) (NaN)" in str(ei.value)


def test_all_one_sided_phe_raises_designed_error():
    """F8 path 3: the empirically observed pilot regime (respondents nearly
    always avert death, so every anchor state is unanimous) must fail as a
    designed data-regime verdict, not crash in pandas internals after a
    reassuring warning.

    Observed pre-fix behavior (this exact trigger, unfixed tree):
    estimate_dws first warned \"dropping 6 one-sided PHE state(s) from the
    anchor stage (logit weights unidentified): ['s000', ..., 's005']\" —
    reassuring, it names a routine drop — and then crashed inside pandas
    concat internals with ValueError 'cannot concatenate unaligned mixed
    dimensional NDFrame objects'."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError) as ei:
            estimate_dws(PC, PHE.assign(y=1))
    assert "no identifiable PHE states remain" in str(ei.value)
    assert "one-sided" in str(ei.value)  # the verdict names the true cause
    # The reassuring drop warning must NOT precede the raise.
    assert not [w for w in caught if "dropping" in str(w.message)]


def test_top_level_exports():
    """Rider P6: bootstrap_dws is importable from the package top level (this
    file's own imports prove it structurally); pin __all__ so the export
    cannot regress silently."""
    assert welfareweights.estimate_dws is estimate_dws
    assert welfareweights.bootstrap_dws is bootstrap_dws
    assert {"estimate_dws", "bootstrap_dws"} <= set(welfareweights.__all__)
