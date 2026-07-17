"""Failure-mode evidence: every silent-garbage path added after peer review must
now raise or warn, and this file pins each one to a stable substring taken
directly from the source message.

Covered paths (function -> trigger -> match string):
  fit_phe        one-sided PHE state (constant y within a state)     "one-sided PHE state"
  estimate_dws   drops one-sided PHE states, warns, keeps PC weights "one-sided PHE state"
  fit_pc         separation: a state healthier in every comparison   "healthier in all or none of their comparisons"
  fit_anchor_map <3 shared states                                    "at least three states"
  fit_anchor_map <5 shared states (warns, does not raise)            "shared states"
  fit_anchor_map non-negative fitted slope                           "slope is non-negative"
  fit_phe        deaths argument contradicts per-row deaths column   "deaths argument contradicts the per-row deaths column"
  build_pc_design (via fit_pc) a row pairs a state with itself       "pair a state with itself"
  estimate_dws   anchor-map R^2 below min_anchor_r2                  "fits poorly"

Tests 1/3/6/7 use tiny hand-built DataFrames (no simulate calls, deterministic,
instantaneous). Test 4 duck-types PCFit/PHEFit with types.SimpleNamespace,
since fit_anchor_map only reads pc_fit.{states,beta} and
phe_fit.{states,logit_dw,logit_dw_se} and Python does not enforce dataclass
types — this isolates the anchor-map checks from stage A/B entirely. Test 5
simulates a real pipeline with the paired-comparison slope sign flipped, so
stage A's healthiness scale increases (rather than decreases) with the
stage-B logit weight. Tests 2 and 8 share one module-scoped clean simulated
fixture pair, matching test_recovery's fixture-reuse style.
"""

import types
import warnings

import numpy as np
import pandas as pd
import pytest

from welfareweights.pipeline import estimate_dws
from welfareweights.probit import fit_pc
from welfareweights.anchor import fit_phe
from welfareweights.rescale import fit_anchor_map
from welfareweights.simulate import make_states, simulate_pc, simulate_phe, simulate_true_dws


def _mid_range_anchor_states(states, true_dws, n=6, lo=0.05, hi=0.90):
    """States nearest true dw = 0.5, excluding the extremes (dw outside
    [lo, hi]) where the PHE censoring thresholds rarely bracket the latent
    distribution and a state can turn one-sided by chance alone — same
    exclusion test_recovery.py applies (there via a fixed 0.05-0.70 band)."""
    mid = [s for s, d in zip(states, true_dws) if lo <= d <= hi]
    return sorted(mid, key=lambda s: abs(true_dws[states.index(s)] - 0.5))[:n]

# ---------------------------------------------------------------------------
# Tests 1, 3, 6, 7: tiny hand-built DataFrames, no simulation.
# ---------------------------------------------------------------------------


def test_fit_phe_raises_on_one_sided_state():
    """State 'a' answers the same way on every row (y all 1): its logit weight
    diverges (quasi-separation), and fit_phe must refuse before ever touching
    thresholds() or the probit fit (see anchor.py module docstring)."""
    df = pd.DataFrame(
        {
            "state": ["a", "a", "a", "a", "b", "b", "b", "b"],
            "y": [1, 1, 1, 1, 1, 0, 1, 0],
            "n_cases": [2000] * 8,
        }
    )
    with pytest.raises(ValueError, match="one-sided PHE state"):
        fit_phe(df)


def test_fit_pc_raises_on_separation():
    """A comparison graph where at least one state is judged healthier (or
    less healthy) in every one of its appearances is separated, like an
    unbeaten team in Bradley-Terry: its coefficient diverges. Rows form a
    single connected component {a, b, c}, so the disconnected-graph check
    does not preempt the separation check."""
    df = pd.DataFrame(
        {
            "state_1": ["a", "a", "b", "c"],
            "state_2": ["b", "c", "c", "b"],
            "y": [1, 1, 1, 1],
        }
    )
    with pytest.raises(ValueError, match="healthier in all or none of their comparisons"):
        fit_pc(df)


def test_fit_phe_raises_on_deaths_contradiction():
    """When phe_df carries a per-row deaths column, a contradictory `deaths`
    argument must raise before thresholds() or the one-sided check even run
    — the threshold has to use the deaths figure respondents actually saw."""
    df = pd.DataFrame(
        {
            "state": ["a", "a", "b", "b"],
            "y": [1, 0, 1, 0],
            "n_cases": [2000] * 4,
            "deaths": [1000] * 4,
        }
    )
    with pytest.raises(ValueError, match="deaths argument contradicts the per-row deaths column"):
        fit_phe(df, deaths=500)


def test_build_pc_design_raises_on_self_pairing():
    """Row 0 pairs state 'a' with itself. Wins/appearances are hand-tallied so
    the separation pre-check in fit_pc passes (a=3/6, b=2/4, c=2/4 — none is
    0 or 100%) and execution reaches build_pc_design, whose same-state check
    then fires. Graph is one connected component."""
    df = pd.DataFrame(
        {
            "state_1": ["a", "a", "b", "a", "c", "b", "c"],
            "state_2": ["a", "b", "a", "c", "a", "c", "b"],
            "y": [1, 1, 1, 1, 1, 1, 1],
        }
    )
    with pytest.raises(ValueError, match="pair a state with itself"):
        fit_pc(df)


# ---------------------------------------------------------------------------
# Test 4: fit_anchor_map's shared-state-count checks, via duck-typed fakes.
# ---------------------------------------------------------------------------


def _fake_fits(logit_dw_by_state: dict, beta_by_state: dict):
    """Minimal stand-ins for PCFit/PHEFit carrying only the attributes
    fit_anchor_map reads: pc_fit.states, pc_fit.beta, phe_fit.states,
    phe_fit.logit_dw, phe_fit.logit_dw_se. Dataclass types are not enforced
    by Python, so these duck-typed objects exercise fit_anchor_map in
    isolation from fit_pc/fit_phe."""
    states = list(logit_dw_by_state)
    pc_fit = types.SimpleNamespace(states=states, beta=pd.Series(beta_by_state))
    phe_fit = types.SimpleNamespace(
        states=states,
        logit_dw=pd.Series(logit_dw_by_state),
        logit_dw_se=pd.Series({s: 1.0 for s in states}),
    )
    return pc_fit, phe_fit


def test_fit_anchor_map_raises_below_three_shared_states():
    """Only 2 shared states: the anchor line has 2 free parameters (slope,
    intercept) and 2 points determine it exactly, leaving no way to assess
    fit — fit_anchor_map must refuse."""
    pc_fit, phe_fit = _fake_fits(
        logit_dw_by_state={"s0": -1.0, "s1": 1.0},
        beta_by_state={"s0": 1.0, "s1": -1.0},
    )
    with pytest.raises(ValueError, match="at least three states"):
        fit_anchor_map(pc_fit, phe_fit)


def test_fit_anchor_map_warns_below_five_shared_states():
    """4 shared states with beta exactly affine-decreasing in logit_dw (a
    perfect fit, so the slope is unambiguously negative and the separate
    non-negative-slope check does not also fire): fit_anchor_map must warn
    that R^2 is a weak diagnostic at this size, but still return a fit."""
    pc_fit, phe_fit = _fake_fits(
        logit_dw_by_state={"s0": -2.0, "s1": -1.0, "s2": 0.0, "s3": 1.0},
        beta_by_state={"s0": 2.0, "s1": 1.0, "s2": 0.0, "s3": -1.0},
    )
    with pytest.warns(UserWarning, match="shared states"):
        amap = fit_anchor_map(pc_fit, phe_fit)
    assert amap.slope < 0  # confirms the warn path, not the raise path, ran


# ---------------------------------------------------------------------------
# Test 5: fit_anchor_map's non-negative-slope check, via a real (but
# orientation-reversed) simulated pipeline.
# ---------------------------------------------------------------------------


def test_fit_anchor_map_raises_on_reversed_orientation():
    """simulate_pc with slope=+1.0 flips the assumed sign convention: stage-A
    healthiness now INCREASES with logit(dw) instead of decreasing, while
    simulate_phe always encodes L = logit(dw) + eps regardless of pc's slope
    knob. Regressing stage-A beta on stage-B logit_dw over states that
    co-increase with true dw produces a positive slope, which fit_anchor_map
    must refuse."""
    k = 8
    states = make_states(k)
    true_dws = simulate_true_dws(k)
    anchor_states = _mid_range_anchor_states(states, true_dws)

    pc_df = simulate_pc(true_dws, states, n_respondents=400, slope=1.0, intercept=0.3, rng=201)
    phe_df = simulate_phe(
        true_dws, states, n_respondents=400, sigma=0.8, anchor_states=anchor_states, rng=202
    )
    pc_fit = fit_pc(pc_df)
    phe_fit = fit_phe(phe_df)
    with pytest.raises(ValueError, match="slope is non-negative"):
        fit_anchor_map(pc_fit, phe_fit)


# ---------------------------------------------------------------------------
# Tests 2 & 8: shared clean simulated fixtures (estimate_dws end-to-end).
# ---------------------------------------------------------------------------

K = 10
STATES = make_states(K)
TRUE_DWS = simulate_true_dws(K)
# 6 states nearest true dw = 0.5, extremes excluded: PHE thresholds
# logit(deaths/c) bracket the latent distribution best there, mirroring
# test_recovery's mid-range anchor subset.
ANCHOR_STATES = _mid_range_anchor_states(STATES, TRUE_DWS)


@pytest.fixture(scope="module")
def clean_pc_df():
    return simulate_pc(TRUE_DWS, STATES, n_respondents=800, slope=-1.0, intercept=0.3, rng=101)


@pytest.fixture(scope="module")
def clean_phe_df():
    return simulate_phe(
        TRUE_DWS, STATES, n_respondents=600, sigma=0.8, anchor_states=ANCHOR_STATES, rng=102
    )


def test_estimate_dws_drops_one_sided_state_with_warning(clean_pc_df, clean_phe_df):
    """Forcing every response for one anchor state to y=1 makes it one-sided.
    estimate_dws must warn and drop it from the anchor stage, but still
    return dw for every PC state (the drop only affects which states inform
    the anchor line, not stage C's application of that line to every
    pc_fit.beta)."""
    target = ANCHOR_STATES[0]  # anchor state closest to true dw = 0.5
    modified_phe_df = clean_phe_df.copy()  # copy: must not mutate the shared fixture
    modified_phe_df.loc[modified_phe_df["state"] == target, "y"] = 1

    with pytest.warns(UserWarning, match="one-sided PHE state"):
        weights, diag = estimate_dws(clean_pc_df, modified_phe_df)

    assert diag["one_sided_dropped"] == [target]
    assert set(weights.index) == set(STATES)  # all 10 PC states still present
    assert weights["dw"].notna().all()


def test_estimate_dws_warns_on_phe_only_state(clean_pc_df, clean_phe_df):
    """F6: a state present in the PHE responses but absent from the paired
    comparisons is mathematically-correctly excluded from the weights (no
    stage-A coefficient exists to map), but the exclusion must be LOUD — a
    state silently disappearing from published output is the one anomaly an
    unattended publisher's warning channel would otherwise never surface.
    The audit verified the pre-fix behavior: a state with ample two-sided
    PHE responses and zero PC rows was absent from the weights index with no
    warning (unlike every other data-quality anomaly in the pipeline), only
    the silent diagnostics['phe_only_states'] entry recording it."""
    extra = pd.DataFrame(
        {
            "respondent_id": [5000 + i for i in range(40)],
            "state": ["sphe"] * 40,
            "n_cases": [2000] * 40,
            "deaths": [1000] * 40,
            "y": [1, 0] * 20,  # two-sided: the one-sided drop path must not fire
        }
    )
    phe_plus = pd.concat([clean_phe_df, extra], ignore_index=True)

    with pytest.warns(UserWarning, match="no paired-comparison data") as rec:
        weights, diag = estimate_dws(clean_pc_df, phe_plus)

    assert "sphe" not in weights.index
    assert diag["phe_only_states"] == ["sphe"]  # the diagnostics entry stays
    fired = [str(m.message) for m in rec if "no paired-comparison data" in str(m.message)]
    assert fired and "sphe" in fired[0]


def test_no_phe_only_warning_on_clean_data(clean_pc_df, clean_phe_df):
    """F6 silence leg: when every PHE state also has paired-comparison data,
    the phe-only warning must not fire."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimate_dws(clean_pc_df, clean_phe_df)
    assert not [w for w in caught if "no paired-comparison data" in str(w.message)]


def test_estimate_dws_warns_below_min_anchor_r2(clean_pc_df, clean_phe_df):
    """Self-calibrating trigger: read off the R^2 actually achieved on clean
    data (min_anchor_r2=0.0 is passed so this first call cannot itself warn),
    then require a warning when min_anchor_r2 is set just above that
    achieved value — the easiest honest way to force the R^2 floor to bind."""
    _, diag0 = estimate_dws(clean_pc_df, clean_phe_df, min_anchor_r2=0.0)
    achieved = diag0["anchor_map"].r_squared

    with pytest.warns(UserWarning, match="fits poorly"):
        estimate_dws(clean_pc_df, clean_phe_df, min_anchor_r2=achieved + 1e-3)
