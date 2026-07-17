"""The curvature gate: the one misspecification that damages weight levels
while evading R^2 (studies/RESULTS-misspecification.md) must now warn.

The gate requires BOTH materiality (impact > 0.02 in weight units) and
statistical significance (quadratic-term p < 0.05). Calibration behind that
rule, measured across seeded reps: on clean data the impact metric's noise
floor widens at small samples (up to 0.034 at 300/400 respondents) but its
p-values stay large (0.12-0.86 across every clean design tested); under
curvature=0.15 (true max level error ~0.16) p <= 0.005 and impact >= 0.033
in every rep. Mild curvature (0.05, true error ~0.04) is below the
diagnostic's power at these anchor counts; that detection limit is
documented in METHODS.md and shrinks with wider anchor sets.
"""

import types
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from welfareweights.pipeline import estimate_dws
from welfareweights.rescale import fit_anchor_map
from welfareweights.simulate import make_states, simulate_pc, simulate_phe, simulate_true_dws

K = 20
STATES = make_states(K)
TRUE_DWS = simulate_true_dws(K)
ANCHOR_STATES = [s for s, dw in zip(STATES, TRUE_DWS) if 0.05 <= dw <= 0.70]


def _simulate(curvature, seed=300):
    pc = simulate_pc(
        TRUE_DWS, STATES, 1500, slope=-1.0, intercept=0.3, curvature=curvature, rng=seed
    )
    phe = simulate_phe(TRUE_DWS, STATES, 2500, sigma=0.8, anchor_states=ANCHOR_STATES, rng=seed + 400)
    return pc, phe


def test_clean_data_does_not_fire_the_curvature_warning():
    pc, phe = _simulate(curvature=0.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, diag = estimate_dws(pc, phe)
    assert not [w for w in caught if "curvature" in str(w.message)]
    assert diag["anchor_map"].curvature_pvalue > 0.05


def test_noisy_small_sample_impact_alone_does_not_fire():
    """At 300/400 respondents the impact metric's noise floor exceeds the
    materiality threshold on CLEAN data (seed 101 gives ~0.034); the p-value
    condition is what keeps this from being a false alarm."""
    k = 10
    states = make_states(k)
    true = simulate_true_dws(k)
    anchors = [s for s, dw in zip(states, true) if 0.02 <= dw <= 0.90]
    pc = simulate_pc(true, states, 300, slope=-1.0, intercept=0.3, rng=101)
    phe = simulate_phe(true, states, 400, sigma=0.8, anchor_states=anchors, rng=102)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, diag = estimate_dws(pc, phe)
    assert diag["anchor_map"].curvature_impact > 0.02  # material on its face...
    assert diag["anchor_map"].curvature_pvalue > 0.05  # ...but not distinguishable from noise
    assert not [w for w in caught if "curvature" in str(w.message)]


def test_severe_curvature_fires_the_warning_r2_misses():
    """curvature=0.15 produces ~0.16 level errors while anchor R^2 stays ~0.985,
    above the min_anchor_r2=0.9 gate; only this diagnostic catches it."""
    pc, phe = _simulate(curvature=0.15)
    with pytest.warns(UserWarning, match="curvature"):
        _, diag = estimate_dws(pc, phe)
    assert diag["anchor_map"].curvature_impact > 0.02
    assert diag["anchor_map"].curvature_pvalue < 0.05
    assert diag["anchor_map"].r_squared > 0.9  # the R^2 gate indeed stays silent


def test_curvature_impact_equals_brute_force_inversion():
    """F2 (plan-09 rubric 3): curvature_impact must equal the EXACT line-
    inversion error, verified against brute force, not just asserted.

    Stage C inverts the fitted line, x = (y - intercept)/slope, so when the
    true link is the quadratic through the same anchors the x-error of that
    inversion at anchor i is (quad_i - lin_i)/slope — SIGNED. slope < 0 is
    enforced upstream, so dividing by abs(slope) flips every displacement,
    and because expit is asymmetric away from 0 the max is then taken in the
    wrong direction. A symmetric anchor set hides the defect exactly
    (expit(-a) = 1 - expit(a) makes the max sign-invariant), so this test
    uses an asymmetric one, where the two formulas visibly disagree.

    Duck-typed fits (the test_failure_modes _fake_fits idiom) carry a
    NOISELESS quadratic link over 5 anchors, so the quadratic refit inside
    fit_anchor_map is exact and the equality below is analytic, not
    approximate.

    Observed pre-fix behavior (these exact constants, unfixed tree):
    reported curvature_impact 0.037466204869012976 vs brute-force
    line-inversion error 0.04108409750239256 — 8.8% under-sized — while the
    abs-slope formula recomputed inline matched the reported value exactly."""
    x = np.array([-3.0, -2.0, -1.0, 0.0, 1.0])
    y = -1.2 * x + 0.4 + 0.15 * x * x  # noiseless quadratic link, slope < 0
    states = [f"s{i}" for i in range(len(x))]
    pc_fit = types.SimpleNamespace(states=states, beta=pd.Series(y, index=states))
    phe_fit = types.SimpleNamespace(
        states=states,
        logit_dw=pd.Series(x, index=states),
        logit_dw_se=pd.Series(1.0, index=states),
    )
    amap = fit_anchor_map(pc_fit, phe_fit)
    assert amap.slope < 0

    # Brute force: invert the fitted line at the quadratic's y values and
    # measure the weight-space error of that inversion.
    x_inv = (y - amap.intercept) / amap.slope
    brute = float(np.max(np.abs(expit(x_inv) - expit(x))))
    assert np.isclose(amap.curvature_impact, brute, rtol=0, atol=1e-12)

    # The pre-fix abs-slope formula, recomputed inline, is measurably NOT the
    # line-inversion error on an asymmetric anchor set.
    lin = amap.slope * x + amap.intercept
    quad = y  # noiseless: the quadratic refit reproduces y exactly
    pre_fix = float(np.max(np.abs(expit(x + (quad - lin) / abs(amap.slope)) - expit(x))))
    assert abs(pre_fix - brute) > 1e-3


def test_signed_slope_flips_the_gate():
    """F2 consequence leg (plan 09's required verification shape): a curved
    simulation where the corrected metric sits above the gate threshold and
    the pre-fix metric below it — the configuration on which the pre-fix
    gate stayed silent must now fire.

    Tuned constants: curvature=0.10, pc rng=315, phe rng=715 (the module's
    _simulate shape), threshold t=0.0477. Observed pre-fix behavior (unfixed
    tree): estimate_dws(..., max_curvature_impact=0.0477) completed with
    ZERO warnings — reported curvature_impact 0.04542681949718741 (below t)
    with curvature_pvalue 0.00307343333527047 — while the exact signed
    metric was 0.04993 (above t). This echoes the audit's 40-run sweep,
    where 2 runs flipped the gate, one at true anchor-state weight error
    0.0284 with the warning suppressed."""
    t = 0.0477
    pc, phe = _simulate(curvature=0.10, seed=315)
    with pytest.warns(UserWarning, match="curvature"):
        _, diag = estimate_dws(pc, phe, max_curvature_impact=t)
    amap = diag["anchor_map"]
    assert amap.curvature_impact > t
    assert amap.curvature_pvalue < 0.05

    # Recompute the PRE-fix metric exactly as the old code did, from the same
    # fitted stages: it sits below t, so the old gate stayed silent here.
    shared = [s for s in diag["pc_fit"].states if s in set(diag["phe_fit"].states)]
    x = diag["phe_fit"].logit_dw[shared].to_numpy()
    yv = diag["pc_fit"].beta[shared].to_numpy()
    slope, intercept = np.polyfit(x, yv, 1)
    lin = slope * x + intercept
    Xq = np.column_stack([np.ones_like(x), x, x * x])
    coef, *_ = np.linalg.lstsq(Xq, yv, rcond=None)
    quad = Xq @ coef
    pre_fix = float(np.max(np.abs(expit(x + (quad - lin) / abs(slope)) - expit(x))))
    assert pre_fix < t


def test_fewer_than_five_anchors_yields_nan_and_no_warning():
    """A quadratic through 3-4 points is (near-)exact interpolation, so the
    diagnostic would measure pure noise; it must report NaN and stay quiet."""
    small_anchor = ANCHOR_STATES[:4]
    pc = simulate_pc(TRUE_DWS, STATES, 1500, slope=-1.0, intercept=0.3, rng=300)
    phe = simulate_phe(TRUE_DWS, STATES, 2500, sigma=0.8, anchor_states=small_anchor, rng=700)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _, diag = estimate_dws(pc, phe)
    assert np.isnan(diag["anchor_map"].curvature_impact)
    assert np.isnan(diag["anchor_map"].curvature_pvalue)
    assert not [w for w in caught if "curvature" in str(w.message)]
