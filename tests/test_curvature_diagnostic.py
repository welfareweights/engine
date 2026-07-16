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

import warnings

import numpy as np
import pytest

from welfareweights.pipeline import estimate_dws
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
