"""The extrapolation diagnostic (audit finding F5): weights for states mapped
outside the PHE anchor range rest on the fitted line where no anchor data
exists, and the curvature gate — which measures misfit only INSIDE the anchor
range — is structurally blind to the harm. The always-published
`extrapolation` column (distance past the nearest end of the anchor range, in
anchor-span units) is the primary deliverable; the warning past
max_extrapolation (default 0.5 spans) is the alert-channel leg. Threshold
justification lives in the estimate_dws docstring.

The narrow mid-band-anchor configuration here is the audit's F5 reproduction
shape AND the designed production configuration: PHE covers fewer states than
PC by design, so extreme states are routinely extrapolated.
"""

import warnings

import numpy as np
import pytest

from welfareweights.inference import bootstrap_dws
from welfareweights.pipeline import estimate_dws
from welfareweights.simulate import make_states, simulate_pc, simulate_phe, simulate_true_dws

K = 20
STATES = make_states(K)
TRUE_DWS = simulate_true_dws(K)
# Narrow mid band: 5 anchors with true dw 0.199-0.585, PC spanning 0.005-0.95.
NARROW_ANCHORS = [s for s, dw in zip(STATES, TRUE_DWS) if 0.15 <= dw <= 0.65]


def _simulate(curvature, seed):
    pc = simulate_pc(
        TRUE_DWS, STATES, 1500, slope=-1.0, intercept=0.3, curvature=curvature, rng=seed
    )
    phe = simulate_phe(
        TRUE_DWS, STATES, 2500, sigma=0.8, anchor_states=NARROW_ANCHORS, rng=seed + 400
    )
    return pc, phe


def test_extrapolation_column_measures_distance():
    """The column is the hand-computable distance past the nearest end of
    diagnostics['anchor_map'].anchor_range, as a fraction of the anchor span:
    exactly 0.0 for states mapped inside the range, strictly positive for the
    extreme states the narrow anchor band cannot reach."""
    pc, phe = _simulate(curvature=0.0, seed=500)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w, diag = estimate_dws(pc, phe)

    lo, hi = diag["anchor_map"].anchor_range
    span = hi - lo
    assert span > 0
    x = w["logit_dw"].to_numpy()
    expected = np.maximum(np.maximum(lo - x, x - hi), 0.0) / span
    np.testing.assert_allclose(w["extrapolation"].to_numpy(), expected, rtol=1e-12, atol=0)
    inside = (x >= lo) & (x <= hi)
    assert inside.any()
    assert (w["extrapolation"].to_numpy()[inside] == 0.0).all()
    for s in ("s000", "s019"):  # extreme states, far outside the mid band
        assert w.at[s, "extrapolation"] > 0.5


def test_extrapolation_warning_fires_where_curvature_gate_is_blind():
    """The audit's F5 reproduction: mild curvature + narrow mid-band anchors +
    full-span PC. The curvature gate reads the anchor scatter as clean while
    extrapolated states' actual weight errors exceed the pipeline's own
    damage threshold (max_curvature_impact=0.025); the extrapolation warning
    is the diagnostic that fires, naming the extreme states.

    Tuned constants: curvature=0.05, pc rng=502, phe rng=902. Measured on
    this configuration against the unfixed tree: the run emitted ZERO
    warnings — reported curvature_impact 0.0012 (curvature_pvalue 0.928,
    R^2-clean) while the actual weight error among states mapped more than
    0.5 anchor-spans outside the range reached 0.0476, nearly twice the
    pipeline's declared 0.025 tolerance (the audit measured the same shape:
    0.0305 actual vs 0.0045 reported). Post-fix the same run publishes the
    extrapolation column and warns on the extreme states."""
    pc, phe = _simulate(curvature=0.05, seed=502)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        w, diag = estimate_dws(pc, phe)

    msgs = [str(c.message) for c in caught]
    assert not [m for m in msgs if "curvature" in m]  # the in-range gate is quiet
    fired = [m for m in msgs if "outside the PHE anchor range" in m]
    assert len(fired) == 1
    assert "s000" in fired[0] and "s019" in fired[0]

    err = np.abs(w["dw"].to_numpy() - TRUE_DWS)
    far = w["extrapolation"].to_numpy() > 0.5
    assert far.any()
    assert diag["anchor_map"].curvature_impact < 0.025  # gate cannot see the harm...
    assert err[far].max() > 0.025  # ...which exceeds the pipeline's own tolerance


def test_extrapolation_silent_on_full_coverage():
    """Anchors covering every state: the column is present (schema stability
    for the nightly's diffs) with every value far below the threshold, and no
    extrapolation warning fires — no alert fatigue on full-coverage designs."""
    k8 = make_states(8)
    true8 = simulate_true_dws(8, low=0.05, high=0.7)
    pc = simulate_pc(true8, k8, n_respondents=400, rng=31)
    phe = simulate_phe(true8, k8, n_respondents=600, rng=32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        w, _ = estimate_dws(pc, phe)

    assert not [c for c in caught if "outside the PHE anchor range" in str(c.message)]
    assert "extrapolation" in w.columns
    assert (w["extrapolation"] < 0.5).all()


def test_extrapolation_column_carried_by_bootstrap():
    """bootstrap_dws publishes the point run's extrapolation column unchanged
    (the table the nightly actually consumes), and the warning is emitted
    exactly once — from the full-sample call, not once per replicate
    (replicate-loop warnings are suppressed)."""
    k8 = make_states(8)
    true8 = simulate_true_dws(8)
    anchors = [s for s, dw in zip(k8, true8) if 0.04 <= dw <= 0.70]
    pc = simulate_pc(true8, k8, n_respondents=250, pairs_per_respondent=6, rng=41)
    phe = simulate_phe(true8, k8, n_respondents=350, sigma=0.8, anchor_states=anchors, rng=42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        point, _ = estimate_dws(pc, phe)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = bootstrap_dws(pc, phe, n_boot=20, rng=5)

    assert "extrapolation" in res.weights.columns
    np.testing.assert_array_equal(
        res.weights["extrapolation"].to_numpy(), point["extrapolation"].to_numpy()
    )
    fired = [c for c in caught if "outside the PHE anchor range" in str(c.message)]
    assert len(fired) == 1
