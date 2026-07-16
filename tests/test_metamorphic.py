"""Metamorphic invariance tests: apply a transform to the input tables that a
correct pipeline must be indifferent to, and require the transformed rerun to
reproduce the baseline exactly (or to a tolerance set only by floating-point
arithmetic, never by statistical noise). Each test pins one structural
property of the estimator that the recovery tests (test_recovery.py) cannot
exercise, because recovery only checks that fitted weights are numerically
close to the true DGP weights, not that the estimator treats provably
equivalent inputs identically.
"""

import numpy as np
import pandas as pd

from welfareweights.pipeline import estimate_dws
from welfareweights.simulate import make_states, simulate_pc, simulate_phe, simulate_true_dws

K = 10
STATES = make_states(K)
TRUE_DWS = simulate_true_dws(K)
# Anchor subset: mid-range weights only, where the PHE thresholds actually
# bracket the latent distribution (extreme states give one-sided quasi-
# separated responses — see test_recovery.py). At K=10 the 0.05/0.70 bounds
# used there leave fewer than 5 shared anchor states and trip the "fitted on
# <5 states" warning, so this widens to 0.02/0.90 to keep 7 of the 10 states
# (s002..s008) anchored.
ANCHOR_STATES = [s for s, dw in zip(STATES, TRUE_DWS) if 0.02 <= dw <= 0.90]

pc_df = simulate_pc(TRUE_DWS, STATES, n_respondents=300, slope=-1.0, intercept=0.3, rng=101)
phe_df = simulate_phe(TRUE_DWS, STATES, n_respondents=400, sigma=0.8, anchor_states=ANCHOR_STATES, rng=102)

_baseline_weights, _baseline_diag = estimate_dws(pc_df, phe_df)


def test_label_permutation_invariance():
    """Renaming states must permute the final dw identically: the pipeline's
    default reference state is `sorted(states)[0]` (probit.fit_pc), so a
    relabeling that changes sort order changes WHICH state is dropped as the
    reference — this test exercises that the proven reference-state
    invariance (test_recovery.test_reference_state_invariance) actually
    survives an end-to-end relabeling, not just an explicit ref_state swap.
    """
    rng = np.random.default_rng(1)
    shuffled = rng.permutation(STATES)
    relabel = dict(zip(STATES, shuffled))

    pc_relabeled = pc_df.assign(
        state_1=pc_df["state_1"].map(relabel), state_2=pc_df["state_2"].map(relabel)
    )
    phe_relabeled = phe_df.assign(state=phe_df["state"].map(relabel))

    weights, _ = estimate_dws(pc_relabeled, phe_relabeled)
    baseline = _baseline_weights["dw"].reindex(STATES).to_numpy()
    relabeled = weights["dw"].reindex([relabel[s] for s in STATES]).to_numpy()
    np.testing.assert_allclose(relabeled, baseline, atol=1e-5)


def test_swap_and_flip_invariance():
    """Swapping state_1/state_2 and flipping y=1-y on a subset of PC rows must
    leave the fitted weights unchanged: under the +1/-1/0 design coding
    (design.py), a row with (state_1=A, state_2=B, y) contributes
    Phi(beta_A - beta_B) to the likelihood if y=1 and its complement if y=0;
    swapping to (state_1=B, state_2=A, y'=1-y) contributes
    Phi(beta_B - beta_A) if y'=1 (i.e. y=0, unchanged contribution) or its
    complement if y'=0 (i.e. y=1, again unchanged) — so this is an exact MLE
    identity, not a statistical coincidence, and should hold near machine
    precision. phe_df and the state universe/sort order are untouched, so the
    default reference state matches the baseline automatically, isolating
    this transform from test 1's relabeling effect.
    """
    rng = np.random.default_rng(2)
    mask = rng.random(len(pc_df)) < 0.5
    pc_swapped = pc_df.copy()
    pc_swapped.loc[mask, ["state_1", "state_2"]] = pc_df.loc[mask, ["state_2", "state_1"]].to_numpy()
    pc_swapped.loc[mask, "y"] = 1 - pc_swapped.loc[mask, "y"]

    weights, _ = estimate_dws(pc_swapped, phe_df)
    baseline = _baseline_weights["dw"].reindex(STATES).to_numpy()
    transformed = weights["dw"].reindex(STATES).to_numpy()
    np.testing.assert_allclose(transformed, baseline, atol=1e-6)


def test_threshold_ratio_invariance():
    """Scaling the PHE deaths column and n_cases column by a common factor
    must leave the fitted weights unchanged: the censoring threshold is
    t = logit(deaths / n_cases) (anchor.thresholds), and deaths/n_cases is
    invariant to any common scale factor applied to both — only the ratio
    ever enters the likelihood. deaths=None (the default) makes fit_phe read
    the scaled per-row deaths column directly rather than a fixed argument.
    """
    phe_scaled = phe_df.assign(deaths=phe_df["deaths"] * 10, n_cases=phe_df["n_cases"] * 10)

    weights, _ = estimate_dws(pc_df, phe_scaled)
    baseline = _baseline_weights["dw"].reindex(STATES).to_numpy()
    transformed = weights["dw"].reindex(STATES).to_numpy()
    # Near machine precision: the only difference from baseline is floating-
    # point reordering in the *10/*10 arithmetic feeding logit(ratio).
    np.testing.assert_allclose(transformed, baseline, atol=1e-9)


def test_duplication_invariance():
    """Concatenating each input table with itself must leave the point
    estimate unchanged: duplicating every row doubles every observation's
    log-likelihood contribution identically, so the MLE (the value that
    maximizes the sum) is unchanged even though standard errors computed from
    the doubled sample would shrink. Point estimation here never touches
    respondent_id-based clustering, so exact respondent_id collisions after
    concatenation are harmless for this test — only inference.bootstrap_dws
    (not exercised here) would need genuinely distinct respondents.
    """
    pc_dup = pd.concat([pc_df, pc_df], ignore_index=True)
    phe_dup = pd.concat([phe_df, phe_df], ignore_index=True)

    weights, _ = estimate_dws(pc_dup, phe_dup)
    baseline = _baseline_weights["dw"].reindex(STATES).to_numpy()
    transformed = weights["dw"].reindex(STATES).to_numpy()
    np.testing.assert_allclose(transformed, baseline, atol=1e-6)
