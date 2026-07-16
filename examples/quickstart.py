"""End-to-end demonstration on synthetic data with known true weights.

Simulates a small survey from the model the estimator assumes (12 states,
600 paired-comparison respondents, 900 anchoring respondents), runs the
three-stage pipeline plus the respondent-cluster bootstrap, and prints the
estimates against the truth they should recover. Companion reading:
docs/METHODS.md explains each stage; this shows the machine actually work.

Run:  .venv/bin/python examples/quickstart.py   (~15 s, dominated by bootstrap)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from welfareweights.inference import bootstrap_dws
from welfareweights.simulate import make_states, simulate_pc, simulate_phe, simulate_true_dws

K = 12
STATES = make_states(K)
TRUE_DWS = simulate_true_dws(K, low=0.01, high=0.90)
# PHE questions only make sense for states the thresholds logit(1000/c) can
# bracket — the instrument anchors mid-severity states (see METHODS.md sec. 4).
ANCHOR_STATES = [s for s, dw in zip(STATES, TRUE_DWS) if 0.05 <= dw <= 0.70]

print(f"Simulating: {K} states, 600 PC respondents x 15 pairs, "
      f"900 PHE respondents x 3 questions over {len(ANCHOR_STATES)} anchor states")
pc_df = simulate_pc(TRUE_DWS, STATES, n_respondents=600, rng=1)
phe_df = simulate_phe(TRUE_DWS, STATES, n_respondents=900, anchor_states=ANCHOR_STATES, rng=2)

print("Estimating (stages A-C) + respondent-cluster bootstrap (n_boot=100)...")
res = bootstrap_dws(pc_df, phe_df, n_boot=100, rng=3)

amap = res.diagnostics["anchor_map"]
print(f"\nanchor map: slope {amap.slope:.3f} (DGP truth -1.0), "
      f"R^2 {amap.r_squared:.3f}, {amap.n_shared} shared states")
print(f"bootstrap: {res.n_boot - res.n_failed}/{res.n_boot} replicates estimable\n")

w = res.weights.reindex(STATES)
table = pd.DataFrame(
    {
        "true_dw": TRUE_DWS.round(3),
        "estimate": w["dw"].round(3),
        "lo95": w["lo"].round(3),
        "hi95": w["hi"].round(3),
        "covered": (TRUE_DWS >= w["lo"]) & (TRUE_DWS <= w["hi"]),
        "anchored": [s in ANCHOR_STATES for s in STATES],
    },
    index=pd.Index(STATES, name="state"),
)
print(table.to_string())

err = np.abs(w["dw"].to_numpy() - TRUE_DWS)
rank = pd.Series(w["dw"].to_numpy()).corr(pd.Series(TRUE_DWS), method="spearman")
print(f"\nrank correlation {rank:.4f} | mean |error| {err.mean():.4f} | max |error| {err.max():.4f}")
print("Every number above is reproducible: seeds are fixed in this script.")
