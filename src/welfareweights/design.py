"""Design-matrix construction and identification checks for the PC probit.

The non-obvious part of stage A (Salomon 2012 supplement): for k states,
the probit design matrix has k-1 indicator columns (one state omitted as
the reference, its coefficient fixed at 0). For a comparison row, the
first-listed state's column is +1, the second-listed state's is -1, all
others 0, and the outcome is y = 1 iff the first-listed state was judged
healthier. X @ beta then collapses to beta_1 - beta_2.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def infer_states(pc_df: pd.DataFrame) -> list[str]:
    """All states appearing in either position, sorted for determinism."""
    return sorted(set(pc_df["state_1"]) | set(pc_df["state_2"]))


def connected_components(pc_df: pd.DataFrame) -> list[set]:
    """Connected components of the comparison graph (union-find).

    The probit locates states only relative to states they are (transitively)
    compared with: a disconnected comparison graph leaves the relative scale
    between components unidentified, so estimation must refuse to run on one.
    """
    states = infer_states(pc_df)
    parent = {s: s for s in states}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(pc_df["state_1"], pc_df["state_2"]):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    comps: dict[str, set] = {}
    for s in states:
        comps.setdefault(find(s), set()).add(s)
    return list(comps.values())


def build_pc_design(
    pc_df: pd.DataFrame, states: list[str], ref_state: str
) -> tuple[pd.DataFrame, pd.Series]:
    """(X, y) for the stage-A probit under the +1/-1/0 coding.

    X columns are all states except ref_state (whose coefficient is the
    identifying beta = 0 restriction); y = 1 iff state_1 judged healthier.
    """
    if ref_state not in states:
        raise ValueError(f"ref_state {ref_state!r} not among states")
    same = pc_df["state_1"].to_numpy() == pc_df["state_2"].to_numpy()
    if same.any():
        raise ValueError(
            f"{int(same.sum())} comparison row(s) pair a state with itself; "
            "such rows carry no information and must be dropped before fitting"
        )
    col = {s: j for j, s in enumerate(states)}
    n = len(pc_df)
    X = np.zeros((n, len(states)))
    rows = np.arange(n)
    X[rows, pc_df["state_1"].map(col).to_numpy()] = 1.0
    X[rows, pc_df["state_2"].map(col).to_numpy()] = -1.0
    keep = [s for s in states if s != ref_state]
    Xdf = pd.DataFrame(X[:, [col[s] for s in keep]], columns=keep, index=pc_df.index)
    return Xdf, pc_df["y"].astype(int)
