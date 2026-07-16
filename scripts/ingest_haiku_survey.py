"""Ingest the Haiku synthetic-survey workflow output, estimate weights, compare
to the published GHE2021 targets.

The workflow (scratchpad/haiku_survey_embedded.mjs) returns, per respondent,
the ordered A/B and 1/2 answers to that respondent's assigned questions. This
script re-derives the identical assignment (same states + seeds), attaches the
answers, runs the estimator, and reports the correlation and mean error
against the curated published weights.

Usage:
    ingest_haiku_survey.py <workflow_journal.jsonl>

The journal path is <transcript_dir>/journal.jsonl for the run; the final
workflow return value carries {respondents: [{id, pc:[...], phe:[...]}]}.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ENGINE = Path(__file__).resolve().parents[1]
DATA = ENGINE.parent / "data"  # clinical datasets live outside the engine (see data/README.md)
sys.path.insert(0, str(ENGINE / "src"))

from welfareweights.pipeline import estimate_dws  # noqa: E402
from welfareweights.survey import (  # noqa: E402
    assign_pairs,
    assign_phe,
    parse_pc_responses,
    parse_phe_responses,
)

# Must match the assignment used to build the survey (scratchpad generation).
PAIRS_PER_RESPONDENT = 12
PHE_PER_RESPONDENT = 4
PC_SEED, PHE_SEED = 101, 202


def load_curated() -> tuple[list[str], dict[str, float]]:
    states, dw = [], {}
    with open(DATA / "curated" / "survey_states_ghe2021.csv") as f:
        for r in csv.DictReader(f):
            states.append(r["health_state"])
            dw[r["health_state"]] = float(r["dw_ghe2021"])
    return states, dw


def extract_respondents(journal_path: Path) -> list[dict]:
    """Pull the {respondents:[...]} object from the workflow's return value.

    Scans journal.jsonl for the richest record carrying a 'respondents' list;
    the workflow's final return is logged there.
    """
    text = journal_path.read_text()
    # Two supported inputs: a task-output file (one pretty-printed JSON object)
    # and a journal.jsonl (one JSON object per line). Try whole-file first.
    try:
        found = _find_respondents(json.loads(text))
        if found:
            return found
    except json.JSONDecodeError:
        pass
    best: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or "respondents" not in line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        found = _find_respondents(obj)
        if found and len(found) > len(best):
            best = found
    return best


def _find_respondents(obj) -> list[dict] | None:
    if isinstance(obj, dict):
        r = obj.get("respondents")
        if isinstance(r, list) and r and isinstance(r[0], dict) and "pc" in r[0]:
            return r
        for v in obj.values():
            got = _find_respondents(v)
            if got:
                return got
    elif isinstance(obj, list):
        for v in obj:
            got = _find_respondents(v)
            if got:
                return got
    return None


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    journal = Path(sys.argv[1])
    states, dw_true = load_curated()
    answered = extract_respondents(journal)
    if not answered:
        sys.exit(f"no respondent answers found in {journal}")
    n = max(r["id"] for r in answered) + 1
    by_id = {r["id"]: r for r in answered}

    # Re-derive the exact assignment, then keep only respondents who answered.
    pc_assign = assign_pairs(states, n, PAIRS_PER_RESPONDENT, rng=PC_SEED)
    phe_assign = assign_phe(states, n, PHE_PER_RESPONDENT, rng=PHE_SEED)

    pc_rows, phe_rows = [], []
    for rid in sorted(by_id):
        pc_a = by_id[rid]["pc"]
        phe_a = by_id[rid]["phe"]
        pc_sub = pc_assign[pc_assign.respondent_id == rid].reset_index(drop=True)
        phe_sub = phe_assign[phe_assign.respondent_id == rid].reset_index(drop=True)
        if len(pc_a) != len(pc_sub) or len(phe_a) != len(phe_sub):
            print(f"skip respondent {rid}: answer/assignment length mismatch")
            continue
        pc_rows.append(parse_pc_responses(pc_sub, pc_a))
        phe_rows.append(parse_phe_responses(phe_sub, phe_a))

    pc_df = pd.concat(pc_rows, ignore_index=True)
    phe_df = pd.concat(phe_rows, ignore_index=True)
    print(f"ingested {len(pc_rows)} respondents | PC rows {len(pc_df)} | PHE rows {len(phe_df)}")

    weights, diag = estimate_dws(pc_df, phe_df)
    est = weights["dw"].reindex(states)
    true = pd.Series(dw_true).reindex(states)
    comp = pd.DataFrame({"published_ghe2021": true, "haiku_estimate": est.round(3)})
    comp["abs_error"] = (comp["haiku_estimate"] - comp["published_ghe2021"]).abs().round(3)
    comp = comp.sort_values("published_ghe2021")

    rank_corr = comp["published_ghe2021"].corr(comp["haiku_estimate"], method="spearman")
    pear = comp["published_ghe2021"].corr(comp["haiku_estimate"])
    print(f"\nanchor-map R^2 (PC vs PHE scale): {diag['anchor_map'].r_squared:.3f}")
    print(f"Spearman rank corr vs published: {rank_corr:.3f}")
    print(f"Pearson corr vs published:       {pear:.3f}")
    print(f"mean |error|: {comp['abs_error'].mean():.3f}   max |error|: {comp['abs_error'].max():.3f}\n")
    print(comp.to_string())

    out = DATA / "curated" / "haiku_vs_published_ghe2021.csv"
    comp.to_csv(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
