"""Survey assignment -> parse -> estimate round trip on the DGP.

Uses simulated (not LLM) answers so it is fast and deterministic; it checks
that the instrument-building and response-parsing code wires correctly into
the estimator, independent of who answers the questions.
"""

import numpy as np
from scipy.special import expit, logit
from scipy.stats import norm

from welfareweights.pipeline import estimate_dws
from welfareweights.survey import (
    assign_pairs,
    assign_phe,
    parse_pc_responses,
    parse_phe_responses,
)

STATES = [f"s{i:02d}" for i in range(12)]
TRUE_DWS = expit(np.linspace(logit(0.05), logit(0.70), len(STATES)))
DW = dict(zip(STATES, TRUE_DWS))


def _answer_pc(row, rng):
    theta1 = -logit(DW[row.state_1])
    theta2 = -logit(DW[row.state_2])
    return "A" if rng.random() < norm.cdf(theta1 - theta2) else "B"


def _answer_phe(row, rng):
    latent = logit(DW[row.state]) + rng.normal(0, 0.8)
    return "1" if latent < logit(row.deaths / row.n_cases) else "2"


def test_assignment_is_connected_and_distinct():
    pc = assign_pairs(STATES, n_respondents=50, pairs_per_respondent=10, rng=1)
    for _, grp in pc.groupby("respondent_id"):
        pairs = {frozenset((a, b)) for a, b in zip(grp.state_1, grp.state_2)}
        assert len(pairs) == len(grp)  # distinct within respondent


def test_survey_round_trip_recovers_weights():
    rng = np.random.default_rng(0)
    pc = assign_pairs(STATES, n_respondents=1500, pairs_per_respondent=12, rng=1)
    phe = assign_phe(STATES, n_respondents=2500, questions_per_respondent=3, rng=2)

    pc_ans = [_answer_pc(r, rng) for r in pc.itertuples()]
    phe_ans = [_answer_phe(r, rng) for r in phe.itertuples()]
    pc_df = parse_pc_responses(pc, pc_ans)
    phe_df = parse_phe_responses(phe, phe_ans)

    weights, diag = estimate_dws(pc_df, phe_df)
    est = weights["dw"].reindex(STATES).to_numpy()
    assert np.corrcoef(est, TRUE_DWS)[0, 1] > 0.99
    assert np.max(np.abs(est - TRUE_DWS)) < 0.06


def test_parser_drops_unparseable_and_maps_sides():
    pc = assign_pairs(STATES[:3], n_respondents=1, pairs_per_respondent=3, rng=5)  # triangle
    parsed = parse_pc_responses(pc, ["A", "banana", "B"])
    assert list(parsed["y"]) == [1, 0]  # middle row dropped, A->1, B->0
