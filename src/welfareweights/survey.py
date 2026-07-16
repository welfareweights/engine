"""Build survey instruments and parse responses into the engine's long tables.

This mirrors the GBD web instrument (Salomon 2012): each respondent answers a
handful of randomly assigned paired-comparison (PC) questions plus a few
population-health-equivalence (PHE) questions. The same assignment/parse code
serves any respondent source — human web survey, or the synthetic Haiku
respondents used to exercise the pipeline before real microdata arrives.

A survey run is fully specified by a seeded RNG so it is reproducible without
Date.now()/random state leaking in.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

from welfareweights.design import connected_components
from welfareweights.simulate import C_CHOICES, DEATHS


def assign_pairs(
    states: list[str],
    n_respondents: int,
    pairs_per_respondent: int = 15,
    rng: np.random.Generator | int | None = None,
) -> pd.DataFrame:
    """Assign each respondent a set of distinct unordered PC pairs.

    Order within a pair is randomized (which state is shown first), matching
    the instrument and avoiding a first-position bias in the design matrix.
    The pooled comparison graph is required to be connected — otherwise the
    probit is unidentified across components (see design.connected_components).
    """
    rng = np.random.default_rng(rng)
    k = len(states)
    max_pairs = k * (k - 1) // 2
    if pairs_per_respondent > max_pairs:
        raise ValueError(f"cannot draw {pairs_per_respondent} distinct pairs from {k} states")
    all_pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    rows = []
    for r in range(n_respondents):
        chosen = rng.choice(len(all_pairs), size=pairs_per_respondent, replace=False)
        for idx in chosen:
            i, j = all_pairs[idx]
            if rng.random() < 0.5:
                i, j = j, i
            rows.append({"respondent_id": r, "state_1": states[i], "state_2": states[j]})
    df = pd.DataFrame(rows)
    comps = connected_components(df.assign(y=0))
    if len(comps) > 1:
        raise ValueError(
            f"assigned comparison graph is disconnected ({len(comps)} components); "
            "increase pairs_per_respondent or n_respondents"
        )
    return df


def assign_phe(
    states: list[str],
    n_respondents: int,
    questions_per_respondent: int = 3,
    c_choices: tuple[int, ...] = C_CHOICES,
    deaths: int = DEATHS,
    rng: np.random.Generator | int | None = None,
) -> pd.DataFrame:
    """Assign each respondent PHE questions: a state and a beneficiary count c."""
    rng = np.random.default_rng(rng)
    rows = []
    for r in range(n_respondents):
        s_idx = rng.integers(0, len(states), size=questions_per_respondent)
        cs = rng.choice(np.asarray(c_choices), size=questions_per_respondent)
        for si, c in zip(s_idx, cs):
            rows.append(
                {"respondent_id": r, "state": states[si], "n_cases": int(c), "deaths": deaths}
            )
    return pd.DataFrame(rows)


def pc_question_text(desc: dict[str, str], state_1: str, state_2: str) -> str:
    """Render one PC question from lay descriptions (for a human or LLM respondent)."""
    return (
        "Two people have the same life expectancy. Assume each condition lasts "
        "for the rest of their life.\n"
        f"Person A {desc[state_1]}\n"
        f"Person B {desc[state_2]}\n"
        "Who is healthier overall — A or B?"
    )


def phe_question_text(desc: dict[str, str], state: str, n_cases: int, deaths: int) -> str:
    """Render one PHE question."""
    return (
        "Two health programs cost the same. Which produces the greater health benefit?\n"
        f"Program 1 prevents {deaths} people from dying (they would otherwise die "
        "this year).\n"
        f"Program 2 prevents {n_cases} people from having this for the rest of their "
        f"life: someone who {desc[state]}\n"
        "Which program — 1 or 2?"
    )


# Parsing contract, tried in order (see _extract_choice):
#   1. the whole answer is the bare token, allowing surrounding quotes/punctuation;
#   2. exactly one DISTINCT standalone token appears in the full text;
#   3. exactly one distinct standalone token appears in the last non-empty line.
# Anything still ambiguous is dropped and counted, never guessed. The PC token
# pattern matches uppercase only: after upcasing, prose like "It is a close
# call, but B" would match the article "a" and parse as A. The PHE pattern
# excludes digits inside larger numbers ("2,000", "1000", "1.5").
_PC_EXACT = re.compile(r"^[\"'(]*([ABab])[\"').:]*$")
_PC_TOKEN = re.compile(r"\b([AB])\b")
_PHE_EXACT = re.compile(r"^[\"'(]*([12])[\"').:]*$")
_PHE_TOKEN = re.compile(r"(?<![\d.,])([12])(?!\d|[.,]\d)")


def _extract_choice(text: object, exact_re: re.Pattern, token_re: re.Pattern) -> str | None:
    if not isinstance(text, str) or not text.strip():
        return None
    m = exact_re.match(text.strip())
    if m:
        return m.group(1).upper()
    last_line = text.rstrip().splitlines()[-1]
    for chunk in (text, last_line):
        found = set(token_re.findall(chunk))
        if len(found) == 1:
            return found.pop()
    return None


def _parse(
    assignment: pd.DataFrame,
    answers: list[str],
    exact_re: re.Pattern,
    token_re: re.Pattern,
    mapping: dict[str, int],
    cols: list[str],
    kind: str,
) -> pd.DataFrame:
    if len(answers) != len(assignment):
        raise ValueError("answers must align 1:1 with the assignment rows")
    out = assignment.copy().reset_index(drop=True)
    tokens = [_extract_choice(a, exact_re, token_re) for a in answers]
    out["y"] = pd.Series(tokens, index=out.index, dtype=object).map(mapping)
    n_dropped = int(out["y"].isna().sum())
    if n_dropped:
        warnings.warn(f"dropped {n_dropped}/{len(out)} unparseable or ambiguous {kind} answer(s)")
    return out.dropna(subset=["y"]).astype({"y": int})[cols]


def parse_pc_responses(assignment: pd.DataFrame, answers: list[str]) -> pd.DataFrame:
    """Attach A/B answers to a PC assignment, producing [.., y].

    answers[i] answers assignment row i; y = 1 iff state_1 (Person A) was
    judged healthier. Parsing follows the contract above _PC_EXACT.
    """
    return _parse(
        assignment,
        answers,
        _PC_EXACT,
        _PC_TOKEN,
        {"A": 1, "B": 0},
        ["respondent_id", "state_1", "state_2", "y"],
        "PC",
    )


def parse_phe_responses(assignment: pd.DataFrame, answers: list[str]) -> pd.DataFrame:
    """Attach 1/2 answers to a PHE assignment, producing [.., y].

    y = 1 iff Program 1 (averting deaths) was chosen. The assignment's deaths
    column, when present, is kept so fit_phe computes thresholds from the
    figure the respondent actually saw.
    """
    cols = ["respondent_id", "state", "n_cases", "y"]
    if "deaths" in assignment.columns:
        cols.insert(3, "deaths")
    return _parse(assignment, answers, _PHE_EXACT, _PHE_TOKEN, {"1": 1, "2": 0}, cols, "PHE")
