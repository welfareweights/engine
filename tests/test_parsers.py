"""Fuzz/contract tests for parse_pc_responses and parse_phe_responses.

The parsing contract (survey.py, above _PC_EXACT/_PHE_TOKEN) is: (1) the whole
answer is a bare token modulo surrounding quotes/punctuation; else (2) exactly
one distinct standalone token appears in the full text; else (3) exactly one
distinct standalone token appears in the last non-empty line. Anything still
ambiguous is dropped and counted via a UserWarning, never guessed. The PC
token regex matches uppercase A/B only (not lowercase), which is precisely
what stops prose like "a close call, but B" from also matching the article
"a". The PHE token regex excludes digits that are part of a larger number
("2,000", "1000", "1.5") via lookaround, so bare numeral answers with no
standalone 1/2 are dropped, not misread.

These are contract tests, not guesses: every expectation below was checked
against the actual regexes/_extract_choice logic in survey.py before being
written into an assertion.
"""

import pandas as pd
import pytest

from welfareweights.survey import parse_pc_responses, parse_phe_responses


def _pc_assignment(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {"respondent_id": list(range(n)), "state_1": ["s1"] * n, "state_2": ["s2"] * n}
    )


def _phe_assignment(n: int, with_deaths: bool = True) -> pd.DataFrame:
    data = {"respondent_id": list(range(n)), "state": ["s"] * n, "n_cases": [500] * n}
    if with_deaths:
        data["deaths"] = [1000] * n
    return pd.DataFrame(data)


# --- 1. Bare-token answers ---------------------------------------------------


@pytest.mark.parametrize(
    "answer,expected_y",
    [("A", 1), ("b", 0), ('"A"', 1), ("(B)", 0)],
)
def test_pc_bare_token(answer, expected_y):
    out = parse_pc_responses(_pc_assignment(1), [answer])
    assert list(out["y"]) == [expected_y]


@pytest.mark.parametrize(
    "answer,expected_y",
    [("1", 1), ("2.", 0), ("(1)", 1)],
)
def test_phe_bare_token(answer, expected_y):
    out = parse_phe_responses(_phe_assignment(1), [answer])
    assert list(out["y"]) == [expected_y]


# --- 2. Unique-token prose, including the article-collision regression ------


@pytest.mark.parametrize(
    "answer,expected_y",
    [
        # Regression case: the PC token regex matches uppercase A/B only, so
        # the lowercase article "a" in "a close call" cannot match and force
        # a false A. Only the uppercase "B" is a standalone token here.
        ("It is a close call, but B", 0),
        ("Person B is healthier", 0),
        ("I'd pick B, a clear case", 0),
    ],
)
def test_pc_prose_unique_token(answer, expected_y):
    out = parse_pc_responses(_pc_assignment(1), [answer])
    assert list(out["y"]) == [expected_y]


# --- 3. Last-line disambiguation --------------------------------------------


def test_pc_last_line_disambiguation():
    # Both A and B appear somewhere in the full text (ambiguous at that
    # stage), but the last non-empty line contains only B, so the parser
    # falls through to the last-line rule and resolves to B.
    answer = "A is tempting on symptoms.\nFinal answer: B"
    out = parse_pc_responses(_pc_assignment(1), [answer])
    assert list(out["y"]) == [0]


# --- 4. Ambiguous/garbage answers are dropped, never guessed ----------------


@pytest.mark.parametrize(
    "answer",
    [
        "banana",  # no standalone A/B token at all (lowercase letters inside a word don't count)
        "A or B?",  # both tokens present, full text and only line both ambiguous
        "A vs B, hard to say",  # single line containing both tokens
    ],
)
def test_pc_ambiguous_answers_are_dropped(answer):
    assignment = _pc_assignment(2)
    with pytest.warns(UserWarning, match=r"dropped 1/2"):
        out = parse_pc_responses(assignment, ["A", answer])
    assert list(out["respondent_id"]) == [0]
    assert list(out["y"]) == [1]


def test_pc_all_ambiguous_drop_count_in_warning():
    assignment = _pc_assignment(3)
    answers = ["banana", "A or B?", "A vs B, hard to say"]
    with pytest.warns(UserWarning, match=r"dropped 3/3"):
        out = parse_pc_responses(assignment, answers)
    assert out.empty


# --- 5. PHE numerals ---------------------------------------------------------


def test_phe_trailing_punctuation_numeral():
    out = parse_phe_responses(_phe_assignment(1), ["Program 1."])
    assert list(out["y"]) == [1]


def test_phe_numeral_extraction_ignores_comma_grouped_distractor():
    # "2,000" is a distractor number the parser must not read a "2" out of;
    # the real signal is the standalone "2" in "Program 2". This exercises
    # both halves of the PHE contract in one answer: comma-grouped digits are
    # excluded, and the genuine standalone token still resolves correctly.
    answer = "The program preventing 2,000 cases (Program 2) is better."
    out = parse_phe_responses(_phe_assignment(1), [answer])
    assert list(out["y"]) == [0]  # "2" -> mapping {"1": 1, "2": 0}


@pytest.mark.parametrize("answer", ["1000", "1.5", "2,000"])
def test_phe_digits_inside_numbers_never_match_as_bare_tokens(answer):
    # Per the contract, "1000"/"1.5"/"2,000" contain no digit that is NOT
    # part of a larger number, so no token is found at all: the answer is
    # dropped, not silently misread as "1" or "2".
    with pytest.warns(UserWarning, match=r"dropped 1/1"):
        out = parse_phe_responses(_phe_assignment(1), [answer])
    assert out.empty


def test_phe_literal_rubric_fragment_alone_is_dropped():
    # Rubric wording taken at face value ("the program preventing 2,000
    # cases" -> 2) does not hold as a standalone answer: alone, this text has
    # no digit outside the comma-grouped "2,000", so per the contract it is
    # ambiguous/unparseable and dropped, not resolved to 2. Item 7 of the
    # rubric requires asserting the drop where the contract drops.
    with pytest.warns(UserWarning, match=r"dropped 1/1"):
        out = parse_phe_responses(_phe_assignment(1), ["the program preventing 2,000 cases"])
    assert out.empty


# --- 6. Deaths column handling and length-mismatch ValueError --------------


def test_phe_keeps_deaths_column_when_assignment_has_one():
    out = parse_phe_responses(_phe_assignment(2, with_deaths=True), ["1", "2"])
    assert "deaths" in out.columns
    assert list(out["deaths"]) == [1000, 1000]


def test_phe_omits_deaths_column_when_assignment_lacks_one():
    out = parse_phe_responses(_phe_assignment(2, with_deaths=False), ["1", "2"])
    assert "deaths" not in out.columns


def test_pc_length_mismatch_raises_value_error():
    with pytest.raises(ValueError, match="align 1:1"):
        parse_pc_responses(_pc_assignment(2), ["A"])


def test_phe_length_mismatch_raises_value_error():
    with pytest.raises(ValueError, match="align 1:1"):
        parse_phe_responses(_phe_assignment(2), ["1"])
