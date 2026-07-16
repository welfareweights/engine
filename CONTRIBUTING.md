# Contributing

This project is a cross between an academic replication and an open-source library, so contributions are judged by both standards: code must pass the suite, and claims must be reproducible.

The contributions we most want, in order:

1. **Attacks on the validation argument.** Read `docs/VALIDATION.md` and try to break it: a misspecification the battery missed, data that fools a gate, a flaw in a study design. A demonstrated hole is more valuable than a feature.
2. **Independent replication.** Rerun the studies in `studies/` (seeds are recorded) and report any number that does not reproduce.
3. **Pointers to accessible disability-weight microdata.** The single blocking input; see `ROADMAP.md` item 1.
4. Code and documentation improvements.

Ground rules for pull requests:

- `pytest -q` must pass; new estimation behavior needs a test that would fail without it (see the house style in `tests/test_recovery.py`: docstrings state what evidence the test provides, tolerances are justified in comments).
- Every number in a document must be reproducible from committed code with a recorded seed.
- Every hyperlink must be verified before shipping: run `.venv/bin/python tools/check_links.py` (CI runs it too).
- No clinical health-state names or descriptions in this repository; the engine works on synthetic labels by design.
