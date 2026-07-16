# Power curves for survey planning

Run: `studies/power_curves.py`, output `studies/results/power_curves.csv`. K=20 synthetic states (matches `test_recovery.py`; keeps the narrow anchor set at 4 states and the mid set at 9, both above `fit_anchor_map`'s 3-state minimum). Every design cell runs 20 seeded reps (distinct seeds per cell, rep, and module) of simulate-PC + simulate-PHE + `estimate_dws`; the precision proxy per cell is the across-rep SD of each state's estimate averaged over states, plus mean max|err| and mean mean|err|. PHE questions per respondent stays at the instrument default of 3 throughout. Zero reps failed in any cell. Total measured runtime 197 s, well inside the 10-minute cap, so nothing was cut — neither reps nor grid cells.

## Sample size (n_respondents at pairs=10, mid anchor)

Mean max|err| falls 0.0611, 0.0379, 0.0319, 0.0207, 0.0154 across N = 250, 500, 1000, 2000, 4000; mean mean|err| falls 0.0182, 0.0106, 0.0079, 0.0055, 0.0042; the across-rep SD proxy falls 0.0222, 0.0130, 0.0102, 0.0068, 0.0052. The max/mean error ratio sits around 3.4-4.0 throughout: the worst state (an extreme-weight state far from the anchor set) is consistently about 3-4x worse than the average state.

## Pairs per respondent (at N=1000, mid anchor)

Pairs = 5, 10, 15 give mean max|err| = 0.0381, 0.0319, 0.0277 and mean mean|err| = 0.0099, 0.0079, 0.0075. Tripling PC responses per person (5 to 15, +10,000 total responses) buys about what doubling N from 500 to 1000 buys — real but clearly diminishing, because past pairs=10 the binding error source is the PHE anchor stage, which pairs do not touch.

## PHE anchor width (at N=1000, pairs=10)

Narrowing the anchor set from dw in [0.05, 0.70] (9 states) to [0.10, 0.40] (4 states) raises mean max|err| from 0.0319 to 0.0446 and mean mean|err| from 0.0079 to 0.0142 — an 80% increase in average error at identical respondent count and identical questions per respondent. The anchor line is fitted on fewer, less-spread states, and the damage propagates to every weight through the rescaling, not just the excluded states.

## Root-N consistency check

Log-log regression of mean mean|err| on N over the five sample sizes gives slope -0.517 (r^2 = 0.984), confirming the expected 1/sqrt(N) convergence; the proxy behaves like a sampling-error measure and can be extrapolated as one.

## Bootstrap cross-check

At the baseline point (N=1000, pairs=10, mid anchor), `inference.bootstrap_dws` with n_boot=80 (0 failed) gives mean 95% CI width 0.0397, against the proxy-implied width 2 x 1.96 x 0.0102 = 0.0400 — a ratio of 0.99, so the cheap across-rep SD proxy and the respondent-cluster bootstrap agree essentially exactly at this design point.

## Design guidance

The smallest grid design meeting mean max|err| < 0.05 is N=500 respondents, pairs=10, mid anchor (max|err| = 0.0379 on 6,500 total responses); N=250 fails at 0.0611. The margin that buys the most precision per response is the PHE anchor width, because it costs zero additional responses: widening the anchor set from [0.10, 0.40] to [0.05, 0.70] cuts mean error by 44% with the same respondents answering the same number of questions, a benefit/cost ratio that N and pairs cannot match since both trade responses for error along a ~1/sqrt(responses) curve (each halving of error via N costs 4x the responses). Planning order: first make the PHE anchor set as wide as the instrument tolerates, then buy respondents to the error target (N=500 for a 0.05 max-error bar, N=2000 for 0.02), and treat pairs beyond 10 per respondent as the weakest margin.
