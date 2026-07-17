"""Open implementation of the GBD disability-weight estimation pipeline.

Methodology: Salomon et al. 2012 (Lancet 380:2129-43, GBD 2010) and
Salomon et al. 2015 (Lancet Glob Health, GBD 2013). The pipeline has
three stages:

  A. Probit on paired-comparison (PC) responses -> health values on an
     arbitrary linear scale (``probit``).
  B. Interval-censored normal regression on population-health-equivalence
     (PHE) responses in logit space -> logit-scale disability weights
     anchored to death (``anchor``).
  C. Linear map of stage-A coefficients onto the stage-B scale, then
     numerical integration through the inverse logit -> disability
     weights on [0, 1] (``rescale``).

``simulate`` generates synthetic PC/PHE data from the exact data-generating
process the estimator assumes, so recovery of known true weights is the
package's core correctness test. ``pipeline`` chains A-C end to end.
"""

from welfareweights.inference import bootstrap_dws
from welfareweights.pipeline import estimate_dws

__all__ = ["estimate_dws", "bootstrap_dws"]
__version__ = "0.1.0"
