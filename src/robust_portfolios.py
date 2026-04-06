"""
robust_portfolios.py  —  Member 3: Quant Analyst
=================================================
TODO (M3):
  - shrink_beta(beta_hat)           -> beta_shrink (43,)
  - capm_mu(beta_shrink, mkt_mu)    -> mu_capm (43,)
  - constant_corr_cov(excess)       -> V_CC (43x43)
  - shrink_cov(V_hat, V_CC)         -> V_shrink (43x43)
  - build_tan_robust(V_shrink, mu_capm)   -> weights (43,)
  - build_gmv_robust(V_shrink)            -> weights (43,)
  - run_oos_evaluation(train, test)  -> results dict

Deliverables:
  - outputs/tables/oos_3x6.csv
  - outputs/figures/sigma_vs_er_oos.png
"""

from data_pipeline import get_split, get_industry_names
from portfolio_core import build_tan, build_gmv, compute_metrics
import numpy as np

# ── implement below ────────────────────────────────────────────────────────────
