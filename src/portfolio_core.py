"""
portfolio_core.py  —  Member 2: Portfolio Theory Specialist
===========================================================
TODO (M2):
  - build_ewp(excess)          -> weights (43,)
  - build_tan(mu, cov)         -> weights (43,)
  - build_gmv(cov)             -> weights (43,)
  - compute_metrics(weights, excess, mkt) -> dict {mu, sigma, sharpe, beta}
  - efficient_frontier(mu, cov, n_points) -> (sigmas, mus)

Deliverables:
  - outputs/tables/insample_3x4.csv
  - outputs/figures/sigma_vs_er_insample.png
  - outputs/figures/beta_vs_er_insample.png
"""

from data_pipeline import get_excess, get_split, get_industry_names
import numpy as np

# ── implement below ────────────────────────────────────────────────────────────
