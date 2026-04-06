"""
data_challenge.py  —  Member 4: Strategy Lead
==============================================
TODO (M4):
  - compare_strategies(oos_results)    -> ranked summary
  - select_final_weights(...)          -> weights (43,)  summing to 1
  - export_csv(weights, group_num)     -> submission/Recommendation_G#.csv

Rules (from project brief):
  - weights must sum to 1
  - no info from 2016 onwards
  - static weights (no rebalancing)
  - evaluated on Sharpe ratio over 2016-2020

Deliverables:
  - submission/Recommendation_G#.csv   (43 rows, header = "G#")
  - docs/investment_thesis.md
"""

from data_pipeline import get_excess, get_split, get_industry_names
import numpy as np
import pandas as pd

GROUP_NUMBER = 1   # <-- change to your actual group number

# ── implement below ────────────────────────────────────────────────────────────
