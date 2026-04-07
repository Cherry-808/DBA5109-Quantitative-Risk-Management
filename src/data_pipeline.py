"""
data_pipeline.py  —  Member 1: Data Engineer
=============================================
Functions exported for use by M2, M3, M4:
  - load_raw()          raw DataFrame (360 rows x 46 cols)
  - get_excess()        excess returns matrix: industries + MKT (360 x 44)
  - get_split()         (train_df, test_df) split at 1986-2010 / 2011-2015
  - get_industry_names() list of 43 industry column names (cleaned)
  - summary()           quick sanity-check printout

Units: all returns are in % per month (as supplied by French Data Library).
Treat RF = 0 going forward once excess returns are computed.

Data notes:
  - 'Month' column is YYYYMM integer format (e.g. 198601 = Jan 1986)
  - 'Mkt-RF' is already the market EXCESS return (MKT minus RF)
  - 'RF' is the riskless rate (kept for reference, not used after this step)
  - 43 industry columns have trailing spaces in their names — stripped below
  - No missing values in the dataset
"""

import pandas as pd
import numpy as np

# ── file path ──────────────────────────────────────────────────────────────────
# Paths are relative to the project root (one level above src/)
# This works whether you run from src/ or from the project root.
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))  # → .../GP_G1/src
_ROOT = _os.path.dirname(_HERE)                        # → .../GP_G1

DATA_PATH     = _os.path.join(_ROOT, "data", "gp_data_1986_to_2015.csv")
IND_DESC_PATH = _os.path.join(_ROOT, "data", "industry_descriptions.csv")

# ── internal helper ────────────────────────────────────────────────────────────
def _strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Strip trailing/leading spaces from all column names."""
    df.columns = df.columns.str.strip()
    return df


# ── public API ─────────────────────────────────────────────────────────────────
def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the raw CSV exactly as-is, but strip column name whitespace.
    Returns a 360-row DataFrame with columns:
      Month, Mkt-RF, RF, Food, Beer, ..., Fin  (46 cols total, names cleaned)
    """
    df = pd.read_csv(path)
    df = _strip_cols(df)
    return df


def get_industry_names(path: str = DATA_PATH) -> list:
    """
    Return the list of 43 industry column names (whitespace-stripped).
    Order matches the original CSV column order.
    """
    df = load_raw(path)
    # All columns except Month, Mkt-RF, RF
    return [c for c in df.columns if c not in ('Month', 'Mkt-RF', 'RF')]


def get_excess(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Compute and return excess returns for all assets.

    Returns a DataFrame with 360 rows (Jan 1986 – Dec 2015) and 44 columns:
      - 'Mkt-RF'      : market excess return (already provided as-is)
      - 'Food', 'Beer', ..., 'Fin'   : 43 industry excess returns = r_i - RF

    !! WARNING — PORTFOLIO OPTIMIZATION (M2, M3) !!
    This function returns 44 columns: 43 industries + 1 market column (Mkt-RF).
    When building EWP / TAN / GMV, you must use ONLY the 43 industry columns.
    NEVER pass the full 44-column DataFrame into the optimizer — Mkt-RF is a
    benchmark, not an investable asset in this problem.

    Correct usage:
        excess      = get_excess()           # 44 cols
        industries  = get_industry_names()   # 43 industry names
        r_industries = excess[industries]    # 43 cols  ← use this for optimization
        r_mkt        = excess['Mkt-RF']      # 1 col    ← use this for beta / SML

    !! WARNING — TIME-SERIES PLOTTING (M2, M3) !!
    This DataFrame has NO Month column. To label the x-axis when plotting:
        months = get_month_index()           # 360 YYYYMM integers, same row order
    Then align manually: months and excess share the same integer index 0..359.

    Note: After this step, treat RF = 0. All downstream portfolio math
    (Sharpe = mu/sigma, tangency weights, etc.) uses these excess returns directly.
    """
    df = load_raw(path)
    rf = df['RF'].values  # shape (360,)

    industries = get_industry_names(path)

    excess = pd.DataFrame()
    excess['Mkt-RF'] = df['Mkt-RF'].values  # already excess

    for col in industries:
        excess[col] = df[col].values - rf   # r_excess = r_industry - RF

    return excess


def get_split(path: str = DATA_PATH):
    """
    Split excess returns into training and test sets.

    Training : Jan 1986 – Dec 2010  (25 years = 300 months, rows 0..299)
    Test     : Jan 2011 – Dec 2015  ( 5 years =  60 months, rows 300..359)

    Returns
    -------
    train : pd.DataFrame  shape (300, 44)
    test  : pd.DataFrame  shape ( 60, 44)

    Both DataFrames have the same 44 columns as get_excess():
      43 industry columns + 'Mkt-RF'. Use get_industry_names() to slice
      just the 43 investable assets before running any optimization.

    Correct usage (M3):
        train, test = get_split()
        industries  = get_industry_names()
        r_train     = train[industries]   # (300, 43)  for optimization
        mkt_train   = train['Mkt-RF']     # (300,)     for beta / CAPM mu
        r_test      = test[industries]    # (60, 43)   for OOS evaluation
        mkt_test    = test['Mkt-RF']      # (60,)      for OOS beta check

    Indices are reset to 0-based within each split.
    """
    excess = get_excess(path)
    train = excess.iloc[:300].reset_index(drop=True)
    test  = excess.iloc[300:].reset_index(drop=True)
    return train, test


def get_month_index(path: str = DATA_PATH) -> pd.Series:
    """
    Return the 'Month' column as a Series (YYYYMM integers), aligned with
    get_excess() row order. Useful for plotting time-series charts.
    """
    df = load_raw(path)
    return df['Month'].reset_index(drop=True)


def load_industry_descriptions(path: str = IND_DESC_PATH) -> pd.DataFrame:
    """
    Load industry_descriptions.csv and return a clean lookup table.
    Columns: ['code', 'name']  where 'code' matches get_industry_names().
    """
    desc = pd.read_csv(path)
    desc.columns = desc.columns.str.strip()
    # Rename for convenience
    desc = desc.rename(columns={
        'Industry Code': 'code',
        'Industry Name': 'name',
        'Industry Number': 'number'
    })
    desc['code'] = desc['code'].str.strip()
    return desc[['number', 'code', 'name']]


# ── sanity check ───────────────────────────────────────────────────────────────
def summary(path: str = DATA_PATH):
    """Print a quick sanity-check of the pipeline outputs."""
    raw = load_raw(path)
    excess = get_excess(path)
    train, test = get_split(path)
    industries = get_industry_names(path)
    months = get_month_index(path)

    print("=" * 55)
    print("DATA PIPELINE SUMMARY")
    print("=" * 55)
    print(f"Raw data        : {raw.shape[0]} rows x {raw.shape[1]} cols")
    print(f"Date range      : {months.iloc[0]} – {months.iloc[-1]}")
    print(f"Missing values  : {raw.isnull().sum().sum()}")
    print(f"Industry count  : {len(industries)}")
    print()
    print(f"Excess returns  : {excess.shape}  (44 cols = MKT + 43 industries)")
    print(f"Training set    : {train.shape}  (rows 0–299, 1986–2010)")
    print(f"Test set        : {test.shape}   (rows 300–359, 2011–2015)")
    print()
    print("Excess return sample (first 3 rows, first 5 cols):")
    print(excess.iloc[:3, :5].to_string(index=False))
    print()

    # Verify: excess = raw industry - RF for one column
    raw2 = load_raw(path)
    col = industries[0]  # 'Food'
    check = (raw2[col].values - raw2['RF'].values - excess[col].values)
    print(f"Verification (Food excess re-check max diff): {np.max(np.abs(check)):.2e}")
    print()
    print("Column names (industries):")
    print(industries)
    print()
    print('MKT mean:', excess['Mkt-RF'].mean())
    print('MKT std:', excess['Mkt-RF'].std())
    print('Ind mean range:', excess[industries].mean().min(), excess[industries].mean().max())
    print('Ind std range:', excess[industries].std().min(), excess[industries].std().max())
    print("=" * 55)


# ── run sanity check when executed directly ────────────────────────────────────
if __name__ == "__main__":
    summary()