# Group Project — GP_G4

---

## 📁 Project Structure

```
GP_G4/
├── data/                               ← Raw data (read-only)
│   ├── gp_data_1986_to_2015.csv        360 months × 46 cols (Kenneth R. French)
│   └── industry_descriptions.csv       43 industry code → full name
│
├── src/                                ← All source code (run from here)
│   ├── data_pipeline.py     M1 ✅      Data loading, excess returns, train/test split
│   ├── portfolio_core.py    M2 ✅      EWP / TAN / GMV + in-sample metrics + plots
│   ├── robust_portfolios.py M3 ✅      Shrinkage estimators + OOS evaluation + plots
│   └── data_challenge.py    M4 ✅      Final strategy, weight selection, CSV export
│
├── outputs/
│   ├── tables/
│   │   ├── insample_3x4.csv            M2: μ / σ / Sharpe for MKT, EWP, TAN, GMV
│   │   └── oos_3x6.csv                 M3: μ / σ / Sharpe for 6 portfolios OOS
│   └── figures/
│       ├── sigma_vs_er_insample.png    M2: in-sample efficient frontier plot
│       ├── beta_vs_er_insample.png     M2: SML plot
│       └── sigma_vs_er_oos.png         M3: OOS plot with true + realized EF
│
├── docs/
│   ├── README.md                       This file
│   └── investment_thesis.md            M4: Part 2 strategy writeup
│
└── submission/                         ← Final zip contents assembled here
    └── Recommendation_G#.csv           M4: 43 weights, header "G#", rows sum to 1
```
---

## Quick Start

```bash
# All scripts must be run from src/ so imports resolve correctly
cd src/
python data_pipeline.py       # verify M1 pipeline
python portfolio_core.py      # M2 run after implementing
python robust_portfolios.py   # M3 run after implementing
python data_challenge.py      # M4 run after implementing
```

Dependencies: `pip install numpy pandas scipy matplotlib`

---

## Import API (data_pipeline.py — M1 complete)

```python
from data_pipeline import get_excess, get_split, get_industry_names

excess       = get_excess()         # (360, 44): Mkt-RF + 43 industry excess returns
train, test  = get_split()          # train (300,44) 1986-2010 / test (60,44) 2011-2015
industries   = get_industry_names() # ['Food','Beer',...,'Fin']  43 strings
months       = get_month_index()    # YYYYMM integers for time-axis plots
desc         = load_industry_descriptions()  # code → full name lookup
```

Data facts: 360 rows, Jan 1986–Dec 2015, units = % per month, 0 missing values.
`Mkt-RF` is already an excess return — do not subtract RF again.

---

## Dependency Chain

```
data_pipeline.py  →  portfolio_core.py  →  robust_portfolios.py  →  data_challenge.py
```

Always run from `src/`. Do not move files between folders.

---
