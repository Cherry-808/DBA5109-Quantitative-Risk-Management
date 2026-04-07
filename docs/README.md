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
│   ├── portfolio_core.py    M2 🔲      EWP / TAN / GMV + in-sample metrics + plots
│   ├── robust_portfolios.py M3 🔲      Shrinkage estimators + OOS evaluation + plots
│   └── data_challenge.py    M4 🔲      Final strategy, weight selection, CSV export
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

## 📋 PDF Deliverables Checklist

### Part 1 — Data Exploration and Analysis (15 pts)

#### § 1.1 Pre-processing
- [x] **1.1.1** Compute excess returns for 43 industries → `get_excess()` in `data_pipeline.py`

#### § 1.2 Basic Portfolio Construction & In-sample Analysis
- [ ] **1.2.1** Build EWP, TAN, GMV using full 1986-2015 data
- [ ] **1.2.2** Compute μ, σ, Sharpe, β for all 43 industries + MKT + EWP + TAN + GMV
- [ ] **Table 1** 3×4 in-sample performance table → `outputs/tables/insample_3x4.csv`
- [ ] **Figure 1** σ vs E[r] with 43 industries + special portfolios + in-sample EF → `outputs/figures/sigma_vs_er_insample.png`
- [ ] **Figure 2** β vs E[r] + Security Market Line (SML) → `outputs/figures/beta_vs_er_insample.png`

#### § 1.3 Robust Construction & Out-of-sample Analysis
- [ ] **1.3.1** Beta shrinkage: β_shrink = 0.5·β̄ + 0.5·β̂
- [ ] **1.3.2** CAPM expected return: μ_CAPM = β_shrink × E[r_MKT] (training mean)
- [ ] **1.3.3** Constant correlation cov matrix V_CC; shrinkage: V_shrink = 0.3·V_CC + 0.7·V̂
- [ ] **1.3.4** Build TAN-robust (uses V_shrink + μ_CAPM) and GMV-robust (uses V_shrink)
- [ ] **Figure 3** OOS σ vs E[r] — all portfolios on test data; include "true" EF + "realized" EF → `outputs/figures/sigma_vs_er_oos.png`
- [ ] **Table 2** 3×6 OOS performance table → `outputs/tables/oos_3x6.csv`
- [ ] **Written analysis** Contrast in-sample vs OOS; findings, insights, limitations → in report

### Part 2 — The Data Challenge (35 pts)
- [ ] **Strategy design** Any method; no post-2015 info; weights sum to 1; static
- [ ] **Recommendation CSV** 43 rows, one column, header = "G#" → `submission/Recommendation_G#.csv`
- [ ] **Writeup** Thought process, intermediate steps, visualizations → `docs/investment_thesis.md`
- [ ] **Week 13 presentation** All members present strategy, findings, reflections

### Final Zip — GP_G#.zip
- [ ] `report.pdf` or `report.docx`
- [ ] `slides.pptx`
- [ ] All `.py` source files
- [ ] `Recommendation_G#.csv`

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

## Formula Reference

### In-sample (M2)
- μ = sample mean of excess returns (full 360 months)
- σ = sample std of excess returns
- Sharpe = μ / σ  (RF treated as 0 after excess return step)
- β = Cov(r_i, r_MKT) / Var(r_MKT)
- w_EWP = 1/43
- w_TAN ∝ Σ⁻¹μ, normalized to sum = 1
- w_GMV ∝ Σ⁻¹1, normalized to sum = 1

### Robust estimation (M3)
- β_shrink = 0.5·β̄ + 0.5·β̂  where β̄ = mean of all 43 estimated betas
- μ_CAPM = β_shrink × E[r_MKT]  where E[r_MKT] = training-set mean of Mkt-RF
- V_CC[i,j] = σ_i·σ_j·ρ̄ for i≠j,  σ_i² for i=j  (ρ̄ = mean of all pairwise correlations)
- V_shrink = 0.3·V_CC + 0.7·V̂  (shrinkage constants fixed by problem brief)
- w_TAN-robust ∝ V_shrink⁻¹·μ_CAPM, normalized
- w_GMV-robust ∝ V_shrink⁻¹·1, normalized

### OOS evaluation split (M3)
- Training: rows 0–299, 1986–2010 (construct portfolios here)
- Test: rows 300–359, 2011–2015 (evaluate μ, σ, Sharpe here)
- "True" EF = efficient frontier computed from test-set parameters
- "Realized" EF = training-set weights applied to test-set returns

---

*Update the checklist above as each deliverable is completed.*
