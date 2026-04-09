import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_pipeline import get_split, get_industry_names
from portfolio_core import ewp, gmv, tan

# ======================
# LOAD DATA
# ======================
train, test = get_split()
industries = get_industry_names()

r_train = train[industries].values   # (300, 43)
r_test  = test[industries].values    # (60, 43)

mkt_train = train['Mkt-RF'].values
mkt_test  = test['Mkt-RF'].values

n = r_train.shape[1]

# ======================
# BETA ESTIMATION
# ======================
def compute_beta_vector(r, mkt):
    betas = []
    for i in range(r.shape[1]):
        cov = np.cov(r[:, i], mkt)[0,1]
        var = np.var(mkt, ddof=1)
        betas.append(cov / var)
    return np.array(betas)

beta_hat = compute_beta_vector(r_train, mkt_train)

# SHRINKAGE
beta_bar = np.mean(beta_hat)
beta_shrink = 0.5 * beta_bar + 0.5 * beta_hat

# ======================
# CAPM EXPECTED RETURN
# ======================
mu_mkt_train = np.mean(mkt_train)

mu_capm = beta_shrink * mu_mkt_train   # μ_CAPM

# ======================
# COVARIANCE SHRINKAGE
# ======================
Sigma_hat = np.cov(r_train, rowvar=False)

# Constant correlation matrix
corr = np.corrcoef(r_train, rowvar=False)
avg_corr = np.mean(corr[np.triu_indices(n, k=1)])

std = np.sqrt(np.diag(Sigma_hat))

Sigma_cc = avg_corr * np.outer(std, std)
np.fill_diagonal(Sigma_cc, std**2)

# Final shrinkage
Sigma_shrink = 0.3 * Sigma_cc + 0.7 * Sigma_hat

# ======================
# BASE PORTFOLIOS (TRAIN)
# ======================
mu_train = r_train.mean(axis=0)

w_ewp, _, _ = ewp(n)
w_gmv, _, _ = gmv(Sigma_hat)
w_tan, _, _ = tan(mu_train, Sigma_hat)

# ======================
# ROBUST PORTFOLIOS
# ======================
w_tan_r, _, _ = tan(mu_capm, Sigma_shrink)
w_gmv_r, _, _ = gmv(Sigma_shrink)

# ======================
# OOS TABLE (3 x 6 WITH MKT)
# ======================

def evaluate_oos(w, r_test):
    r = r_test @ w
    mu = np.mean(r)
    sigma = np.std(r, ddof=1)
    sharpe = mu / sigma
    return mu, sigma, sharpe

# ---- Compute metrics ----
metrics = {}

metrics["EWP"] = evaluate_oos(w_ewp, r_test)
metrics["TAN"] = evaluate_oos(w_tan, r_test)
metrics["TAN-robust"] = evaluate_oos(w_tan_r, r_test)
metrics["GMV"] = evaluate_oos(w_gmv, r_test)
metrics["GMV-robust"] = evaluate_oos(w_gmv_r, r_test)

# ---- MKT (IMPORTANT: use test data directly) ----
mu_mkt = np.mean(mkt_test)
sigma_mkt = np.std(mkt_test, ddof=1)
sharpe_mkt = mu_mkt / sigma_mkt

metrics["MKT"] = (mu_mkt, sigma_mkt, sharpe_mkt)

# ---- Enforce correct column order ----
cols = ["MKT", "EWP", "TAN", "TAN-robust", "GMV", "GMV-robust"]

# ---- Build 3x6 table ----
rows = ["Return", "Sigma", "Sharpe"]

data = []
for i in range(3):
    row = []
    for key in cols:
        row.append(metrics[key][i])
    data.append(row)
    
# Add index name so it shows in CSV
df_oos = pd.DataFrame(data, index=rows, columns=cols)
df_oos.index.name = "Metric"

# SAVE TABLE
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

output_path = os.path.join(BASE_DIR, "outputs", "tables")
os.makedirs(output_path, exist_ok=True)

df_oos.to_csv(os.path.join(output_path, "M3_oos_3x6.csv"), index=True)

print("\nOOS Performance Table:")
print(df_oos)

# ======================
# TRUE vs REALIZED EF
# ======================

def efficient_frontier(mu, Sigma, points=60):
    from scipy.optimize import minimize
    
    targets = np.linspace(mu.min(), mu.max(), points)
    sigmas = []

    for t in targets:
        res = minimize(
            lambda w: w @ Sigma @ w,
            x0=np.ones(len(mu))/len(mu),
            constraints=[
                {'type':'eq','fun':lambda w: w.sum()-1},
                {'type':'eq','fun':lambda w: w @ mu - t}
            ]
        )
        sigmas.append(np.sqrt(res.fun) if res.success else np.nan)

    return targets, np.array(sigmas)

# ===== TRUE EF (TRAIN) =====
mu_train_vec = r_train.mean(axis=0)
Sigma_train = np.cov(r_train, rowvar=False)
mu_true, sigma_true = efficient_frontier(mu_train_vec, Sigma_train)

# ===== REALIZED EF (TEST) =====
mu_test_vec = r_test.mean(axis=0)
Sigma_test = np.cov(r_test, rowvar=False)
mu_real, sigma_real = efficient_frontier(mu_test_vec, Sigma_test)

# ===== INDUSTRY POINTS (TEST EVALUATION) =====
asset_mus = r_test.mean(axis=0)
asset_sigmas = r_test.std(axis=0, ddof=1)

# ======================
# FULL OOS σ vs E[r] PLOT
# ======================

# ===== PLOT =====
plt.figure(figsize=(12, 8))

# 43 industries
plt.scatter(asset_sigmas, asset_mus,
            color='lightsteelblue', alpha=0.6, s=40,
            label='43 Industry Portfolios')

# Label a few (avoid clutter)
for i, name in enumerate(industries):
    if asset_mus[i] > asset_mus.mean() + asset_mus.std():
        plt.annotate(name, (asset_sigmas[i], asset_mus[i]),
                     fontsize=7, alpha=0.7)

# ===== TRUE EF =====
valid_true = ~np.isnan(sigma_true)
plt.plot(sigma_true[valid_true], mu_true[valid_true],
         linestyle='--', linewidth=2,
         label='True EF (Train)')

# ===== REALIZED EF =====
valid_real = ~np.isnan(sigma_real)
plt.plot(sigma_real[valid_real], mu_real[valid_real],
         linestyle='-', linewidth=2,
         label='Realized EF (Test)')

# ===== SPECIAL PORTFOLIOS (evaluated on TEST) =====
def plot_point(name, w, color):
    mu, sigma, _ = evaluate_oos(w, r_test)
    plt.scatter(sigma, mu, color=color, s=120, marker='*', label=name)

plot_point("EWP", w_ewp, "orange")
plot_point("TAN", w_tan, "red")
plot_point("GMV", w_gmv, "green")
plot_point("TAN-R", w_tan_r, "purple")
plot_point("GMV-R", w_gmv_r, "darkgreen")

# ===== MKT (TEST) =====
mu_mkt = np.mean(mkt_test)
sigma_mkt = np.std(mkt_test, ddof=1)

plt.scatter(sigma_mkt, mu_mkt,
            color='black', s=150, marker='D',
            label='MKT')

# ===== FINAL STYLING =====
plt.xlabel('σ — Volatility (%)', fontsize=12)
plt.ylabel('E[r] — Mean Excess Return (%)', fontsize=12)
plt.title('Out-of-Sample σ vs E[r] (2011–2015)', fontsize=14)

plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)

#Save figure
fig_path = os.path.join(BASE_DIR, "outputs", "figures")
os.makedirs(fig_path, exist_ok=True)

plt.savefig(os.path.join(fig_path, "M3_sigma_vs_er_oos.png"))

print("\n✅ OOS analysis complete")