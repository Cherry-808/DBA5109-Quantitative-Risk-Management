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
# OOS PERFORMANCE
# ======================
def evaluate_oos(w, r_test):
    r = r_test @ w
    mu = np.mean(r)
    sigma = np.std(r, ddof=1)
    sharpe = mu / sigma
    return mu, sigma, sharpe

results = []

# MKT
mu = np.mean(mkt_test)
sigma = np.std(mkt_test, ddof=1)
sharpe = mu / sigma
results.append(["MKT", mu, sigma, sharpe])

# portfolios
for name, w in [
    ("EWP", w_ewp),
    ("TAN", w_tan),
    ("TAN-robust", w_tan_r),
    ("GMV", w_gmv),
    ("GMV-robust", w_gmv_r),
]:
    mu, sigma, sharpe = evaluate_oos(w, r_test)
    results.append([name, mu, sigma, sharpe])

df_oos = pd.DataFrame(results, columns=["Portfolio","Return","Vol","Sharpe"])

# SAVE TABLE
os.makedirs("../outputs/tables", exist_ok=True)
df_oos.to_csv("../outputs/tables/oos_3x6.csv", index=False)

print("\nOOS Performance Table:")
print(df_oos)

# ======================
# STEP 7 — TRUE vs REALIZED EF
# ======================
def efficient_frontier(mu, Sigma, points=50):
    target = np.linspace(mu.min(), mu.max(), points)
    sigmas = []

    for t in target:
        from scipy.optimize import minimize
        res = minimize(
            lambda w: w @ Sigma @ w,
            x0=np.ones(n)/n,
            constraints=[
                {'type':'eq','fun':lambda w: w.sum()-1},
                {'type':'eq','fun':lambda w: w @ mu - t}
            ]
        )
        sigmas.append(np.sqrt(res.fun) if res.success else np.nan)

    return target, np.array(sigmas)

# TRUE EF (train)
mu_train = r_train.mean(axis=0)
target_train, sigma_train = efficient_frontier(mu_train, Sigma_hat)

# REALIZED EF (test)
mu_test = r_test.mean(axis=0)
Sigma_test = np.cov(r_test, rowvar=False)
target_test, sigma_test = efficient_frontier(mu_test, Sigma_test)

# ======================
# STEP 8 — PLOT OOS
# ======================
plt.figure(figsize=(10,7))

# EF curves
plt.plot(sigma_train, target_train, label="True EF (Train)", linestyle='--')
plt.plot(sigma_test, target_test, label="Realized EF (Test)", linestyle='-')

# portfolios
for name, w in [
    ("EWP", w_ewp),
    ("TAN", w_tan),
    ("TAN-R", w_tan_r),
    ("GMV", w_gmv),
    ("GMV-R", w_gmv_r),
]:
    mu, sigma, _ = evaluate_oos(w, r_test)
    plt.scatter(sigma, mu, label=name)

# MKT
plt.scatter(np.std(mkt_test), np.mean(mkt_test), label="MKT", marker='x')

plt.xlabel("σ")
plt.ylabel("E[r]")
plt.legend()
plt.title("Out-of-Sample σ vs E[r]")
plt.grid(True)

os.makedirs("../outputs/figures", exist_ok=True)
plt.savefig("../outputs/figures/sigma_vs_er_oos.png")
plt.close()

print("\n✅ OOS analysis complete")