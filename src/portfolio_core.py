import sys, os
sys.path.insert(0, os.path.dirname(__file__))  # ensures src/ is on the path

from data_pipeline import get_split, get_industry_names
import numpy as np
import pandas as pd
from scipy.optimize import minimize

#change wd
os.chdir(os.path.dirname(os.path.abspath(__file__)))

train, test = get_split()
industries = get_industry_names()
#gets train and test dataframes, and the list of industry column names (43 total)

r_train = train[industries].values
mkt_train = train['Mkt-RF'].values
n = r_train.shape[1]

#============================================ Helper Functions ===================================================#

# -------- calculate betas ------------------#
def get_beta(port_returns, mkt_returns):
    cov_matrix = np.cov(port_returns, mkt_returns)
    beta = cov_matrix[0, 1] / np.var(mkt_returns, ddof=1)
    return beta

#---------- calculate sharpe ratios ---------#
def get_sharpe(mu, sigma):
    """Calculate Sharpe ratio. Since we're in excess return space, RF=0, so Sharpe = mu/sigma."""
    return mu / sigma

#================================================================================================================#



#------------- Get mean return ---------------#
def get_mu_vec(r_train):
    """Calculate the mean return vector (mu_vec) for the training data."""
    mu_vec = r_train.mean(axis=0)  # gives a (43,) vector of mean returns of each industry
    return mu_vec

#test print
mu_vec = get_mu_vec(r_train)
print("Mean returns (mu_vec):")
print(pd.DataFrame(mu_vec[np.newaxis, :], columns=industries))  
print(100*'-')

#------------- Get std of returns ---------------#
def get_std_vec(r_train):
    """Calculate the standard deviation vector (std_vec) for the training data."""
    std_vec = r_train.std(axis=0, ddof=1)  # gives a (43,) vector of std devs of each industry
    return std_vec  
#test print
std_vec = get_std_vec(r_train)
print("Standard deviations (std_vec):")
print(pd.DataFrame(std_vec[np.newaxis, :], columns=industries))
print(100*'-')

#------------- Get covariance matrix ---------------#
def get_cov_matrix(r_train):
    """Calculate the covariance matrix (Sigma) for the training data."""
    Sigma = np.cov(r_train, rowvar=False)  # gives a (43, 43) covariance matrix
    return Sigma
#test print
Sigma = get_cov_matrix(r_train)
print("Covariance matrix (Sigma):")
print(pd.DataFrame(Sigma, columns=industries, index=industries))
print(100*'-')


#-------------- EWP --------------------------------#
def ewp(n):   #returns ewp weights, mean and std
    """Calculate the Equal-Weighted Portfolio (EWP) weights."""
    w_ewp = np.ones(n) / n  # gives a (43,) vector of equal weights (1/43)
    mu = w_ewp @ mu_vec  # portfolio mean return = w' * mu_vec
    sigma = np.sqrt(w_ewp @ Sigma @ w_ewp)  #
    return w_ewp, mu, sigma

#test print
w_ewp, mu_ewp, sigma_ewp = ewp(n)
beta_ewp = get_beta(r_train @ w_ewp, mkt_train)
sharpe_ewp = get_sharpe(mu_ewp, sigma_ewp)
print("Equal-Weighted Portfolio (EWP) weights:")    # basically 0.023256 for every asset
print(pd.DataFrame(w_ewp[np.newaxis, :], columns=industries))
print("Mean return (mu_ewp):", mu_ewp)
print("Standard deviation (sigma_ewp):", sigma_ewp)
print("Beta (beta_ewp):", beta_ewp)
print("Sharpe ratio (sharpe_ewp):", sharpe_ewp)
print(100*'-')


#--------------- GMV ----------------------#
def gmv(cov):   #returns gmv weights, mean and std
    """Minimum variance portfolio: minimise w'Σw s.t. sum(w)=1, w≥0."""
    n = cov.shape[0]
    result = minimize(
        fun     = lambda w: w @ cov @ w,
        x0      = np.ones(n) / n,
        method  = 'SLSQP',
        constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
    )

    mu = result.x @ mu_vec
    sigma = np.sqrt(result.x @ cov @ result.x)
    return result.x, mu, sigma

w_gmv, mu_gmv, sigma_gmv = gmv(Sigma)
beta_gmv = get_beta(r_train @ w_gmv, mkt_train)
sharpe_gmv = get_sharpe(mu_gmv, sigma_gmv)

#test print
print("Minimum Variance Portfolio (GMV) weights:")
print(pd.DataFrame(w_gmv[np.newaxis, :], columns=industries))
print("Mean return (mu_gmv):", mu_gmv)
print("Standard deviation (sigma_gmv):", sigma_gmv)
print("Beta (beta_gmv):", beta_gmv)
print("Sharpe ratio (sharpe_gmv):", sharpe_gmv)
print(100*'-')


#------------------ TAN ----------------------#
def tan(mu_vec, cov):
    n = cov.shape[0]
    result = minimize(
        fun     = lambda w: -(w @ mu_vec) / np.sqrt(w @ cov @ w),  # just μ/σ
        x0      = np.ones(n) / n,
        method  = 'SLSQP',
        constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
    )
    mu    = result.x @ mu_vec
    sigma = np.sqrt(result.x @ cov @ result.x)
    return result.x, mu, sigma

# call it
w_tan, mu_tan, sigma_tan = tan(mu_vec, Sigma)
beta_tan = get_beta(r_train @ w_tan, mkt_train)
sharpe_tan = get_sharpe(mu_tan, sigma_tan)

print("Tangency Portfolio (TAN) weights:")
print(pd.DataFrame(w_tan[np.newaxis, :], columns=industries))
print("Mean return (mu_tan):", mu_tan)
print("Standard deviation (sigma_tan):", sigma_tan)
print("Beta (beta_tan):", beta_tan)
print("Sharpe ratio (sharpe_tan):", sharpe_tan)
print(100*'-')



#--------------- MKT ----------------------#
def mkt(mkt_train):   #returns mkt weights, mean and std
    """Market portfolio: uses Mkt-RF directly as the benchmark."""
    mu    = mkt_train.mean()
    sigma = mkt_train.std(ddof=1)
    beta  = 1.0    # by definition
    return mu, sigma, beta

mu_mkt, sigma_mkt, beta_mkt = mkt(mkt_train)
beta_mkt = get_beta(mkt_train, mkt_train)  # should be 1 by definition
sharpe_mkt = get_sharpe(mu_mkt, sigma_mkt)

#test print
print("Market Portfolio (MKT):")
print("Mean return (mu_mkt):", mu_mkt)
print("Standard deviation (sigma_mkt):", sigma_mkt)
print("Beta (beta_mkt):", beta_mkt)
print("Sharpe ratio (sharpe_mkt):", sharpe_mkt)
print(100*'-')


#-------- Each 43 Industry portfolio (1 asset only) ------------------#
def get_industry_stats(r_train, mkt_train, industries):
    """
    Calculate mu, sigma, sharpe, beta for each individual industry.
    Returns a DataFrame with industries as rows and stats as columns.
    """
    mu_mkt = mkt_train.mean()
    results = []

    for i, name in enumerate(industries):
        r_i     = r_train[:, i]                                          # (300,) return series for industry i
        mu      = r_i.mean()
        sigma   = r_i.std(ddof=1)
        sharpe  = mu / sigma
        beta    = np.cov(r_i, mkt_train)[0, 1] / np.var(mkt_train, ddof=1)

        results.append({
            'Industry' : name,
            'μ'        : round(mu, 4),
            'σ'        : round(sigma, 4),
            'Sharpe'   : round(sharpe, 4),
            'β'        : round(beta, 4),
        })

    df = pd.DataFrame(results).set_index('Industry')
    return df


# test print
industry_stats = get_industry_stats(r_train, mkt_train, industries)
print("\nIndividual Industry Stats (In-Sample: 1986–2010)")
print("=" * 55)
print(industry_stats.to_string())
print("=" * 55)
print(100*'-')



#----- plot EF, with GMV, TAN, EWP points highlighted------#
import matplotlib.pyplot as plt

def plot_ef(mu_vec, Sigma, mu_gmv, sigma_gmv, mu_tan, sigma_tan, mu_ewp, sigma_ewp, mu_mkt, sigma_mkt, industries):
    """
    Plot the Efficient Frontier with GMV, TAN, EWP, and individual assets marked.
    """

    # ── 1. Individual asset points 
    asset_mus    = mu_vec                                        # (43,)
    asset_sigmas = np.sqrt(np.diag(Sigma))                      # (43,) — std of each asset

    # ── 2. Trace the Efficient Frontier 
    target_mus = np.linspace(asset_mus.min(), asset_mus.max(), 200)
    ef_sigmas  = []

    for target in target_mus:
        res = minimize(
            fun         = lambda w: w @ Sigma @ w,
            x0          = np.ones(n) / n,
            method      = 'SLSQP',
            constraints = [
                {'type': 'eq', 'fun': lambda w: w.sum() - 1},
                {'type': 'eq', 'fun': lambda w: w @ mu_vec - target}
            ]
        )
        ef_sigmas.append(np.sqrt(res.fun) if res.success else np.nan)

    ef_sigmas = np.array(ef_sigmas)

    # ── 3. Plot 
    fig, ax = plt.subplots(figsize=(12, 7))

    # Individual assets
    ax.scatter(asset_sigmas, asset_mus,
               color='lightsteelblue', alpha=0.6, s=40, zorder=2, label='Individual Assets')

    # Label a few notable assets to avoid clutter
    for i, name in enumerate(industries):
        if asset_mus[i] > 0.8 or asset_sigmas[i] < 3 or asset_sigmas[i] > 9:
            ax.annotate(name, (asset_sigmas[i], asset_mus[i]),
                        fontsize=7, alpha=0.7,
                        xytext=(4, 2), textcoords='offset points')

    # Efficient Frontier curve
    valid = ~np.isnan(ef_sigmas)
    ax.plot(ef_sigmas[valid], target_mus[valid],
            color='steelblue', linewidth=2, zorder=3, label='Efficient Frontier')

    # GMV point
    ax.scatter(sigma_gmv, mu_gmv,
               color='green', s=120, zorder=5, marker='*', label=f'GMV  (μ={mu_gmv:.3f}, σ={sigma_gmv:.3f})')

    # TAN point
    ax.scatter(sigma_tan, mu_tan,
               color='red', s=120, zorder=5, marker='*', label=f'TAN  (μ={mu_tan:.3f}, σ={sigma_tan:.3f})')

    # EWP point
    ax.scatter(sigma_ewp, mu_ewp,
               color='orange', s=120, zorder=5, marker='*', label=f'EWP  (μ={mu_ewp:.3f}, σ={sigma_ewp:.3f})')

    # MKT point
    ax.scatter(sigma_mkt, mu_mkt,
               color='black', s=120, zorder=5, marker='D', label=f'MKT  (μ={mu_mkt:.3f}, σ={sigma_mkt:.3f})')
    
    # Capital Market Line from origin through TAN
    cml_x = np.linspace(0, ef_sigmas[valid].max() * 1.1, 100)
    sharpe_tan = mu_tan / sigma_tan
    ax.plot(cml_x, sharpe_tan * cml_x,
            color='red', linewidth=1.2, linestyle='--', alpha=0.6, label=f'CAL (Sharpe={sharpe_tan:.3f})')

    ax.set_xlabel('σ — Monthly Std Dev (%)', fontsize=12)
    ax.set_ylabel('μ — Monthly Mean Excess Return (%)', fontsize=12)
    ax.set_title('Efficient Frontier (In-Sample: 1986–2010)', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    # Label the industry with the lowest mean return
    min_idx = np.argmin(asset_mus)
    ax.annotate(industries[min_idx], (asset_sigmas[min_idx], asset_mus[min_idx]),
                fontsize=7, alpha=0.7,
                xytext=(4, 2), textcoords='offset points')
    plt.tight_layout()
    plt.savefig('M2_sigma_vs_er_insample.png', dpi=150, bbox_inches='tight')
    plt.show()

plot_ef(mu_vec, Sigma, mu_gmv, sigma_gmv, mu_tan, sigma_tan, mu_ewp, sigma_ewp, mu_mkt, sigma_mkt, industries)


#-------------------- plot SML / Beta vs E[r] diagram --------------------------#

def plot_sml(industry_stats, mu_ewp, mu_gmv, mu_tan, mu_mkt,
             beta_ewp, beta_gmv, beta_tan, beta_mkt, mkt_train):
    """
    Plot the Security Market Line (SML) with all 43 industries, MKT, EWP, GMV, TAN marked.
    """

    # ── 1. Individual asset betas and mus from industry_stats
    asset_betas = industry_stats['β'].values
    asset_mus   = industry_stats['μ'].values
    asset_names = industry_stats.index.tolist()

    # ── 2. SML line 
    mu_mkt_val  = mkt_train.mean()
    beta_min    = min(asset_betas.min(), beta_ewp, beta_gmv, beta_tan) * 0.9
    beta_max    = max(asset_betas.max(), beta_ewp, beta_gmv, beta_tan) * 1.1
    beta_range  = np.linspace(beta_min, beta_max, 200)
    sml         = beta_range * mu_mkt_val   # E[r] = beta * mu_mkt (RF=0)

    # ── 3. Plot 
    fig, ax = plt.subplots(figsize=(11, 7))

    # Individual industry assets
    ax.scatter(asset_betas, asset_mus,
               color='lightsteelblue', alpha=0.7, s=50, zorder=2, label='43 Industry Portfolios')

    # Label all 43 industries
    for i, name in enumerate(asset_names):
        ax.annotate(name, (asset_betas[i], asset_mus[i]),
                    fontsize=6.5, alpha=0.75,
                    xytext=(4, 2), textcoords='offset points')

    # SML
    ax.plot(beta_range, sml,
            color='black', linewidth=1.5, linestyle='--', zorder=3,
            label=f'SML: E[r] = β × {mu_mkt_val:.3f}')

    # MKT point
    ax.scatter(beta_mkt, mu_mkt,
               color='black', s=150, zorder=5, marker='D',
               label=f'MKT  (β={beta_mkt:.3f}, μ={mu_mkt:.3f})')

    # EWP point
    ax.scatter(beta_ewp, mu_ewp,
               color='orange', s=200, zorder=5, marker='*',
               label=f'EWP  (β={beta_ewp:.3f}, μ={mu_ewp:.3f})')

    # GMV point
    ax.scatter(beta_gmv, mu_gmv,
               color='green', s=200, zorder=5, marker='*',
               label=f'GMV  (β={beta_gmv:.3f}, μ={mu_gmv:.3f})')

    # TAN point
    ax.scatter(beta_tan, mu_tan,
               color='red', s=200, zorder=5, marker='*',
               label=f'TAN  (β={beta_tan:.3f}, μ={mu_tan:.3f})')

    # Reference lines
    ax.axhline(0, color='grey', linewidth=0.8, linestyle='-', alpha=0.4)
    ax.axvline(0, color='grey', linewidth=0.8, linestyle='-', alpha=0.4)

    ax.set_xlabel('β (Beta)', fontsize=12)
    ax.set_ylabel('E[r] — Monthly Mean Excess Return (%)', fontsize=12)
    ax.set_title('Security Market Line (In-Sample: 1986–2010)', fontsize=14)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('M2_beta_vs_er_insample.png', dpi=150, bbox_inches='tight')
    plt.show()

plot_sml(industry_stats, mu_ewp, mu_gmv, mu_tan, mu_mkt,
         beta_ewp, beta_gmv, beta_tan, beta_mkt, mkt_train)


######## SAVE OUTPUTS ###################

#----------- in sample 3x4 -------------------#
summary = pd.DataFrame({
    'MKT': [mu_mkt,   sigma_mkt,   sharpe_mkt],
    'EWP': [mu_ewp,   sigma_ewp,   sharpe_ewp],
    'GMV': [mu_gmv,   sigma_gmv,   sharpe_gmv],
    'TAN': [mu_tan,   sigma_tan,   sharpe_tan],
}, index=['mu', 'sigma', 'Sharpe'])

summary.to_csv('M2_insample_3x4.csv')
print("\nIn-Sample Summary (3x4):")
print(summary.round(4))
