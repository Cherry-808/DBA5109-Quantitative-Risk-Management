import numpy as np
#from portfolio_core import ewp, gmv, tan, evaluate_oos
import pandas as pd

from data_pipeline import get_split, get_industry_names
train, test = get_split()
industries = get_industry_names()
r_test = test[industries].values

def evaluate_oos(w, r_test):
    r = r_test @ w
    mu = np.mean(r)
    sigma = np.std(r, ddof=1)
    sharpe = mu / sigma
    return mu, sigma, sharpe

w_tan_r = np.array([ 0.05382095, -0.01975782, -0.02873589,  0.04746801,  0.16087652, -0.04446856,
 -0.04788742,  0.10186013,  0.03658721,  0.01135562,  0.01038386, -0.05565989,
 -0.01297929, -0.05059939, -0.02588856,  0.01566927,  0.03217007, -0.03062757,
 -0.04304643, -0.04828846, -0.04701428,  0.01585091, -0.01070437,  0.00888854,
 -0.05998093,  0.12911917,  0.28017127,  0.09772577, -0.01593142,  0.01551813,
  0.02047629,  0.01035753, -0.01856693,  0.05921578,  0.03548136,  0.13007504,
  0.14164976,  0.10033058,  0.06017061, -0.04133737, -0.00796258,  0.00827636,
  0.02593841])

w_gmv_r = np.array([0.08270895, -0.0079598, -0.02837922, 0.05400745, 0.20726999, -0.05924713,
-0.0511063, 0.11247958, 0.02364939, -0.02066647, -0.01560621, -0.05699074,
-0.04591732, -0.09043591, -0.0619129, 0.03044909, 0.00802749, -0.07829327,
-0.06044069, -0.0809222, -0.04759449, 0.05565136, 0.01469323, 0.00906463,
-0.06146388, 0.15822545, 0.4240995, 0.09714305, -0.01977289, -0.02259856,
0.0246238, 0.00867301, -0.04150843, 0.07805877, 0.06387852, 0.17711543,
0.12559952, 0.12335196, 0.07299444, -0.077208, -0.04438661, 0.03208427,
-0.01143784])


# allocate weights based on Sharpe
sharpe_tan_r = 0.4242503906441533	
sharpe_gmv_r = 0.40156935543186156
w_blend = (sharpe_tan_r * w_tan_r + sharpe_gmv_r * w_gmv_r) / (sharpe_tan_r + sharpe_gmv_r)
print("Blended weights:\n", w_blend)


mu, sigma, sharpe = evaluate_oos(w_blend, r_test)

print("Blended weights OOS：")
print("Return :", mu)
print("Sigma  :", sigma)
print("Sharpe :", sharpe)
print("Sum of weights :", w_blend.sum())


