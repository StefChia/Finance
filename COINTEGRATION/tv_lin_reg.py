import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
import plotly.express as px

import statsmodels.api as sm
from sklearn.metrics import r2_score
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
from pykalman import KalmanFilter  # install pykalman

from framework import download_historical_prices, compute_returns, cointegration_hold, autocorrelation_at_lags, bipolar_correlation, bipolar_autocorrelation, alt_corr_1, cointegration_daily
def plot_res(res,index=None):
    "Plot residuals"
    fig,ax = plt.subplots()
    ax.plot(index if index is not None else range(len(res)), res)
    plt.show()

def check_autocorr(series):
    series = pd.Series(series).dropna()
    shifted = series.shift(1)
    corr = series.corr(shifted)
    print(f"Lag-1 autocorrelation: {corr:.4f}")
    return corr




tickers = ['ETH-USD','SOL-USD']

prices = download_historical_prices(tickers,type='Close')
ret = compute_returns(prices)


l = int(len(ret) * 0.75)
index_test = prices.index[l:]

#print(prices)
#print(ret)

# Features + constant
X = sm.add_constant(prices[['SOL-USD']])
x_train = X.iloc[:l]
x_test = X.iloc[l:]
#X = prices[['BTC-USD','SOL-USD']]
y = prices['ETH-USD']



#Try TV-model (Kalman Filter)

# 1) Work on aligned prices
df = prices[['ETH-USD','SOL-USD']].dropna()
y = df['ETH-USD'].to_numpy(float).reshape(-1, 1)  # (T,1)
x = df['SOL-USD'].to_numpy(float)                 # (T,)
T = len(x)

# 2) Time-varying observation matrix: Z_t = [1, x_t]
Z = np.column_stack([np.ones(T), x]).reshape(T, 1, 2)  # (T,1,2)

# 3) State-space: [alpha_t, beta_t]' is a random walk
n_state = 2
Q = 1e-8 * np.eye(n_state)           # state noise (tune or learn via EM)   #IMPORTANTE:QUESTO è fondamentale, più è alto più il modello corregge velocemnte.
R = np.array([[1e-3]])               # obs noise  (tune or learn via EM)

kf = KalmanFilter(
    transition_matrices=np.eye(n_state),       # (2,2)
    observation_matrices=Z,                    # (T,1,2)
    transition_covariance=Q,                   # (2,2)
    observation_covariance=R,                  # (1,1)
    initial_state_mean=np.zeros(n_state),      # (2,)
    initial_state_covariance=1e3*np.eye(n_state)
)

# Optional: learn Q,R from data
# kf = kf.em(y, n_iter=30)

state_means, state_covs = kf.filter(y)
alpha_t = state_means[:, 0]
beta_t  = state_means[:, 1]
resid_t = y[:, 0] - (alpha_t + beta_t * x)     # time-varying residual


print(state_means[1:15,:2])
print(state_means[-15:,:2])




plot_res(resid_t)
autocorr = check_autocorr(resid_t)
#print(autocorr)

#ALTERNATIVE METRICS
bipolar = bipolar_autocorrelation(resid_t)
print(bipolar)
b,c = alt_corr_1(resid_t[1:],resid_t[:-1])
print(b)
print(c)



resid_t_diff = np.diff(resid_t)
plot_res(resid_t_diff)
autocorr_diff = check_autocorr(resid_t_diff)
#print(autocorr_diff)

#ALTERNATIVE METRICS
bipolar_diff = bipolar_autocorrelation(resid_t_diff)
print(bipolar_diff)
b,c = alt_corr_1(resid_t_diff[1:],resid_t_diff[:-1])
print(b)
print(c)









#TEST Augmented-Dickey Fuller
#RECALL: Null hypothesis (H₀): Residuals have a unit root (non-stationary).
adf_result = adfuller(resid_t)

# 5. Show results
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])
print("Critical Values:", adf_result[4])




#GOING OOS
res_tr = resid_t[:l]
resid_t = resid_t[l:]
resid_t_diff = np.diff(resid_t)

#TEST FOR TRADING POTENTIAL ASSESSMENT (c<b)
print('trade')

autocorr_trade = np.corrcoef(resid_t[:-1],resid_t_diff)[0][1]
print(autocorr_trade)

#ALTERNATIVE METRICS
bipolar_trade = bipolar_correlation(resid_t[:-1],resid_t_diff)
print(bipolar_trade)
b,c = alt_corr_1(resid_t[:-1],resid_t_diff,0.5)
print(b)
print(c)
"""

#TRADING
#HOLDING STRATEGY
res_mean = np.mean(res_tr[-500:])
res_sd = np.std(res_tr[-500:])

up = res_mean +  1 * res_sd
down = res_mean -  1 * res_sd

up_clos = res_mean + 0.3 * res_sd
down_clos = res_mean - 0.3 * res_sd

res = pd.DataFrame(resid_t,index = index_test, columns=['COINT'])

d = cointegration_hold(res,res.index,down,up,down_clos,up_clos)



#POSIZIONE

#position to buy/hold the shock (check)
position = np.array([1,-state_means[-1,1]])
side = resid_t + state_means[l:,0]

print(position)
fig, ax = plt.subplots()
ax.plot(side)
plt.show()

needed = prices @ position
print(needed)
"""



#EVERYDAY-STRATEGY
res_mean = np.mean(res_tr[-500:])
res_sd = np.std(res_tr[-500:])

up = res_mean +  0.5 * res_sd
down = res_mean -  0.5 * res_sd

res = pd.DataFrame(resid_t,index = index_test, columns=['COINT'])

d = cointegration_daily(res,res.index,down,up)