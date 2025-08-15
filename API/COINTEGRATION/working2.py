
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

from framework import download_historical_prices, compute_returns, cointegration_hold, autocorrelation_at_lags, bipolar_correlation, bipolar_autocorrelation, alt_corr_1


tickers = ['ETH-USD','SOL-USD']

prices = download_historical_prices(tickers,type='Close')
ret = compute_returns(prices)


l = int(len(ret) * 0.75)

#print(prices)
#print(ret)

# Features + constant
X = sm.add_constant(prices[['SOL-USD']])
x_train = X.iloc[:l]
x_test = X.iloc[l:]
#X = prices[['BTC-USD','SOL-USD']]
y = prices['ETH-USD']
y_train = y.iloc[:l]
y_test = y.iloc[l:]  

# Fit OLS model
model = sm.OLS(y_train, x_train).fit()

# Predictions
y_pred = model.predict(x_train)

# R² (you can also use model.rsquared)
r2 = r2_score(y_train, y_pred)

# Summary output (includes coef, std err, p-values, R²)
print(model.summary())

# If you just want the key parts programmatically:
print("\nIntercept and Coefficients:")
print(model.params)

print("\nStandard Errors:")
print(model.bse)

print("\nP-values:")
print(model.pvalues)

print("\nR² score:", r2)
#print("\nPredictions:", y_pred.values)




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


residuals = model.resid
plot_res(residuals,residuals.index)
autocorr = check_autocorr(residuals)
#print(autocorr)

#ALTERNATIVE METRICS
bipolar = bipolar_autocorrelation(residuals)
print(bipolar)
b,c = alt_corr_1(residuals[1:],residuals[:-1])
print(b)
print(c)



residuals_diff = np.diff(residuals)
plot_res(residuals_diff)
autocorr_diff = check_autocorr(residuals_diff)
#print(autocorr_diff)

#ALTERNATIVE METRICS
bipolar_diff = bipolar_autocorrelation(residuals_diff)
print(bipolar_diff)
b,c = alt_corr_1(residuals_diff[1:],residuals_diff[:-1])
print(b)
print(c)


#TEST FOR TRADING POTENTIAL ASSESSMENT (c<b)
print('trade')

autocorr_trade = np.corrcoef(residuals[:-1],residuals_diff)[0][1]
print(autocorr_trade)

#ALTERNATIVE METRICS
bipolar_trade = bipolar_correlation(residuals[:-1],residuals_diff)
print(bipolar_trade)
b,c = alt_corr_1(residuals[:-1],residuals_diff)
print(b)
print(c)





"""

#TEST Augmented-Dickey Fuller
#RECALL: Null hypothesis (H₀): Residuals have a unit root (non-stationary).
adf_result = adfuller(residuals)

# 5. Show results
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])
print("Critical Values:", adf_result[4])





#GOING OOS

# Predictions
y_pred = model.predict(x_test)

# R² (you can also use model.rsquared)
r2 = r2_score(y_test, y_test)
print(f'r2 OOS: {r2}')
residuals_test = y_test - model.predict(x_test)

#TRADING
#HOLDING STRATEGY
res_mean = np.mean(residuals)
res_sd = np.std(residuals)

up = res_mean + 0.8 * res_sd
down = res_mean - 0.8 * res_sd

up_clos = res_mean + 0.3 * res_sd
down_clos = res_mean - 0.3 * res_sd

res = pd.DataFrame(residuals_test, index=residuals_test.index, columns=['COINT'])

d = cointegration_hold(res,res.index,down,up,down_clos,up_clos)

#EVERYDAY-STRATEGY

"""


"""
#position to buy/hold the shock (check)
position = np.array([1,-model.params[1]])
side = residuals + model.params[0]

print(position)
fig, ax = plt.subplots()
ax.plot(prices.index,side)
plt.show()

needed = prices @ position
print(needed)
"""








#Try different lags
#Hand-made grid-search
"""
h = 10
autocorrelation_at_lags(residuals.values,h)
autocorrelation_at_lags(residuals_diff,h)
"""

#Try to fit a ornstein-Uhleinback process on residuals (in discrete time AR process)
"""
ar_model = AutoReg(residuals_diff,lags=1)
res = ar_model.fit()

print(res.summary())
  
# Predictions
preds = res.predict(start=0, end=10)
print(preds)"""




#Try TV-model (Kalman Filter)

import numpy as np
from pykalman import KalmanFilter  # install pykalman

x = np.asarray(X); y = np.asarray(y)
Z = np.c_[np.ones_like(x), x]      # observation design for [alpha_t, beta_t]
n = 2                              # state dim

# Observation y_t = Z_t @ state_t + eps
# State random walk: state_t = state_{t-1} + w_t
kf = KalmanFilter(
    transition_matrices=np.eye(n),
    observation_matrices=Z[:, None, :],  # time-varying design
    transition_covariance=1e-6 * np.eye(n),  # Q (tune; controls drift)
    observation_covariance=1e-3,            # R (tune/estimate)
    initial_state_mean=np.zeros(n),
    initial_state_covariance=1e3 * np.eye(n),
)

state_means, state_covs = kf.filter(y)
alpha_t = state_means[:,0]
beta_t  = state_means[:,1]
resid_t = y - (alpha_t + beta_t * x)        # time-varying spread for signals




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


#TEST FOR TRADING POTENTIAL ASSESSMENT (c<b)
print('trade')

autocorr_trade = np.corrcoef(resid_t[:-1],resid_t_diff)[0][1]
print(autocorr_trade)

#ALTERNATIVE METRICS
bipolar_trade = bipolar_correlation(resid_t[:-1],resid_t_diff)
print(bipolar_trade)
b,c = alt_corr_1(resid_t[:-1],resid_t_diff)
print(b)
print(c)





"""

#TEST Augmented-Dickey Fuller
#RECALL: Null hypothesis (H₀): Residuals have a unit root (non-stationary).
adf_result = adfuller(resid_t)

# 5. Show results
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])
print("Critical Values:", adf_result[4])

"""