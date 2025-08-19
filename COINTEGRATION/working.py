
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

from framework import download_historical_prices, compute_returns , threshold_buy_sell, cointegration_hold


tickers = ['ETH-USD','BTC-USD','SOL-USD']

prices = download_historical_prices(tickers,type='Close')
ret = compute_returns(prices)

#print(prices)
#print(ret)

# Features + constant
X = sm.add_constant(prices[['BTC-USD','SOL-USD']])
#X = prices[['BTC-USD','SOL-USD']]
y = prices['ETH-USD']  

# Fit OLS model
model = sm.OLS(y, X).fit()

# Predictions
y_pred = model.predict(X)

# R² (you can also use model.rsquared)
r2 = r2_score(y, y_pred)

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
autocorr = check_autocorr(residuals)
print(autocorr)
plot_res(residuals,prices.index)


residuals_diff = np.diff(residuals)
autocorr_diff = check_autocorr(residuals_diff)
print(autocorr_diff)
plot_res(residuals_diff)





#TEST Augmented-Dickey Fuller
#RECALL: Null hypothesis (H₀): Residuals have a unit root (non-stationary).
adf_result = adfuller(residuals)

# 5. Show results
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])
print("Critical Values:", adf_result[4])





#TRADING

res_mean = np.mean(residuals)
res_sd = np.std(residuals)

up = res_mean + 0.8 * res_sd
down = res_mean - 0.8 * res_sd

up_clos = res_mean + 0.3 * res_sd
down_clos = res_mean - 0.3 * res_sd

res = pd.DataFrame(residuals, index=prices.index, columns=['COINT'])

d = cointegration_hold(res,res.index,down,up,down_clos,up_clos)





"""
#position to buy/hold the shock (check)
position = np.array([1,-model.params[1],-model.params[2]])
side = residuals + model.params[0]

print(position)
fig, ax = plt.subplots()
ax.plot(prices.index,side)
plt.show()

needed = prices @ position
print(needed)
"""