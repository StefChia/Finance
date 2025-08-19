

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from framework import download_returns


#Most traded ETF: "SPY" (SPDR S&P 500 ETF Trust)
#Most traded ETF: "QQQ" (Invesco QQQ Trust)

tickers = ['SPY','QQQ']

ret = download_returns(tickers)

ret_sp500 = ret['SPY'].copy()
X = ret['QQQ'].copy()

states = pd.DataFrame(columns=['SPY','QQQ'])


#IMPLEMENT MARKOV SWITCHING MODEL

# y: target (e.g., returns), X: features (DataFrame or 2D array)
# Include a constant in X if you want an intercept per regime.
y = ret_sp500
X = sm.add_constant(X.values)  # intercept term

# Two-regime switching regression:
# - switching_exog=True  -> coefficients differ by regime
# - switching_variance=True -> variance differs by regime
mod = MarkovRegression(
    endog=y,
    k_regimes=2,
    exog=X,
    switching_exog=True,
    switching_variance=True
)

res = mod.fit(disp=False)

print(res.summary())

# Smoothed probabilities P(S_t = k | y_{1:T})
prob_k0 = res.smoothed_marginal_probabilities[0]  # regime 0
prob_k1 = res.smoothed_marginal_probabilities[1]  # regime 1

# In-sample fitted values and residuals
fitted = res.fittedvalues
resid  = y - fitted

# Predicted regime (argmax of smoothed probabilities)
pred_regime = prob_k0.lt(prob_k1).astype(int)  # 0/1

# If you want the (time-t) transition matrix estimate:
# statsmodels parameterizes transitions; use:
#Tm = res.transition_matrix  # (T, k, k) or (k, k) depending on specification



# 1) Filtered (uses data up to t)
probs_filt_df = res.filtered_marginal_probabilities   # DataFrame (T × k)

# 2) One-step-ahead predicted (uses data up to t-1)
probs_pred_df = res.predicted_marginal_probabilities  # DataFrame (T × k)

# 3) Smoothed (uses full sample y_{1:T}) – NOT what you want
probs_smooth_df = res.smoothed_marginal_probabilities


#Convert to numpy
probs_filt = probs_filt_df.values   # shape (T, k)
probs_smoo = probs_smooth_df.values


#GET THE STATES
#filtered_state = probs_filt_df.idxmax(axis=1)     # labels 0..k-1 (as Series)
# or numeric:
filtered_state_num = probs_filt.argmax(axis=1)    # NumPy array of ints


#print(probs_filt)




#plot

filt = {}
for i in range(len(probs_filt[0])):
    gg = []
    for j in probs_filt:
        gg.append(j[i])
    filt[f'state_{i}'] = gg

filt = pd.DataFrame(filt,index=ret.index)


smoo = {}
for i in range(len(probs_smoo[0])):
    gg = []
    for j in probs_smoo:
        gg.append(j[i])
    smoo[f'state_{i}'] = gg

smoo = pd.DataFrame(smoo,index=ret.index)


fig, ax = plt.subplots(figsize=(10, 4))

# Plot the return series (use the right column name!)
ax.plot(ret.index, filt[f'state_1'],color = 'red',label='filtered')
ax.plot(ret.index, smoo[f'state_1'], color = 'blue',label ='smoothed')
#ax.fill_between(ret.index,filt[f'state_0'],filt[f'state_1'],color='lightblue', alpha=0.4)

ax.set_title('Prob_state_1')
ax.set_xlabel('Date')
ax.set_ylabel('Prob_state_1')
ax.legend()
plt.tight_layout()
plt.show()