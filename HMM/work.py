

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import yfinance as yf
from hmmlearn.hmm import GaussianHMM

from framework import download_returns


#Most traded ETF: "SPY" (SPDR S&P 500 ETF Trust)
#Most traded ETF: "QQQ" (Invesco QQQ Trust)

tickers = ['SPY','QQQ']

ret = download_returns(tickers)

ret_sp500 = ret['SPY'].copy()

states = pd.DataFrame(columns=['SPY','QQQ'])



#SP500 2States

# 2. Prepare features (HMM expects 2D array)
X = ret_sp500.values.reshape(-1, 1)

# 3. Fit Gaussian HMM with 2 hidden states
model = GaussianHMM(n_components=2, covariance_type="full", n_iter=1000, random_state=42)
model.fit(X)


#print(model.means_)       # mean return per state
#print(model.covars_)      # variance per state
A = model.transmat_          # transition matrix P(S_t -> S_{t+1})
pi = model.startprob_        # initial state distribution


# 4. Predict hidden states
hidden_states = model.predict(X)
probs = model.predict_proba(X)

# 5. Append hidden states to DataFrame
data_states = pd.DataFrame(index=ret_sp500.index)
data_states['SPY'] = hidden_states
data_states['ret'] = ret_sp500

probabilities = {}
for i in range(len(probs[0])):
    gg = []
    for j in probs:
        gg.append(j[i])
    probabilities[f'state_{i}'] = gg

data_prob = pd.DataFrame(probabilities,index=ret_sp500.index)



#SP500 3States

# 2. Prepare features (HMM expects 2D array)
X = ret_sp500.values.reshape(-1, 1)

# 3. Fit Gaussian HMM with 2 hidden states
model_3 = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, random_state=42)
model_3.fit(X)

#print(model.means_)       # mean return per state
#print(model.covars_)      # variance per state
#print(data_prob)
# After model.fit(X)
A_3 = model_3.transmat_          # transition matrix P(S_t -> S_{t+1})
pi_3 = model_3.startprob_        # initial state distribution

# 4. Predict hidden states
hidden_states_3 = model_3.predict(X)
probs_3 = model_3.predict_proba(X)

# 5. Append hidden states to DataFrame

data_states_3 = pd.DataFrame(index=ret_sp500.index)
data_states_3['SPY'] = hidden_states_3
data_states_3['ret'] = ret_sp500.iloc

probabilities_3 = {}
for i in range(len(probs_3[0])):
    gg = []
    for j in probs_3:
        gg.append(j[i])
    probabilities_3[f'state_{i}'] = gg

pr_3 = pd.DataFrame(probabilities_3,index=ret_sp500.index)
        
    

#EXPORT
params = {'ret':ret_sp500.tolist(), '2s':{'params':[model.means_.tolist(),model.covars_.tolist()],'t_mat':[A.tolist(),pi.tolist()],'hidden':hidden_states.tolist(),'probs':probs.tolist()},'3s':{'params':[model_3.means_.tolist(),model_3.covars_.tolist()],'t_mat':[A_3.tolist(),pi_3.tolist()],'hidden':hidden_states_3.tolist(),'probs':probs_3.tolist()},'Dates':pd.to_datetime(ret_sp500.index).strftime('%Y-%m-%d').tolist()}
path = Path('/Users/stefanochiapparini/Desktop/PYTHON/Finance/API/HMM/states_sp500.json')
path.write_text(json.dumps(params))

"""#plot
fig, ax = plt.subplots(figsize=(10, 4))

# Plot the return series (use the right column name!)
ax.plot(data_states.index, probabilities[f'state_1'], color = 'red')
ax.plot(data_states_3.index, probabilities_3[f'state_2'], color = 'blue')
ax.fill_between(data_states.index,probabilities[f'state_1'],probabilities_3[f'state_2'],color='lightblue', alpha=0.4)

ax.set_title('Prob_state_1')
ax.set_xlabel('Date')
ax.set_ylabel('Prob_state_1')
ax.legend()
plt.tight_layout()
plt.show()


#plot
fig, ax = plt.subplots(figsize=(10, 4))

# Plot the return series (use the right column name!)
ax.plot(data_states.index, probabilities[f'state_0'], color = 'red')
ax.plot(data_states_3.index, probabilities_3[f'state_0'], color = 'blue')
ax.fill_between(data_states.index,probabilities[f'state_0'],probabilities_3[f'state_0'],color='lightblue', alpha=0.4)

ax.set_title('Prob_state_0')
ax.set_xlabel('Date')
ax.set_ylabel('Prob_state_0')
ax.legend()
plt.tight_layout()
plt.show()



#plot
fig, ax = plt.subplots(figsize=(10, 4))

# Plot the return series (use the right column name!)
ax.plot(data_states.index, probabilities[f'state_0'], color = 'red')
ax.plot(data_states_3.index, probabilities_3[f'state_1'], color = 'blue')
ax.fill_between(data_states.index,probabilities[f'state_0'],probabilities_3[f'state_1'],color='lightblue', alpha=0.4)

ax.set_title('Prob_state_0')
ax.set_xlabel('Date')
ax.set_ylabel('Prob_state_0')
ax.legend()
plt.tight_layout()
plt.show()"""