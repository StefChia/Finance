import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import yfinance as yf
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix


from framework import compute_prices_paths

data = pd.read_csv('dataset_model2.csv')
data['up'] = np.where(data['Ret']> 0, 1, 0)
print(data)

X = data[['sent','sent1','sent2','sent3']]
y = data['up']


# --- FIT LOGIT ---
Xc = sm.add_constant(X)                 
logit_mod = sm.Logit(y, Xc)
#pen_res = sm.Logit(y, Xc).fit_regularized(alpha=1.0, L1_wt=0.0)  # ridge (set L1_wt=1.0 for lasso)
logit_res = logit_mod.fit(disp=False)   

# --- INFERENCE: p-values, CIs, odds ratios ---
print(logit_res.summary())              # full table with p-values
print("\nOdds ratios:")
print(np.exp(logit_res.params))
print("\n95% CI for odds ratios:")
print(np.exp(logit_res.conf_int()))

# --- PREDICTIONS & METRICS ---
p = logit_res.predict(Xc)               # P(y=1 | X)
yhat = (p >= 0.5).astype(int)

acc = accuracy_score(y, yhat)
prec, rec, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
auc = roc_auc_score(y, p)
cm = confusion_matrix(y, yhat)

print(f"\nAccuracy:  {acc:.3f}")
print(f"Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}  ROC-AUC: {auc:.3f}")
print("Confusion matrix (rows=true, cols=pred):\n", cm)


#SET TRADING STRATEGY

signals = np.zeros_like(p, dtype=int)  # default 0
signals[p > 0.65] = 1
signals[p < 0.35] = -1

values = data['Ret']*signals
transaction_costs = 0.002
values[values!=0] -= transaction_costs 

trading_ret = pd.DataFrame(values, index= data.index,columns=['Ret'])
trades = compute_prices_paths(trading_ret)

fig, ax = plt.subplots()
ax.plot(data.index,trades)
plt.show()



