import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm


data = pd.read_csv('dataset.csv')

# Features + constant
X = sm.add_constant(data[['sent', 'Vol_Ret']] )
y = data['Ret']  

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