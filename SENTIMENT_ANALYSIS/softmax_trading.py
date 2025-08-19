
import numpy as np
import pandas as pd
#import json
#from pathlib import Path
#from datetime import datetime
import matplotlib.pyplot as plt
#import yfinance as yf
#import statsmodels.api as sm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from framework import trade_up_down


data = pd.read_csv('dataset_model2.csv')
data['up'] = np.where(data['Ret']> 0, 1, 0)
print(data)

X = data[['sent','sent1','sent2','sent3']]
y = data['up']


X_np = np.asarray(X, dtype=np.float32)
y_np = np.asarray(y, dtype=np.int32)

# one-hot for softmax
y_oh = keras.utils.to_categorical(y_np, num_classes=2)

model = keras.Sequential([
    layers.Input(shape=(X_np.shape[1],)),
    # this is just logistic regression expressed as a 2-unit softmax layer
    layers.Dense(2, activation="softmax")
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(X_np, y_oh, epochs=200, batch_size=32, verbose=0)

# predictions & metrics (same spirit as before)
p = model.predict(X_np, verbose=0)[:, 1]          # P(y=1)
yhat = (p >= 0.5).astype(int)

acc = accuracy_score(y_np, yhat)
prec, rec, f1, _ = precision_recall_fscore_support(y_np, yhat, average="binary", zero_division=0)
auc = roc_auc_score(y_np, p)
cm = confusion_matrix(y_np, yhat)

print(f"Accuracy:  {acc:.3f}")
print(f"Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}  ROC-AUC: {auc:.3f}")
print("Confusion matrix (rows=true, cols=pred):\n", cm)

# "Coefficients" analogue:
W, b = model.layers[0].get_weights()     # W shape (n_features, 2), b shape (2,)
# Convert softmax weights to a single log-odds form: beta = w_class1 - w_class0
beta = W[:, 1] - W[:, 0]
beta0 = b[1] - b[0]
print("\nNN-as-logit parameters (derived):")
print("Intercept:", beta0)
print("Coefficients:", beta)



signals, trading_ret, prices = trade_up_down(p,data['Ret'],plot=True)
