
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import yfinance as yf

from framework import compute_log_returns

path = Path('/Users/stefanochiapparini/Desktop/PYTHON/Finance/API/SENTIMENT_ANALYSIS/google_searches_dataset.json')
data = json.loads(path.read_text())
dfs = []
for i in range(4):
    df = pd.DataFrame(data[i])
    if i < 2:
        f = '%Y-%m-%d'
    else:
        f = '%Y-%m-%d %H:%M'    
    df.index = pd.to_datetime(df.index, format= f)
    df = df.drop(columns=['isPartial'])
    dfs.append(df)
df1,df2,df3,df4 =dfs[0],dfs[1],dfs[2],dfs[3]
means = pd.DataFrame({'5y':df1.mean(),'3m':df2.mean(),'7d':df3.mean(),'1d':df4.mean()})



"""#DOWNLOAD RETURNS
# Example: Ethereum Classic in USD
ticker = "ETH-USD"
asset = yf.Ticker(ticker)

# Fetch weekly historical data
df = asset.history(period="3mo", interval="1d")  # last 1 year, weekly
p_v = df[['Close','Volume']]
df.index = pd.to_datetime(df.index, format= '%Y-%m-%d')"""


#UPLOAD RETURNS
df = pd.read_csv('pv_dataset.csv')
p_v = df[['Close','Volume']]
p_v.index = df['Date']


returns = compute_log_returns(p_v)
returns.columns = ['Ret','Vol_Ret']

daily = p_v.copy()
values = df2['Ethereum'].copy()
daily['sent']= values.iloc[1:].values
daily = pd.concat([daily,returns],axis=1)
#print(daily.corr())


#Create DATASET FOR MODEL1: SIMPLE h-step-ahead linear regression

h = 1 #step ahead
day = daily[['sent','Ret','Vol_Ret']]
day['sent'] = day['sent'].shift(h)
day['Vol_Ret'] = day['Vol_Ret'].shift(h)

day = day.dropna()

print(day.corr())

fig, ax = plt.subplots()
ax.plot(day['Vol_Ret'])
plt.show()

day.to_csv('dataset_model1.csv')



#Create a dataset for MODEL2

h = 1 #step ahead

day = daily[['sent','Ret']].copy()
day['sent'] = day['sent'].shift(h)
day = day.dropna()

day['sent1']= day['sent']-day['sent'].shift(1)
day = day.dropna()

#flag
day['flag'] = np.where(day['sent1'] > 0, 1, -1)

#FEATURES engineering
day['sent2']= np.where(day['flag']==1, day['sent1']**2,0)
day['sent3']= np.where(day['flag']==-1, day['sent1']**2,0)

day.to_csv('dataset_model2.csv')

