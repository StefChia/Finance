
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

#print(df1.head())
#print(len(df1))


#DOWNLOAD RETURNS
# Example: Ethereum Classic in USD
ticker = "SOL-USD"
asset = yf.Ticker(ticker)

# Fetch weekly historical data
df = asset.history(period="5y", interval="1wk")  # last 5 years, weekly
p_v = df[['Close','Volume']]
p_v.index = pd.to_datetime(df.index, format= '%Y-%m-%d')

#p_v.to_csv('returns_dataset_weekly.csv')


returns = compute_log_returns(p_v)
returns.columns = ['Ret','Vol_Ret']

#print(len(returns))
#print(df1.corr())

weekly = returns.copy()
values = df1['Solana'].copy()
values.columns = ['sent']

#1 DAY OF delay between the two (google 1 week ahead)

#weekly = pd.concat([weekly,returns],axis=1)

#print(len(weekly))
#print(len(values))

values = values.iloc[1:]
values = values.iloc[:-1]

#print(weekly)
#print(values)
set_index = weekly.index
values.index = set_index
weekly = pd.concat([weekly,values],axis=1)
weekly.columns = ['Ret','Vol_Ret','sent']
print(weekly)



#Create DATASET FOR MODEL1: SIMPLE h-step-ahead linear regression

h = 1 #step ahead
week = weekly[['sent','Ret','Vol_Ret']]
week['sent'] = week['sent'].shift(h)
week['Vol_Ret'] = week['Vol_Ret'].shift(h)

week = week.dropna()

print(week.corr())

fig, ax = plt.subplots()
ax.plot(week['Vol_Ret'])
plt.show()

week.to_csv('dataset_model1_weekly.csv')



#Create a dataset for MODEL2

h = 1 #step ahead

week = weekly[['sent','Ret']].copy()
week['sent'] = week['sent'].shift(h)
week = week.dropna()

week['sent1']= week['sent']-week['sent'].shift(1)
week = week.dropna()

#flag
week['flag'] = np.where(week['sent1'] > 0, 1, -1)

#FEATURES engineering
week['sent2']= np.where(week['flag']==1, week['sent1']**2,0)
week['sent3']= np.where(week['flag']==-1, week['sent1']**2,0)


print(week.corr())

week.to_csv('dataset_model2_weekly.csv')