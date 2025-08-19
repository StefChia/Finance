
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from framework import download_historical_prices , threshold_buy_sell

ticker = ['ENAV.MI']

#USE CLOSE, NOT ADJUSTED CLOSE
dl = yf.download(ticker, start="2021-01-01", actions=True, auto_adjust=False)

data = dl[['Close']]
data.columns = [ticker]
#print(data)


"""
data = download_historical_prices(ticker)
print(data)"""

prices = data.copy()
data = threshold_buy_sell(ticker,prices,3.75,4.05,500,)

"""
#simple threshold-based trading simulation, testing a buy–sell rule.

low_thres = 3.8
high_thres = 4.0

budget = 500

flag = 0
buy_price = 0
buy_q = 0
trades = []
pnl = []

for i in range(len(prices)):
    curr_price = prices[i]
    if flag == 0:
        if curr_price < low_thres:
            flag = 1
            buy_price = curr_price
            buy_q = np.floor(budget/curr_price).tolist()
            trades.append(-buy_price)
            pnl.append(0)
        else:
            trades.append(0)
            pnl.append(0)
    else:
        if curr_price > high_thres:
            flag = 0
            sell_price = curr_price
            trades.append(sell_price)
            spread = sell_price - buy_price
            pnl.append(spread * buy_q)
            buy_price = 0
            buy_q = 0
        else:
            trades.append(0)
            pnl.append(0)
            

data['Trades'] = trades
data['PNL'] = pnl
data['CumPNL'] = data['PNL'].cumsum()



# Create the plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# price
ax1.plot(data.index, data[ticker], label='Price')

# robust boolean masks via NumPy
m_buy  = (data['Trades'].to_numpy() < 0)
m_sell = (data['Trades'].to_numpy() > 0)

ax1.scatter(data.index[m_buy],  data.loc[m_buy,  ticker], marker='^', color = 'green', label='Buy',  s=100)
ax1.scatter(data.index[m_sell], data.loc[m_sell, ticker], marker='v', color = 'red', label='Sell', s=100)
ax1.axhline(low_thres, color='green', linestyle='--', alpha=0.5)
ax1.axhline(high_thres, color='red', linestyle='--', alpha=0.5)
ax1.set_ylabel("Price (€)")
ax1.set_title(f"{ticker} Price & Trades")
ax1.legend()

# Cumulative PnL chart
ax2.plot(data.index, data['CumPNL'], color='purple', label='Cumulative PnL')
ax2.set_ylabel("Cumulative PnL (€)")
ax2.set_xlabel("Date")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
    """
    
