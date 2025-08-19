import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
import plotly.express as px


tickers = ['AAPL','META']
data = pd.DataFrame()


"""#Getting yf tickers via API

from yahooquery import search

def get_yf_tickers(names):
    Returns tickers in a list.
    tickers = []
    for name in names:
        try:
            result = search(name)
        except ValueError:
            print(f'Could not find a valid ticker for {name}')
            result = None
        tickers.append(result)
    return tickers"""
        
    
def download_historical_prices(tickers):
    tickers = tickers
    """Download historical prices from YF returns the dataframe of returns"""
    start_date = datetime.datetime(2022,7,31)
    #end_date = datetime.datetime(2025,7,31)
    #period="3mo", interval="1d"
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
        except:
            print(f'{ticker} not found.')
            continue
        else:
            data[ticker] = stock.history(start=start_date)['Close']
    
    data.index = pd.to_datetime(data.index)
    data.index = data.index.tz_localize(None).date
    return data

    
            
def compute_returns(prices):
    """Returns the dataframe of returns."""
    prices = prices
    tickers = prices.columns
    data = pd.DataFrame(index=prices.index)
    
    for ticker in tickers:
        data[ticker]= (prices[ticker]/prices[ticker].shift(1)) -1

    return data.dropna()

def compute_log_returns(prices):
    """Returns the dataframe of returns."""
    prices = prices
    tickers = prices.columns
    data = pd.DataFrame(index=prices.index)
    
    for ticker in tickers:
        data[ticker]= np.log(prices[ticker]/prices[ticker].shift(1))

    return data.dropna()


def download_returns(tickers):
    """Return the returns for the selected tickers."""
    prices = download_historical_prices(tickers)
    return compute_log_returns(prices)
    
def compute_prices_paths(returns,p_0 = 100):
    """Show prices paths."""
    tickers = returns.columns
    data = pd.DataFrame()
    data.index = returns.index
    
    for ticker in tickers:
        p = p_0
        pr = []
        for i in returns[ticker]:
            p *= np.exp(i)
            pr.append(p)
        data[ticker] = pr
    return data




#TRADING STRATEGIES

def trade_up_down(p,ret,transaction_costs=0.0020,up_thresh=0.6,low_thresh=0.4,plot=None):
    """Given an array of probabilities of up, it gives you the trading signals."""
    #SET TRADING STRATEGY

    signals = np.zeros_like(p, dtype=int)  # default 0
    signals[p > up_thresh] = 1
    signals[p < low_thresh] = -1

    values = ret*signals
    #transaction_costs = 0.002
    values[values!=0] -= transaction_costs 

    trading_ret = pd.DataFrame(values, index= data.index,columns=['Ret'])
    prices = compute_prices_paths(trading_ret)

    if plot is not None:
        fig, ax = plt.subplots()
        ax.plot(data.index,prices)
        plt.show()
    return signals,trading_ret,prices






#TRADING



#simple threshold-based trading simulation, testing a buy–sell rule.
def threshold_buy_sell(ticker,prices,low_thresh,high_thresh,budget,tr_cost=0.50, plot=True):
    """Simple threshold-based trading simulation, testing a buy–sell rule.
    Input is prices as dataframe.
    Returns dataframe with prices, trades, PNL, PNLcumsum."""
    
    data = prices.copy()
    prices = prices[ticker].values
    
    flag = 0
    buy_price = 0
    buy_q = 0
    trades = []
    pnl = []

    for i in range(len(prices)):
        curr_price = prices[i]
        if flag == 0:
            if curr_price < low_thresh:
                flag = 1
                buy_price = curr_price
                buy_q = np.floor(budget/curr_price)
                trades.append(- buy_price - tr_cost)
                pnl.append(0)
            else:
                trades.append(0)
                pnl.append(0)
        else:
            if curr_price > high_thresh:
                flag = 0
                sell_price = curr_price
                trades.append(sell_price)
                spread = sell_price - buy_price
                pnl.append(spread * buy_q - tr_cost)
                buy_price = 0
                buy_q = 0
            else:
                trades.append(0)
                pnl.append(0)
                
    
    data['Trades'] = trades
    data['PNL'] = pnl
    data['CumPNL'] = data['PNL'].cumsum()
    
    if plot is True:
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # price
        ax1.plot(data.index, data[ticker], label='Price')

        # robust boolean masks via NumPy
        m_buy  = (np.ravel(np.asarray(data['Trades'])) < 0)
        m_sell = (np.ravel(np.asarray(data['Trades'])) > 0)


        ax1.scatter(data.index[m_buy],  data.loc[m_buy,  ticker], marker='^', color = 'green', label='Buy',  s=100)
        ax1.scatter(data.index[m_sell], data.loc[m_sell, ticker], marker='v', color = 'red', label='Sell', s=100)
        
        
        ax1.axhline(low_thresh, color='green', linestyle='--', alpha=0.5)
        ax1.axhline(high_thresh, color='red', linestyle='--', alpha=0.5)
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
    
    return data