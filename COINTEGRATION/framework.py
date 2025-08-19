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
        
    
def download_historical_prices(tickers,type='Adj Close'):
    """Download historical prices from YF returns the dataframe of returns"""
    data = pd.DataFrame()
    start_date = datetime.datetime(2020,7,31)
    #end_date = datetime.datetime(2025,7,31)
    #period="3mo", interval="1d"
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
        except:
            print(f'{ticker} not found.')
            continue
        else:
            data[ticker] = stock.history(start=start_date)[type]
    
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








#TAILORED METRICS

def autocorrelation_at_lags(vector,h=1):
    """Show the autocorrelation at lags up to h.
    It's an hand-made grid-search."""
    for i in range(1,h+1):
        vector = np.copy(vector)
        res = np.copy(vector)
        for j in range(i):
            res = np.insert(res,0,0)[:-1] 
        vector_diff_h = vector-res 
        vector_diff_h = vector_diff_h[1:]
        
        corr = np.corrcoef(vector,res)
        print(f"Lag-{i} autocorrelation: {corr[0][1]}")

        

def bipolar_correlation(series1,series2,thresh=0):
    """Compute bipolar metric. 
    Basically it's a measure of movements from threshold of the same sign."""
    l = len(series1)
    series1 = np.array(series1)
    series2 = np.array(series2)
    
    series1 -= thresh
    series2 -= thresh
    
    series1[series1>0] = 1
    series1[series1<0] = -1
    series1[series1==0] = 0
    
    series2[series2>0] = 1
    series2[series2<0] = -1
    series2[series2==0] = 0
    
    bipolar = (series1 @ series2)/(l)
    #print(f'Bipolar metric: {bipolar}')
    return bipolar

def bipolar_autocorrelation(series,h=1,thresh=0):
    """Compute bipolar autocorrelation at lag h."""
    value = bipolar_correlation(series[h:],series[:-h],thresh)
    return value






def alt_corr_1(series1,series2,cap=2):
    """In short is an autocorrelation decomposition in middies and extreme values.
    A sign-concordance metric but up-weighting extremes via a cos **2 transform of each series’ z-score strandardization."""
    
    series1 = np.array(series1)
    series2 = np.array(series2)
    
    m1 = np.mean(series1)
    m2 = np.mean(series2)
    
    s1 = np.std(series1)
    s2 = np.std(series2)
    
    
    #SCALE in pi
    #weights for middle values
    vet1 = np.cos((np.clip((series1-m1)/s1,-cap,+cap)/(2*cap)) * np.pi)**2
    vet2 = np.cos((np.clip((series2-m2)/s2,-cap,+cap)/(2*cap)) * np.pi)**2
    #weights for extreme values
    vet1_2 = np.sin((np.clip((series1-m1)/s1,-cap,+cap)/(2*cap)) * np.pi)**2
    vet2_2 = np.sin((np.clip((series2-m2)/s2,-cap,+cap)/(2*cap)) * np.pi)**2
    
    series1[series1>0] = 1
    series1[series1<0] = -1
    series1[series1==0] = 0
    
    series2[series2>0] = 1
    series2[series2<0] = -1
    series2[series2==0] = 0
    
    v1 = vet1 * series1 
    v2 = vet2 * series2
    
    v1_2 = vet1_2 * series1 
    v2_2 = vet2_2 * series2 
    
    bipolar_mid = v1 @ v2 / (np.abs(v1) @ np.abs(v2)+1e-12)
    bipolar_ex = v1_2 @ v2_2 / (np.abs(v1_2) @ np.abs(v2_2)+1e-12)
    #print(f'Bipolar metric: {bipolar}')
    return bipolar_mid, bipolar_ex


















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
def threshold_buy_sell(ticker,prices,low_thresh,high_thresh,budget=None,tr_cost=0.50, plot=True):
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
                if budget is not None:
                    buy_q = np.floor(budget/curr_price)
                else:
                    buy_q = 1
                trades.append(- buy_price - tr_cost)
                pnl.append(0)
            else:
                trades.append(0)
                pnl.append(0)
        else:
            if curr_price > high_thresh:
                flag = 0
                sell_price = curr_price
                trades.append(sell_price - tr_cost)
                spread = sell_price - buy_price
                pnl.append(spread * buy_q - 2 * tr_cost)
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






def cointegration_hold(residuals,index,low_thresh,high_thresh,low_clos=None,high_clos=None,tr_cost=0.50, plot=True):
    """Simple threshold-based trading simulation, testing a buy–sell rule.
    Input is prices as dataframe.
    Returns dataframe with prices, trades, PNL, PNLcumsum."""
    
    ticker = 'COINT'
    data = pd.DataFrame(residuals, index=index, columns=[ticker])
    residuals = residuals.values
    
    if low_clos is None:
        low_clos = low_thresh
    if high_clos is None:
        high_clos = high_thresh
    
    flag = 0
    charged_price = 0
    q = 1
    in_out = []
    trades = []
    pnl = []

    for i in range(len(residuals)):
        curr_price = residuals[i]
        
        if flag == 0:
            if curr_price < low_thresh:
                flag = 1
                charged_price = curr_price
                """if budget is not None:
                    buy_q = np.floor(budget/curr_price)
                else:
                    buy_q = 1"""
                trades.append( charged_price - tr_cost)
                in_out.append(1)
                pnl.append(0)
            elif curr_price > high_thresh:
                flag = -1
                charged_price = curr_price
                """if budget is not None:
                    buy_q = np.floor(budget/curr_price)
                else:
                    buy_q = 1"""
                trades.append( charged_price - tr_cost)
                in_out.append(1)
                pnl.append(0)
            else:
                trades.append(0)
                in_out.append(0)
                pnl.append(0)
            
        elif flag == 1:
            if curr_price > high_clos:
                sell_price = curr_price
                trades.append(sell_price - tr_cost)
                spread = sell_price - charged_price
                pnl.append(spread * q - 2 * tr_cost)
                charged_price = 0
                q = 1
                in_out.append(-1)
                flag = 0
            else:
                trades.append(0)
                pnl.append(0)
                in_out.append(0)
        
        else:
            if curr_price < low_clos:
                buy_price = curr_price
                trades.append(buy_price - tr_cost)
                spread = - buy_price + charged_price
                pnl.append(spread * q - 2 * tr_cost)
                charged_price = 0
                q = 1
                in_out.append(-1)
                flag = 0
            else:
                trades.append(0)
                pnl.append(0)
                in_out.append(0)
        
                
    data['In-Out'] = in_out
    data['Trades'] = trades
    data['PNL'] = pnl
    data['CumPNL'] = data['PNL'].cumsum()
    
    if plot is True:
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # price
        ax1.plot(data.index, residuals, label='Residuals')

        # robust boolean masks via NumPy
        m_in  = (np.ravel(np.asarray(data['In-Out'])) == 1)
        m_out = (np.ravel(np.asarray(data['In-Out'])) == -1)


        ax1.scatter(data.index[m_in],  data.loc[m_in,  ticker], marker='^', color = 'orange', label='In',  s=100)
        ax1.scatter(data.index[m_out], data.loc[m_out, ticker], marker='v', color = 'blue', label='Out', s=100)
        
        

        ax1.axhline(low_thresh, color='grey', linestyle='--', alpha=0.5)
        ax1.axhline(high_thresh, color='grey', linestyle='--', alpha=0.5)
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






def cointegration_daily(residuals,index,low_thresh,high_thresh,tr_cost=0.50, plot=True):
    """Simple threshold-based daily trading simulation. Daily trade the noise.
    Returns dataframe with prices, trades, PNL, PNLcumsum."""
    
    ticker = 'COINT'
    data = pd.DataFrame(residuals, index=index, columns=[ticker])
    residuals = np.asarray(residuals, dtype=float).ravel()
    res_diff = np.diff(residuals, prepend=0.0)  #one value shorter thus preprond parameter
    
    
    in_out = [] 
    for i in range(len(residuals)):
        curr_e = residuals[i]
        sign = - np.sign(curr_e)
        if curr_e < high_thresh and curr_e > low_thresh:
            sign = 0
        in_out.append(sign)
        
    in_out = np.array(in_out)
    in_out = np.roll(in_out, 1)
    in_out[0] = 0
    
    trades = in_out * residuals
    trades[trades!=0]-= tr_cost
    pnl = res_diff * in_out
    pnl[pnl!=0]-= 2 * tr_cost
    check_pnl = np.sign(pnl)
    
    data['In-Out'] = in_out
    data['Trades'] = trades
    data['PNL'] = pnl
    data['CumPNL'] = data['PNL'].cumsum()
    data['ProfTrade'] = check_pnl
    
    
    if plot is True:
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # price
        ax1.plot(data.index, residuals, label='Residuals')

        # robust boolean masks via NumPy
        m_sell  = (np.ravel(np.asarray(data['ProfTrade'])) == 1)
        m_buy = (np.ravel(np.asarray(data['ProfTrade'])) == -1)


        ax1.scatter(data.index[m_sell],  data.loc[m_sell,  ticker], marker='^', color = 'green', label='Profit trades',  s=40)
        ax1.scatter(data.index[m_buy], data.loc[m_buy, ticker], marker='v', color = 'red', label='Loss trades', s=40)
        
        

        ax1.axhline(low_thresh, color='grey', linestyle='--', alpha=0.5)
        ax1.axhline(high_thresh, color='grey', linestyle='--', alpha=0.5)
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