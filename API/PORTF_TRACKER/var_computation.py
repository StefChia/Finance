

import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
import plotly.express as px


from scipy.stats import norm


tickers = ['SPY','QQQ']
data = pd.DataFrame()
        
    
def download_historical_prices(tickers):
    tickers = tickers
    """Download historical prices from YF returns the dataframe of returns"""
    start_date = datetime.datetime(2015,7,31)
    #end_date = datetime.datetime(2025,7,31)
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


    
#DOWNLOAD THE DATA
prices = download_historical_prices(tickers=tickers)
#show_prices_paths(prices)
returns = compute_returns(prices=prices)
#print(returns)






 
#COMPUTING VAR

np.random.seed(42)


#COMPUTE EWMA

def compute_ewma(returns):
    """Compute the covariance prediction for the assets in the Dataframe.
    Output is a np.ndarray"""
    VARCOV = []
    names = returns.columns
    for n in names:
        row = []
        ret1 = returns[n]
        for m in names:
            ret2 = returns[m]
            value = comp_single_ewma(ret1,ret2)
            row.append(value)
        VARCOV.append(row)
    return np.array(VARCOV)
    
            
def comp_single_ewma(return1,return2,smooth = 0.97):
    """Compute one step ahead forecast of cov for two returns TS."""
    return1 = return1.values
    return2 = return2.values
    
    cov_for = 0
    for i in range(len(return1)):
        new = return1[i]*return2[i]
        cov_for = smooth * cov_for + (1-smooth) * new
    return cov_for
        
V = compute_ewma(returns)
print(V)






#Compute VAR via MONTECARLO SIMULATION

h = 22      #Monthly VAR
thresh = 0.95

#set value in days magnitude
w = np.array([0.5,0.5])
m = np.array([0.0,0.0])
l = V.shape[0]

#Scale for horizon
m *= h
V *= h

#CHOLESKY DECOMPOSITION
L = np.linalg.cholesky(V)

#MONTECARLO
N = 10_000

def sampling_1(l,L):
    """Sample one vector from gaussian, covariance is given via cholesky."""
    ret = []
    for  i in range(l):
        ret_i = np.random.standard_normal()
        ret.append(ret_i)
    ret = np.array(ret)
    #Give back covariance and mean
    ret = L @ ret + m
    return ret

def show_portf_ret_distr(portf_rets,var,thresh=0.95,var_is_positive_loss=True):
    "Plot the portf-ret distribution"
    xline = -var if var_is_positive_loss else var

    fig, ax = plt.subplots()
    ax.hist(portf_rets, bins=100)
    ax.axvline(xline, linestyle='--', linewidth=2,
               label=f"{int(thresh*100)}% VaR cutoff = {xline:.4f}")

    ax.set_title("Portfolio Return Distribution")
    ax.set_xlabel("Return")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.show()
    
port_ret = []
for i in range(N):
    r = sampling_1(l,L)
    ret = w @ r
    port_ret.append(ret)
port_ret = np.array(port_ret)

VAR_MC = - np.percentile(port_ret,(1-thresh)*100)
#print(VAR_MC)

#show_portf_ret_distr(port_ret,VAR_MC)


    



#GET VAR PARAMETRIC
h = 22 #monthly
thresh = 0.95

w = np.array([0.5,0.5])
m = np.array([0.0,0.0])
l = len(returns.columns)

ret_m = w @ (m * h)
ret_v = w @ (V * h) @ w.T
ret_sd = np.sqrt(ret_v)

k = norm.ppf(0.05, loc=0, scale=1)
VAR_param = - ((k * ret_sd)+ret_m)

print(VAR_param)





#COMPUTE VAR via Historical distribution

thresh = 0.95
h = 22

port_ret = (returns * np.sqrt(h)) @ w

#OR
#port_ret_h = (1 + port_ret).rolling(h).apply(np.prod, raw=True) - 1     # for simple returns
#port_ret_h = port_ret.rolling(h).sum()         for log returns
VAR_hist = - np.percentile (port_ret,(1-thresh)*100)    

print(VAR_hist)


