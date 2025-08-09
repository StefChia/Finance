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