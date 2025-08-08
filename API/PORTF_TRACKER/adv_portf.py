
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



def compute_sample_statistics(returns,prices):
    """Compute sample mean, sample variance and sd, correlation matrix and drawdowns.
    Returns a dictionary."""
    returns = returns
    tickers = returns.columns
    basic_stats = {}
    basic_stats_annaulized = {}
    corr_matrix = returns.corr()
    for ticker in tickers:
        basic_stats[ticker]= {}
        basic_stats[ticker]['mean']= returns[ticker].mean()
        basic_stats[ticker]['sd']= returns[ticker].std()
        basic_stats[ticker]['variance']= np.square(returns[ticker].std())
        
        basic_stats_annaulized[ticker] = {}
        basic_stats_annaulized[ticker]['mean'] = returns[ticker].mean() * 250
        basic_stats_annaulized[ticker]['sd'] = returns[ticker].std() * np.sqrt(250)
        basic_stats_annaulized[ticker]['variance'] = np.square(returns[ticker].std()) * 250
        basic_stats_annaulized[ticker]['Sharpe'] = basic_stats_annaulized[ticker]['mean']/basic_stats_annaulized[ticker]['sd']
        
        #ADD Other custom metrics e.g. downside standard deviation
        basic_stats_annaulized[ticker]['Drawdown'] = compute_drawdown_minimal(prices,ticker)[0]
        basic_stats_annaulized[ticker]['Drawdown Recovery'] = compute_drawdown_minimal(prices,ticker)[2]
    
    #print(basic_stats)
    #print(basic_stats_annaulized)
    return basic_stats,basic_stats_annaulized,corr_matrix
    
    
            
            
def find_recovery_index(list,value):
    """Find the first time the price recover, if it does, and return the index and the recovery status"""
    for j, v in enumerate(list):
        if v >= value:
            return j, 1
    return None, 0
    
 
    


def compute_drawdown_minimal(prices: pd.DataFrame, ticker: str):
    """Drop-in replacement mirroring your return shape: [glob_max, index_pair, dat, rec_gl].

    - glob_max is positive fraction (e.g., 0.35)
    - index_pair are integer positions [peak_i, trough_i]
    - dat are the corresponding dates
    - rec_gl is 1 if recovered, 0 otherwise
    """
    dates = list(prices.index)
    series = list(map(float, prices[ticker].astype(float).tolist()))

    if len(series) < 2:
        return [0.0, [0, 0], dates[:1] * 2, 0]

    glob_max = 0.0
    index_pair = [0, 0]
    dat: list = []
    rec_gl = 0

    n = len(series)
    for i in range(n - 1):  # last point can't start a drawdown window
        curr_p = series[i]
        tail = series[i + 1 :]

        idx_up, rec = find_recovery_index(tail, curr_p)
        # Window to search for trough
        if idx_up is not None:
            window = series[i + 1 : i + idx_up + 1]
        else:
            window = series[i + 1 :]

        if not window:
            continue  # nothing to compare

        curr_min = min(window)
        curr_abs = 1 - curr_min / curr_p

        if curr_abs > glob_max:
            glob_max = curr_abs
            trough_offset = (idx_up if idx_up is not None else window.index(curr_min))
            trough_i = i + 1 + trough_offset
            index_pair = [i, trough_i]
            dat = [dates[i], dates[trough_i]]
            rec_gl = rec

    return [glob_max, index_pair, dat, rec_gl]
     
    
def show_df_html(df):
    """Show the dataframe in html format"""
    data = df.copy()
    data.insert(0, data.index.name or "Index", data.index)

    fig = go.Figure(
        data=[go.Table(
            columnorder=list(range(1, len(data.columns) + 1)),
            columnwidth=[
                max(80, min(220, len(str(col)) * 10))
                for col in data.columns
            ],
            header=dict(
                values=list(data.columns),
                align="left",
                fill_color="#C8D4E3",
                font=dict(size=12, color="#2a3f5f")
            ),
            cells=dict(
                values=[data[c].tolist() for c in data.columns],
                align="left",
                fill_color="#EBF0F8",
            )
        )]
    )

    fig.update_layout(
        title=('Annualized summery statistics.'),
        margin=dict(l=10, r=10, t=60, b=10)
    )

    fig.show()


def show_corr_matrix(corr_matrix):
    """Show correlation matrix via plotly."""
    corr_matrix = corr_matrix
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale="RdBu",
        zmin=-1, zmax=1,
        title="Correlation Matrix"
    )
    fig.show()
    


def compute_portf_returns(returns, weights = None):
    """Follow the portfolio weights path."""
    data = pd.DataFrame()
    data.index = returns.index
    
    
    N = len(data.index)
    M = len(returns.columns)
    
    if weights == None:
        w = np.array([1/M for i in range(M)])
    else:
        w = np.array(weights)
    
    data['Portf']= returns @ w
    
    return data


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
        
        
def show_prices_paths(prices):
    """Show the price path"""
    dates = prices.index
    tickers = prices.columns
   
    
    np.random.seed = 42
    
    fig, ax = plt.subplots()
    for ticker in tickers:
        #Random colors
        col = tuple([round(x, 1) for x in np.random.uniform(0, 1, 3)])
        ax.plot(dates, prices[ticker],label=ticker, color =col)
    ax.legend(title="Tickers", loc="best", frameon=True)
    
    
    ax.set_title('Price paths.')
    ax.set_xlabel('Dates')
    ax.set_ylabel('Prices')
    
    plt.show()

        
         
        
    
    
    
#DOWNLOAD THE DATA
prices = download_historical_prices(tickers=tickers)
#show_prices_paths(prices)
returns = compute_returns(prices=prices)
#print(returns)
#prices_from100 = compute_prices_paths(returns)
#show_prices_paths(prices_from100)
#glob_max = compute_drawdown_minimal(prices,'AAPL')
#print(glob_max)



"""
INDIVIDUAL ASSETS
basic_stats,basic_stats_annaulized,corr_matrix =compute_sample_statistics(returns,prices)
basic_stats_df = pd.DataFrame(basic_stats)
basic_stats_annaulized_df = pd.DataFrame(basic_stats_annaulized)
print(basic_stats_annaulized_df)
show_df_html(basic_stats_annaulized_df)
show_corr_matrix(corr_matrix)
"""
#COMPUTE PORTFOLIO RETURNS
port_returns = compute_portf_returns(returns)
port_prices = compute_prices_paths(port_returns)
show_prices_paths(port_prices)

#ASSESS PORTFOLIO RETURNS
port_basic_stats,port_basic_stats_annaulized,port_corr_matrix =compute_sample_statistics(port_returns,port_prices)
port_basic_stats_df = pd.DataFrame(port_basic_stats)
port_basic_stats_annaulized_df = pd.DataFrame(port_basic_stats_annaulized)
#print(port_basic_stats_annaulized_df)
#show_df_html(port_basic_stats_annaulized_df)