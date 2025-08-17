import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
import plotly.express as px
import cvxpy as cp


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
    
def compute_prices_paths(returns,log_ret=True, p_0 = 100):
    """Show prices paths.
    Returns a dataframe."""
    tickers = returns.columns
    data = pd.DataFrame()
    data.index = returns.index
    
    for ticker in tickers:
        p = p_0
        pr = []
        for i in returns[ticker]:
            if log_ret==True:
                p *= np.exp(i)
            else:
                p *= (1 + i)
            pr.append(p)
        data[ticker] = pr
    return data


def compute_drawdowns(prices, plot=False ):
    """Compute the Time Series of drawdowns given prices.
    Returns the TS of drawdowns and a list with the value of maximum drawdowns and the indexes to recover the window of that.
    [peak_index, max_draw_index, recovery_index if there is]"""
    
    running_peak = np.maximum.accumulate(prices)
    drs = prices / running_peak - 1.0  # <= 0
    
    trough_idx = int(np.argmin(drs))
    max_drawdown = float(drs[trough_idx])
    
    peak_value = running_peak[trough_idx]
    peak_idx = int(np.argmax(prices[:trough_idx + 1] == peak_value))
    
    recovery_idx = None
    if trough_idx + 1 < len(prices):
        rec_candidates = np.where(prices[trough_idx + 1:] >= peak_value)[0]
        if rec_candidates.size > 0:
            recovery_idx = int(trough_idx + 1 + rec_candidates[0])
        

    def plot_drawdown_with_markers(drs, peak_idx, trough_idx, recovery_idx=None, dates=None, title="Drawdown"):
        """
        drs: drawdown series (np.ndarray, pd.Series, or 1-col pd.DataFrame)
        peak_idx, trough_idx, recovery_idx: positions (ints) or labels (e.g., datetimes)
        dates: optional index to use when drs is an ndarray (must match len(drs))
        """
        try:
            import pandas as pd
        except Exception:
            pd = None

        # ---- normalize y and x (index) ----
        if pd is not None and isinstance(drs, pd.DataFrame):
            if drs.shape[1] != 1:
                raise ValueError("drs DataFrame must have exactly one column")
            y = drs.iloc[:, 0].to_numpy()
            x = drs.index
        elif pd is not None and isinstance(drs, pd.Series):
            y = drs.to_numpy()
            x = drs.index
        else:
            y = np.asarray(drs).reshape(-1)
            if dates is not None and pd is not None and not isinstance(dates, np.ndarray):
                x = pd.Index(dates)
            else:
                x = np.asarray(dates) if dates is not None else np.arange(len(y))

        # ---- helper: convert a position/label to (x,y) ----
        def get_point(idx_like):
            if idx_like is None:
                return None
            # positional int
            if isinstance(idx_like, (int, np.integer)):
                i = int(idx_like)
                if 0 <= i < len(y):
                    return (x[i], y[i])
                return None
            # label -> locate in index (only if pandas index-like)
            try:
                i = x.get_loc(idx_like)  # works for pandas Index
                if isinstance(i, slice):
                    i = i.start
                elif hasattr(i, "__len__"):
                    i = int(np.asarray(i)[0])
                return (x[i], y[i])
            except Exception:
                return None

        # ---- plot ----
        fig, ax = plt.subplots()
        ax.plot(x, y, label="Drawdown")

        p = get_point(peak_idx)
        t = get_point(trough_idx)
        r = get_point(recovery_idx)

        if p is not None:
            ax.scatter([p[0]], [p[1]], color="red", marker="o", zorder=3, label="Peak")
        if t is not None:
            # avoid duplicate legend label if peak & trough coincide
            ax.scatter([t[0]], [t[1]], color="red", marker="o", zorder=3,
                    label=None if p is not None else "Max DD")
        if r is not None:
            ax.scatter([r[0]], [r[1]], color="green", marker="o", zorder=3, label="Recovery")

        ax.set_title(title)
        ax.set_ylabel("Drawdown")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        plt.show()

    if plot:
        is_df = isinstance(prices, pd.DataFrame)
        if is_df:
            plot_drawdown_with_markers(drs, peak_idx, trough_idx, recovery_idx, dates=prices.index)
        else:
            plot_drawdown_with_markers(drs, peak_idx, trough_idx, recovery_idx)
    
    return drs, [max_drawdown,peak_idx,trough_idx,recovery_idx]
        

    


def compute_sample_statistics(prices,returns,corr_matr=False):
        """Compute sample mean, sample variance and sd, correlation matrix and drawdowns.
        Input is a dataframe.
        Returns a dictionary."""
        #prices = self.prices
        #returns = self.returns
        tickers = prices.columns
        #tickers = self.tickers
        basic_stats = {}
        basic_stats_annualized = {}
        corr_matrix = returns.corr()
        
        for ticker in tickers:
            basic_stats[ticker]= {}
            basic_stats[ticker]['mean']= returns[ticker].mean()
            basic_stats[ticker]['sd']= returns[ticker].std()
            basic_stats[ticker]['variance']= np.square(returns[ticker].std())
            
            basic_stats_annualized[ticker] = {}
            basic_stats_annualized[ticker]['mean'] = returns[ticker].mean() * 252
            basic_stats_annualized[ticker]['sd'] = returns[ticker].std() * np.sqrt(252)
            basic_stats_annualized[ticker]['variance'] = np.square(returns[ticker].std()) * 252
            basic_stats_annualized[ticker]['Sharpe'] = basic_stats_annualized[ticker]['mean']/basic_stats_annualized[ticker]['sd']
            
            #ADD Other custom metrics e.g. downside standard deviation
            #basic_stats_annualized[ticker]['Drawdowns'] = compute_drawdowns(prices[ticker])[0]
            basic_stats_annualized[ticker]['Maximum drawdown'] = compute_drawdowns(prices[ticker])[1][0]
            a,b,c = compute_drawdowns(prices[ticker])[1][1],compute_drawdowns(prices[ticker])[1][2],compute_drawdowns(prices[ticker])[1][3]
            if c is not None:
                recovery = prices.index[c]
            else:
                recovery = c
            basic_stats_annualized[ticker]['Recovery'] = [prices.index[a],prices.index[b],recovery]
        
        #print(basic_stats)
        #print(basic_stats_annualized)
        if corr_matr is False:
            return basic_stats,basic_stats_annualized
        else:
            return basic_stats,basic_stats_annualized,corr_matrix


def print_summary_statistics(dictionary):
    """Print the from the dictionary output of compute_sample_statistics function."""
    tickers = list(dictionary.keys())
    for ticker in tickers:
        di = dictionary[ticker]
        print(f"\nSummary for {ticker}:")
        for key, value in di.items():
            print(f'{key}: {value}')














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



def update_ewma_step(varcov_0,ret_1,lamda=0.97):
        """Update step of ewma."""
        varcov_1 = []
        l = len(ret_1)
        new_obs = []
        for i in range(l):
            row = []
            for j in range(l):
                value_0 = ret_1.iloc[i]*ret_1.iloc[j]
                row.append(value_0)
            new_obs.append(row)
            
        new_obs = np.array(new_obs)
        varcov_1 = varcov_0 * lamda + new_obs * (1 - lamda)
        
        return np.array(varcov_1)
    
    
def update_ewma_in_test_sample(varcov_0,test_sample):
    """Output is a list of the varcov prediction for all the test horizon.
    It start with the varcov_0 given from train sample."""
    varcov_test = [varcov_0]
    for i in range(len(test_sample)):
        varcov = varcov_test[-1]
        ret = test_sample.iloc[i]

        varcov_updated = update_ewma_step(varcov,ret)
        varcov_test.append(varcov_updated)
    
    return varcov_test

   
                
def launch_portf_opt_on_sample_test(exp_ret,varcov_for,returns):
    """Input: list of exp_ret and a list of varcov_forecast for each step.
    Returns the portfolio returns under step by step optimization."""
    
    portf_ret = []
    for i in range(len(returns)):
        m = exp_ret[i]
        varcov = varcov_for[i]
        ret = returns.iloc[i]
        
        res = portf_optim(m,varcov)
        w = res['weights']
        portf_return = ret @ w
        portf_ret.append(portf_return)
        
    return np.array(portf_ret)
        
        








#PORTFOLIO OPTIMIZATION


def portf_optim(ex_values,varcov,type='Both',value=1,long_only=True, integer=False, thres_weig=0.15, solver="OSQP", verbose=False):
    """Given expected value and varcov.
    Returns a vector of weights.
    Type-target pair can be:
    - 'Both' and number for risk aversion
    - 'Target_m' exp ret and threshold
    - 'Target_v' variance and threshold"""
    
    ex_values = np.asarray(ex_values, dtype=float)
    varcov = np.asarray(varcov, dtype=float)
    
    n = len(ex_values)
    
    #Check for PSD-veness of varcov
    def psd_jitter(S, eps=1e-8):
        S = np.asarray(S, dtype=float)
        S = 0.5 * (S + S.T)                     # symmetrize first
        w, V = np.linalg.eigh(S)                # real symmetric eigendecomp
        w = np.maximum(w, eps)                  # clip small/neg eigenvalues
        S_psd = V @ np.diag(w) @ V.T
        S_psd = 0.5 * (S_psd + S_psd.T)         # enforce symmetry again
        return S_psd
    
    varcov = psd_jitter(varcov)
    assert np.allclose(varcov, varcov.T, atol=1e-12)
    
    
    #MODELING THE OPTIMIZATION PROBLEM
    
    #SET PARAMETERS
    m = cp.Parameter((n,1), value=ex_values, name="ex_returns")
    S = cp.Parameter((n, n), PSD=True, name="varcov")   
    S.value = varcov        #THIS POTENTIALLY ALLOWS TO CHANGE VALUES LATER
    
    
    #SET OBJECTIVE/DECISION VARIABLES
    # Choose domain: nonneg=True, integer=True,
    w = cp.Variable(n, name='weights', nonneg=long_only, integer=integer)
    
    #SET CONSTRAINTS
    cons = []
    cons += [cp.sum(w) == 1]
    cons += [w <= thres_weig]   #boundaries on weights
    #cons += [w >= 0] 
    
    
    if type == 'Both':
        #SET OBJECTIVE FUNCTION
        r = cp.Parameter(value= 1/value, name="risk_aversion")
        obj = cp.Minimize(- r * w @ m + cp.quad_form(w,S) )                 
    
    if type == 'Target_m':
        #SET OBJECTIVE FUNCTION
        obj = cp.Minimize(cp.quad_form(w,S) )     
        cons += [w @ m >= value]
    
    if type == 'Target_v':
        #SET OBJECTIVE FUNCTION
        obj = cp.Minimize(-  w @ m  )  
        cons += [cp.quad_form(w,S)  <= value]
    
    #SET THE PROBLEM
    prob = cp.Problem(obj,cons)
    
    #SOLVE THE PROBLE
    prob.solve(verbose=verbose)
    print(f'Status: {prob.status}')
    
    return {
        "status": prob.status,
        "objective": prob.value,
        "weights": None if w.value is None else np.asarray(w.value).ravel(),
    }















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



#TRADING STRATEGIES



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