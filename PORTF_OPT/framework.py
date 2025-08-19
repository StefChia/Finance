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

from arch import arch_model


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
    """Download historical prices from YF returns the dataframe of returns"""
    start_date = datetime.datetime(2018,7,31)
    #end_date = datetime.datetime(2025,7,31)
    #period="3mo", interval="1d"
    
    data = pd.DataFrame()
    
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

    
            
def compute_returns(prices,log_ret=True):
    """Returns the dataframe of returns.
    You can choose if log or simple returns via log_ret = True/False
    log_ret as default."""
    prices = prices
    tickers = prices.columns
    data = pd.DataFrame(index=prices.index)
    
    if log_ret:
        for ticker in tickers:
            data[ticker]= np.log(prices[ticker]/prices[ticker].shift(1))
    else:
        for ticker in tickers:
            data[ticker]= (prices[ticker]/prices[ticker].shift(1)) -1

    return data.dropna()



def download_returns(tickers):
    """Return the returns for the selected tickers."""
    prices = download_historical_prices(tickers)
    return compute_returns(prices)
    
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




def compute_drawdowns(prices, plot=False):
    """
    Compute the drawdowns from a numpy array of prices.
    Args:
        prices: np.ndarray of shape (T,)
        plot: whether to plot drawdowns with markers
    Returns:
        drs: np.ndarray of drawdowns (<=0)
        [max_drawdown, peak_idx, trough_idx, recovery_idx]
    """

    prices = np.asarray(prices, dtype=float).reshape(-1)

    # running peak and drawdowns
    running_peak = np.maximum.accumulate(prices)
    drs = prices / running_peak - 1.0  # always <= 0

    # trough and max drawdown
    trough_idx = int(np.argmin(drs))
    max_drawdown = float(drs[trough_idx])

    # peak index: last time running_peak == peak_value before trough
    peak_value = running_peak[trough_idx]
    peak_candidates = np.where(running_peak[:trough_idx+1] == peak_value)[0]
    peak_idx = int(peak_candidates[0]) if peak_candidates.size > 0 else None

    # recovery: first time prices get back to peak_value after trough
    recovery_idx = None
    if trough_idx + 1 < len(prices):
        rec_candidates = np.where(prices[trough_idx+1:] >= peak_value)[0]
        if rec_candidates.size > 0:
            recovery_idx = int(trough_idx + 1 + rec_candidates[0])

    if plot:
        def _plot_drawdown_with_markers(drs, peak_idx, trough_idx, recovery_idx=None, title="Drawdown"):
            y = np.asarray(drs).reshape(-1)
            x = np.arange(len(y))

            fig, ax = plt.subplots()
            ax.plot(x, y, label="Drawdown")

            if peak_idx is not None:
                ax.scatter([x[peak_idx]], [y[peak_idx]], color="red", marker="o", zorder=3, label="Peak")
            if trough_idx is not None:
                ax.scatter([x[trough_idx]], [y[trough_idx]], color="red", marker="x", zorder=3, label="Trough")
            if recovery_idx is not None:
                ax.scatter([x[recovery_idx]], [y[recovery_idx]], color="green", marker="o", zorder=3, label="Recovery")

            ax.set_title(title)
            ax.set_ylabel("Drawdown")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
            plt.show()
        _plot_drawdown_with_markers(drs, peak_idx, trough_idx, recovery_idx)

    return drs, [max_drawdown, peak_idx, trough_idx, recovery_idx]





    


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














#COMPUTE SAMPLE VARCOV VIA PCs

def compute_sample_varcov_via_PCs_model(returns,thresh = 0.95,uncorr_res = True, jitter=1e-12):
    """Compute the sample variance via PCs model.
    thresh set the number of k eigenvalues that expalin at list the threshold percentage of variance.
    The aim is to de-noise the classical sample variance estimation."""
    
    ret = returns.values
    sample_varcov = np.cov(ret,rowvar=False)
    n = sample_varcov.shape[0]
    
    #COMPUTE FACTOR LOADING MATRIX (n X k)
    
    # symmetric eigendecomposition
    evals, evecs = np.linalg.eigh(sample_varcov)
    order = np.argsort(evals)[::-1]       # descending
    evals = evals[order]
    evecs = evecs[:, order]
    
    #Select k by explained variance ratio
    tot = np.clip(evals.sum(), 1e-30, None)
    cum = np.cumsum(evals) / tot
    k = int(np.searchsorted(cum, float(thresh)))
    k = min(max(1, k), n)
    
    evals = evals[:(k+1)]
    evecs = evecs[:,:(k+1)]                 # (N, k)
    
    #Compute PCs, residuals and their variance
    PCs = ret @ evecs
    residuals = ret - PCs @ evecs.T
    
    varcov_pcs = np.diag(evals)         # (k, k)
    
    if uncorr_res:
        varcov_res = np.diag(np.cov(residuals,rowvar=False))
    else:
        varcov_res = np.cov(residuals,rowvar=False)
    
    #Create the final variance
    varcov = evecs @ varcov_pcs @ evecs.T + varcov_res + jitter * np.eye(n)
    
    # Ensure PSD numerically (clip tiny negatives)
    varcov = 0.5 * (varcov + varcov.T)
    w, V = np.linalg.eigh(varcov)
    w = np.maximum(w, 0.0)
    varcov = V @ (w[:, None] * V.T)
    
    return varcov
    
    
    
    
    









#COMPUTE EWMA

def compute_ewma_train_sample(returns,lamda=0.95,via_PCs=True):
    """Compute the covariance prediction for the assets in the Dataframe.
    Output is a np.ndarray"""
    #Sample mean to initialize
    returns = returns - returns.mean()      #demean
    l = round(len(returns)* 0.4)
    
    if via_PCs:
        sample_v = compute_sample_varcov_via_PCs_model(returns)
    else:
        sample_v = np.cov(returns.iloc[:l],rowvar=False)
    
    
    ret = returns.iloc[l:]
    ret = ret.to_numpy()
    
    VARCOV = [sample_v]
    for i in range(len(ret)):
        r = ret[i]
        varcovt_1 = VARCOV[-1] * lamda + np.outer(r,r) * (1-lamda)
        VARCOV.append(varcovt_1)
    return np.array(VARCOV[-1])
    
    
    
def update_ewma_in_test_sample(varcov_1,test_sample):
    """Output is a list of the varcov prediction for all the test horizon.
    It start with the varcov_0 given from train sample."""
    #test_sample = test_sample - means
    
    varcov_test = [varcov_1]
    for i in range(len(test_sample)-1):
        varcov = varcov_test[-1]
        ret = test_sample.iloc[i]

        varcov_updated = update_ewma_step(varcov,ret)
        varcov_test.append(varcov_updated)
    
    return varcov_test


def update_ewma_step(varcov_1,ret_1,lamda=0.97):
        """Update step of ewma."""
        varcov_2 = varcov_1 * lamda + np.outer(ret_1,ret_1) * (1 - lamda)
        
        return np.array(varcov_2)

 
 #GARCH
 
 #FIT THE GARCH ON THE TRAINING SAMPLE
 
def fit_garch11_train_sample(ret_train):
    """Fit the Garch(1,1) model in the train sample dataframe. Returns a dictionary(keys=tickers) of dictionary(keys=omega,alpha,beta and v1 forecast.)"""
    tickers = ret_train.tickers
    
    data = {}
    for ticker in tickers: 
        model = arch_model(ret_train[ticker], vol='GARCH',p=1,q=1, mean='Zero')
        res = model.fit(disp="off")

        #print(res.summary())
        params = res.params
        forecast = res.forecast(horizon=1)
        v_1 = forecast.variance[-1:].values
        #print(v_0)
        data[ticker]={'omega':params['Omega'],'alpha':params['Alpha','beta':params['Beta'],'v1':v_1]}
    
    return data
"""
def update_garch_test_sample(data,ret_test):
    Given the estimated parameters for Grach models it updates the varcov forecast.
   """ 
        
 
 
 
 
 
 
 
   
   
        
        








#PORTFOLIO OPTIMIZATION


def portf_optim(ex_values,varcov,type='Both',value=1,long_only=True, thres_weig=0.15, solver="OSQP", verbose=False):
    """
    ex_values: length-n expected returns.
    varcov: (n,n) covariance matrix.
    type:
      - 'Both'      -> mean-variance with risk aversion r = 1/value
      - 'Target_m'  -> min variance s.t. w@m >= value
      - 'Target_v'  -> max return   s.t. w' S w <= value
    long_only: nonnegative weights if True.
    integer:   make weights integer (see scale_int).
    thres_weig: per-asset upper bound on weight.
    solver: choose CVXPY solver; if None we pick based on `integer`.
    eps_psd: floor for eigenvalues in PSD repair.
    epsI: add epsI*I to covariance for conditioning.
    scale_int: if not None and integer=True, solve in scaled integers.
               Example: scale_int=100 -> weights are multiples of 1/100.
    """
    ex_values = np.asarray(ex_values, dtype=float).reshape(-1,1)
    varcov = np.asarray(varcov, dtype=float)
    n = ex_values.size
    
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
    w = cp.Variable(n, name='weights', nonneg=long_only)
    
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
    #print(f'Status: {prob.status}')
    
    return {
        "status": prob.status,
        "objective": prob.value,
        "weights": None if w.value is None else np.asarray(w.value).ravel(),
    }


             
def launch_portf_opt_on_sample_test(exp_ret,varcov_for,returns,type='Both',value=1):
    """Input: list of exp_ret and a list of varcov_forecast for each step.
    Returns the portfolio returns under step by step optimization."""
    
    portf_ret = []
    for i in range(len(returns)):
        m = exp_ret[i]
        varcov = varcov_for[i]
        ret = returns.iloc[i]
        
        res = portf_optim(m,varcov,type,value)
        w = res['weights']
        portf_return = ret @ w
        portf_ret.append(portf_return)
        
    return np.array(portf_ret)



def risk_parity_portf(varcov,long_only=True, solver="SCS", eps=1e-8):
    """Compute the weights vector to have a risk parity portfolio."""
    n = varcov.shape[0]
    
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
    
    
    """Theorical Implementation (BUT NOT CONVEX)
    n = cov.shape[0]
    w = cp.Variable(n)

    # portfolio variance
    sigma_p = cp.sqrt(cp.quad_form(w, cov))

    # marginal risk contributions
    mrc = cov @ w / sigma_p

    # risk contributions
    rc = cp.multiply(w, mrc)
    rc = rc / cp.sum(rc)

    # objective: minimize squared deviation from equal risk
    objective = cp.Minimize(cp.sum_squares(rc - 1/n))

    constraints = [cp.sum(w) == 1, w >= 0]"""
    
    #ALTERNATIVE EQUIVALENT OPTIMIZATION PROBLEM
    
    #PARAMETERS
    varcov = cp.Parameter(shape=(n,n),PSD=True, value = varcov, name ='varcov')
    
    w = cp.Variable(n)

    # Constraint: portfolio variance normalized to 1
    constraints = [cp.quad_form(w, varcov) <= 1,
                   w >= eps]

    # Objective: maximize sum of log(w) == minimize -sum log(w)
    obj = cp.Maximize(cp.sum(cp.log(w)))

    prob = cp.Problem(obj,constraints)
    prob.solve(solver=solver)
    
    #print(f'Status: {prob.status}')
    
    return {
        "status": prob.status,
        "objective": prob.value,
        "weights": None if w.value is None else np.asarray(w.value / np.sum(w.value)).ravel(),
    }
    
    

def launch_risk_parity_opt_on_sample_test(varcov_for,returns,long_only=True):
    """Input: list of exp_ret and a list of varcov_forecast for each step.
    Returns the portfolio returns under step by step optimization."""
    
    portf_ret = []
    for i in range(len(returns)):
        varcov = varcov_for[i]
        ret = returns.iloc[i]
        
        res = risk_parity_portf(varcov,long_only)
        w = res['weights']
        portf_return = ret @ w
        portf_ret.append(portf_return)
        
    return np.array(portf_ret)


















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

    trading_ret = pd.DataFrame(values, index= ret.index,columns=['Ret'])
    prices = compute_prices_paths(trading_ret)

    if plot is not None:
        fig, ax = plt.subplots()
        ax.plot(ret.index,prices)
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