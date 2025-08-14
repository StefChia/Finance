
import numpy as np
import pandas as pd
import yfinance as yf
import json
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
import plotly.express as px
import matplotlib.pyplot as plt

from tracker import Tracker


class PortMngmt(Tracker):
    """Portfolio Management advanced analytics"""
    def __init__(self,trans):
        """Initialize the attributes."""
        super().__init__()
        self.names = list(trans.tickers.keys())
        self.tickers = tuple(list(trans.tickers.values()))
        self.dates = trans.dates
        #self.start_date = min(datetime.strptime(d,"%Y-%m-%d") for d in self.dates)
        self.start_date = self.dates[0]
        self.data = trans.data
        
        
        self.prices = pd.DataFrame()
        self.returns = pd.DataFrame()
        self.prices_2 = pd.DataFrame()
        self.returns_2 = pd.DataFrame()
        
        self.hist_units = pd.DataFrame()
        self.hist_w = pd.DataFrame()
        self.port_ret = pd.DataFrame()
        self.port_value = pd.DataFrame()
        
    
    def download_historical_prices(self):
        """Download historical prices from YF returns the dataframe of returns"""
        start_date = datetime.strptime(self.start_date, '%Y-%m-%d')
        #start_date = self.start_date
        #end_date = datetime.datetime(2025,7,31) improve this in order not to need to delete last day price later
        data = pd.DataFrame()
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
            except:
                print(f'{ticker} not found.')
                continue
            else:
                data[ticker] = stock.history(start=start_date)['Close']
        
        data.index = pd.to_datetime(data.index)
        data.index = data.index.tz_localize(None).date
        self.prices = data
        return data


            
    def compute_returns(self):
        """Returns the dataframe of returns."""
        prices = self.prices
        tickers = prices.columns
        data = pd.DataFrame(index=prices.index)
        
        for ticker in tickers:
            data[ticker]= (prices[ticker]/prices[ticker].shift(1)) -1
        
        self.returns = data.dropna()
        return data.dropna()
    
    
    def compute_sample_statistics(self,prices, returns):
        """Compute sample mean, sample variance and sd, correlation matrix and drawdowns.
        Returns a dictionary."""
        #prices = self.prices
        #returns = self.returns
        tickers = prices.columns
        #tickers = self.tickers
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
            basic_stats_annaulized[ticker]['Drawdown'] = self._compute_drawdown_minimal(ticker,prices)[0]
            basic_stats_annaulized[ticker]['Drawdown Recovery'] = self._compute_drawdown_minimal(ticker,prices)[2]
        
        #print(basic_stats)
        #print(basic_stats_annaulized)
        self._show_df_html(basic_stats_annaulized)
        return basic_stats,basic_stats_annaulized,corr_matrix
    
        
        


    def _compute_drawdown_minimal(self,ticker,prices):
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

            idx_up, rec = self._find_recovery_index(tail, curr_p)
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



    def _find_recovery_index(self,list,value):
        """Find the first time the price recover, if it does, and return the index and the recovery status"""
        for j, v in enumerate(list):
            if v >= value:
                return j, 1
        return None, 0
     
     
    
    def _show_df_html(self,df):
        """Show the dataframe in html format"""
        data = df.copy()
        data = pd.DataFrame(data)
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


    def show_corr_matrix(self,corr_matrix):
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
        
        
    #PORTFOLIO PART  
    
    
    def get_port_dyn_ret(self):
        """Get portfolio dynamic returns."""
        self.returns_2 = self.returns[:-1]  #don't take the last day price since could not yet be available
        returns = self.returns_2.copy()
        #returns = returns[returns.index == self.dates[:-1]]
        w = self.get_hist_w()
        #print(returns)
        #print(w)
        self.port_ret = pd.DataFrame(index=returns.index)
        self.port_ret['Portf']= self._compute_portf_returns_changing_weights(returns,w)
        return self.port_ret
        
        
        
    def _compute_portf_returns_changing_weights(self,returns, weights):
        """Follow the portfolio weights path."""
        data = pd.DataFrame()
        data.index = returns.index

        #print(returns.columns)
        #print(weights.columns)
        
        data['Portf']= (returns * weights).sum(axis=1)
        return data      
        
    def get_hist_w(self):
        """Get a Dataframe of portfolio weights per asset in time."""
        self.prices_2 = self.prices[:-1]
        self.dates_2 = self.dates[:-1]
        prices = self.prices_2
        un = self._get_historical_units()
        un = un[:-1]
        units = pd.DataFrame(un,index= prices.index,columns=self.tickers)
        #print(units)
        #print(prices)
        w = prices * units
        t = w.sum(axis=1)
        
        self.port_value = pd.DataFrame(t)
        self.hist_units = units
        
        w = w.div(t, axis=0) 
        self.hist_w = w
        
        w = w.iloc[1:]    #align with returns indexing
        return w
        
    
    def _get_historical_units(self):
        """Get a Dataframe of portfolio units per asset in time."""
        dates = self.prices.index
        units = []
        for thresh_date in dates:
            thresh_int = int(thresh_date.strftime("%Y%m%d"))
            df = self.data[self.data.index <= thresh_int]
            units.append(self._get_dyn_units(df))
        return np.array(units)
            
            
    def _get_dyn_units(self,df):
        """Get the current portfolio at live prices."""
        self.units_dict = {}
        row_w = [] 
        for ticker in self.tickers:
            data = df[df['TICKER']== ticker]
            if data.empty:
                row_w.append(0.0)
            else:
                units = data['Units'].sum()
                row_w.append(float(units))
        return row_w
    
    
    def compute_prices_paths(self,p_0 = 100):
        """Show prices paths."""
        returns = self.port_ret
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
        
        
    def show_prices_paths(self,prices):
        """Show the price path"""
        dates = prices.index
        tickers = prices.columns
        
        np.random.seed(42)
        
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
    
        
        
                
            