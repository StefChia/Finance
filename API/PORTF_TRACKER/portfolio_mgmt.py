
import pandas as pd
import yfinance as yf


class PortMngmt:
    """Portfolio Management advanced analytics"""
    def __init__(self,trans):
        """Initialize the attributes."""
        self.names = trans.uniq_asset_names
        #self.names_dates = trans.names_dates
        
    
    def download_historical_prices(self):
        """Download historical prices from YF and returns the dataframe of returns"""