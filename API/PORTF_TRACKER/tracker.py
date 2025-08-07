
import requests
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"



class Tracker:
    """Create a Portfolio Tracker Object"""
    def __init__(self):
        """Initialize the attributes"""
        self.names = ['Asset name','ISIN','Macro-class','Initial price','Units','Current Price']
        self.data = pd.DataFrame(columns = self.names)
        #self.data.columns = self.names
        self.trans_list = []
        self.uniq_asset_names = []
        
        self.tickers = {}
        self.macro_class = {}
        
        self.portfolio_list = []
        self.current_portfolio = pd.DataFrame()
        self.total_portf_value = ""
    
    
    def fill(self, save = True):
        """Add positions."""
        
        while True:
            name = input('Type the name of the asset: ').lower()
            
            macro_class = input('Type of asset (crypto/stock): ')
            
            if macro_class.lower() == 'crypto':
                isin = input('Type the ISIN or n/a: ')
                curr_p = self._get_price_crypto(name)
            else:
                isin = input('Type the ISIN or n/a: ')
                if isin == 'n/a':
                    isin = self._search_ISIN(name)
                    curr_p = self._get_price(isin.upper(),name)
                else:
                    curr_p = self._get_price(isin.upper(),name)
                    isin = self._search_ISIN(name)
                
            self.macro_class[name]= macro_class.lower()
            self.tickers[name] = isin.upper()
                
            in_p = float(input('Type the initial price: '))
            units = float(input('Type the units: '))
            
            
            values = [name, isin, macro_class, in_p, units,curr_p]
            
            #Store the operation in a Dictionary
            dict_values = {}
            for i,j in enumerate(self.names):
                dict_values[j] =  values[i]
            self.trans_list.append(dict_values)
            
            
            #Store the operations in the dataframe
            self.data = pd.DataFrame(self.trans_list)
            self.uniq_asset_names = self.data['Asset name'].unique()
            
            print('\nNew transaction added.')
            if save == True:
                self.trans_write_on_disk()
                print()
            
            #Check for more
            go_on = input('You want to add another position? (yes/no) ')
            if go_on.lower() == 'no':
                if save == True:
                    self.trans_write_on_disk()
                    print('\nNew transactions saved on disk.')
                    break
            
            
    def show_all_transactions(self):
        """Show all transactions previously defined."""
        print('These are the positions taken so far:')
        for i in self.trans_list:
            print('\nTransaction.')
            for j,k in i.items():
                print(f'The {j} is: {k}')
                
    def show_transactions_database(self):
        if self.data.empty:
            print('The transaction database is empty.')
        else:
            print(self.data)
        
        """fig, ax = plt.subplots()
        ax.plot(self.data)
        plt.show()"""
        
    def trans_write_on_disk(self,title='trans_database.json'):
        """Write and save or update the Database of the transactions"""
        path = Path(title)
        path.write_text(json.dumps([self.trans_list,self.macro_class,self.tickers]))
        
        
        
    def trans_pull_from_database(self, title='trans_database.json'):
        """Pull from the local transactions database."""
        path = Path(title)
        try:
            content = path.read_text()
        except FileNotFoundError: 
            print('Database do not exist. One has been initiated now.')
            self.trans_write_on_disk()
        
        else:
            list_of_dict = json.loads(content)
            self.trans_list = list_of_dict[0]
            if not self.trans_list:
                print('The Database is empty. You have to add transactions.')
                self.fill()
            else:
                self.data = pd.DataFrame(self.trans_list)
                self.uniq_asset_names = self.data['Asset name'].unique()
                self.macro_class = list_of_dict[1]
                self.tickers = list_of_dict[2]

        self.data['Current price'] = self._update_trans_prices(self.data)      
            
                
    
    def reinitiate_trans_database(self):
        self.trans_list = []
        self.uniq_asset_names = []
        self.data = pd.DataFrame(self.trans_list)
        self.current_portfolio = pd.DataFrame()
        self.total_portf_value = ""
        self.trans_list = []
        self.uniq_asset_names = []
        self.tickers = {}
        self.macro_class = {}
        self.trans_write_on_disk()
        
    
    #Show the current situation (updated prices and returns)
    def _update_trans_prices(self,data):
        """"Update the Current price column"""
        macro_class = list(self.macro_class.values())
        tickers = list(self.tickers.values())
        names = list(self.data['Asset name'])
        updated_prices = []
        for i in range(len(data)):
            
            name = names[i]
            macro = macro_class[i]
            ticker = tickers[i]
            
            #Get via API the Current Price
            if macro == 'crypto':
                
                try:
                    url = f'https://api.coingecko.com/api/v3/simple/price?ids={name}&vs_currencies=usd'
                    response = requests.get(url)
                except ValueError:
                    print(f'For {name} is not possible to get an updated price.')
                    continue
                else: 
                    #print(f'Response status: {response.status_code}')
                    content = response.json()
                    updated_price = content[f'{name}']['usd'] 
                    updated_prices.append(updated_price)
            
            else:
                ticker = self.tickers[name]
                updated_price = self._get_price(ticker,name)
                updated_prices.append(updated_price)
        
        return updated_prices
                
        
    

    def download_trans_excel(self,name='Transactions.xlsx'):
        """Create an excel file of the dataframe in the current directory."""
        self.data.to_excel(name)
        
        
        
    
    #PORTFOLIO PART
    
    
    def get_live_portfolio(self):
        """Get the current portfolio at live prices."""
        self.units_dict = {}
        list_rows = [] 
        for name in self.uniq_asset_names:
            data = self.data[self.data['Asset name']== name]
            units = data['Units'].sum()
            weighted_price = data['Initial price']@ data['Units']/units
            self.units_dict[name] = units
            row = {
            'Asset name': name,
            'Macro-class': self.macro_class[name],
            'Units': units,
            'Initial price': round(weighted_price, 2)
            }
            list_rows.append(row)
        
        #Store the portfolio in a Dataframe
        self.current_portfolio = pd.DataFrame(list_rows)
        self.update_prices_portf()
        
        
        
    def update_prices_portf(self):
        """Update the Current price column"""
        names = self.uniq_asset_names
        updated_prices = []
        for name in names:
            #Get via API the Current Price
            if self.macro_class[name] == 'crypto':
                
                try:
                    url = f'https://api.coingecko.com/api/v3/simple/price?ids={name}&vs_currencies=usd'
                    response = requests.get(url)
                except ValueError:
                    print(f'For {name} is not possible to get an updated price.')
                    continue
                else: 
                    #print(f'Response status: {response.status_code}')
                    content = response.json()
                    updated_price = content[f'{name}']['usd'] 
                    updated_prices.append(updated_price)
            
            else:
                ticker = self.tickers[name]
                updated_price = self._get_price(ticker,name)
                updated_prices.append(updated_price)
                
        self.current_portfolio['Current price'] = updated_prices

        
        #HERE ADD RETURNS...
        
        self.current_portfolio['Return (%)'] = round(100*((self.current_portfolio['Current price']/self.current_portfolio['Initial price']) - 1),2) 
        self.current_portfolio['Exposure'] = self.current_portfolio['Current price']*self.current_portfolio['Units']
        self.total_portf_value = self.current_portfolio['Exposure'].sum()
        self.current_portfolio['Exposure (%)'] = round((self.current_portfolio['Exposure']/self.total_portf_value * 100),2)
        
        
        
    
    def show_current_portfolio(self):
        date = datetime.now()
        print(f'\nThis is the portfolio at {date} time.\nThe value of the portfolio is: {round(self.total_portf_value,2)} $')
        print(f'\n{self.current_portfolio}')
        
        
        # Create the table
        
        # Optional: format common numeric columns if they exist
        currency_cols = [c for c in self.current_portfolio.columns if c.lower() in {"price", "initial price", "current price", "exposure"}]
        pct_cols = [c for c in self.current_portfolio.columns if "percent" in c.lower() or c.endswith("(%)")]

        # Build per-column format/prefix/suffix lists matching df columns
        formats = []
        prefixes = []
        suffixes = []
        for col in self.current_portfolio.columns:
            if col in currency_cols:
                formats.append(",.2f"); prefixes.append("$"); suffixes.append("")
            elif col in pct_cols:
                formats.append(",.2f"); prefixes.append(""); suffixes.append("%")
            else:
                formats.append(None); prefixes.append(""); suffixes.append("")
        
        fig = go.Figure(data=[go.Table(
        columnorder=list(range(1, len(self.current_portfolio.columns) + 1)),
        columnwidth=[max(80, min(220, len(str(col))*10)) for col in self.current_portfolio.columns],
        header=dict(
            values=list(self.current_portfolio.columns),
            align="left",
            fill_color="#C8D4E3",
            font=dict(size=12, color="#2a3f5f")
        ),
        cells=dict(
            values=[self.current_portfolio[c].tolist() for c in self.current_portfolio.columns],
            align="left",
            fill_color="#EBF0F8",
            format=formats,
            prefix=prefixes,
            suffix=suffixes
        ))])

        fig.update_layout(
            title=f"Portfolio as of {datetime.now()} — Total value: ${self.total_portf_value:,.2f}",
            margin=dict(l=10, r=10, t=60, b=10))

        return fig.show()
        
        
        """# Matplotlib pie chart
    plt.figure(figsize=(5, 5))
    plt.pie(
        self.current_portfolio['Exposure (%)'],
        labels=self.current_portfolio['Asset name'],
        autopct='%1.1f%%',
        colors=['gold', 'lightblue', 'pink']
    )
    plt.title("Current portfolio weights")
    plt.show()"""
        
    def download_port_excel(self):
        """Create an excel file of the dataframe in the current directory."""
        date = datetime.now()
        name=f'Portfolio at {date}.xlsx'
        self.current_portfolio.to_excel(name)
            
            
            

    def _search_ISIN(self,string):
        """Search the ISIN and return it given a string."""
        string = string
        
        url = 'https://yahoo-finance15.p.rapidapi.com/api/v1/markets/search'
        querystring = {"search":string}
        headers = {
            "x-rapidapi-key": "6487ea259dmsh1cf031f35e0b4e1p17e5a5jsn1a70867302f5",
            "x-rapidapi-host": "yahoo-finance15.p.rapidapi.com"
        }
        
        response = requests.get(url,headers=headers,params=querystring)
        content = response.json()
        return content['body'][0]['symbol']
    
    
    def _get_price_crypto(self,name):
        """Get the price via API"""
        name = name
        #Get via API the Current Price
        try:
            url = f'https://api.coingecko.com/api/v3/simple/price?ids={name}&vs_currencies=usd'
            response = requests.get(url)
        except ValueError:
            print(f'{name} is not a valid crypto.')
        else:
            """if response.status_code == 200:
                print('Name successfully found.')"""
                
            #print(f'Response status: {response.status_code}')
            content = response.json()
            curr_p = content[f'{name}']['usd']
            return curr_p
        
    
    def _get_price(self,ticker,name=''):
        """Get the price via API"""
        ticker = ticker
        if ticker == '':
            name = name
            ticker = self._search_ISIN(name)
        
        
        #Get via API the Current Price
        url = "https://yahoo-finance15.p.rapidapi.com/api/v1/markets/quote"

        querystring = {"ticker":ticker,"type":"STOCKS"}

        headers = {
            "x-rapidapi-key": "6487ea259dmsh1cf031f35e0b4e1p17e5a5jsn1a70867302f5",
            "x-rapidapi-host": "yahoo-finance15.p.rapidapi.com"
        }
        try:
            response = requests.get(url, headers=headers, params=querystring)
        except ValueError:
            new_ticker = self._search_ISIN(name)
            print(f'{ticker} is not a valid ticker. We have used {new_ticker}')
            self._get_price(new_ticker)
        else:
            """if response.status_code == 200:
                print(f'{name} Price successfully found.')"""
            content = response.json()
            curr_p = content['body']['primaryData']['lastSalePrice']
            return float(curr_p[1:])
        
    