
import requests
import pandas as pd
import numpy as np
import json
from pathlib import Path


class Tracker:
    """Create a Portfolio Tracker Object"""
    def __init__(self):
        """Initialize the attributes"""
        self.names = ['Asset name','ISIN', 'Initial price','Units','Current Price']
        self.data = pd.DataFrame(columns = self.names)
        #self.data.columns = self.names
        self.trans_list = []
        self.uniq_asset_names = []
        
        self.portfolio_list = []
        self.currrent_portfolio = pd.DataFrame(columns = self.names)
    
    
    def fill(self, save = True):
        """Add positions."""
        
        while True:
            name = input('Type the name of the asset: ').lower()
            isin = input('Type the ISIN: ')
            in_p = float(input('Type the initial price: '))
            units = float(input('Type the units: '))
            
            
            #Get via API the Current Price
            try:
                url = f'https://api.coingecko.com/api/v3/simple/price?ids={name}&vs_currencies=usd'
                response = requests.get(url)
            except ValueError:
                print(f'{name} is not a valid crypto.')
            else: 
                print(f'Respnse status: {response.status_code}')
                content = response.json()
                curr_p = content[f'{name}']['usd']
            
            
            values = [name,isin,in_p,units,curr_p]
            
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
        print(self.data)
        
    def trans_write_on_disk(self,title='trans_database.json'):
        """Write and save or update the Database of the transactions"""
        path = Path(title)
        path.write_text(json.dumps(self.trans_list))
        
    def trans_pull_from_database(self, title='trans_database.json'):
        """Pull from the local transactions database."""
        path = Path(title)
        try:
            content = path.read_text()
        except FileNotFoundError: 
            print('Database do not exist. One has been initiated now.')
            self.trans_write_on_disk()
        else:
            self.trans_list = json.loads(content)
            self.data = pd.DataFrame(self.trans_list)
    
    def reinitiate_trans_database(self):
        self.trans_list = []
        self.data = pd.DataFrame(self.trans_list)
        self.trans_write_on_disk()
        
    
        #Show the current situation (updated prices and returns)
    def update_prices(self):
        """Update the Current price column"""
        names = list(self.data['Asset name'])
        updated_prices = []
        for name in names:
            #Get via API the Current Price
            try:
                url = f'https://api.coingecko.com/api/v3/simple/price?ids={name}&vs_currencies=usd'
                response = requests.get(url)
            except ValueError:
                print(f'For {name} is not possible to get an updated price.')
                continue
            else: 
                print(f'Respnse status: {response.status_code}')
                content = response.json()
                updated_price = content[f'{name}']['usd'] 
                updated_prices.append(updated_price)
                
        self.data['Current price'] = updated_prices
    

            
            
            
            





class Portfolio:
    """Create a portfolio class."""
    def __init__(self):
        """Initialize the attributes."""
    
    #Manually Create a portfolio
    def manually_create_ptf (self, save = True):
        """Add positions."""
        self.names = ['Asset name','ISIN', 'Initial price','Units','Current price',]
        self.pos = []
        
        while True:
            name = input('Type the name of the asset: ').lower()
            isin = input('Type the ISIN: ')
            in_p = float(input('Type the initial price: '))
            units = float(input('Type the units: '))
            price = float(input('What is the price?'))
            
            values = [name,isin,in_p,units,price]
            
            #Store the operation in a Dictionary
            dict_values = {}
            for i,j in enumerate(self.names):
                dict_values[j] =  values[i]
                self.pos.append(dict_values)
            
            
            #Store the operations in the dataframe
            self.data = pd.DataFrame(self.pos)
            self.uniq_asset_names = self.data['Asset name'].unique()
            
            print('\nNew position added.')
            if save == True:
                self.trans_write_on_disk()
                print()
            
            #Check for more
            go_on = input('You want to add another position? (yes/no) ')
            if go_on.lower() == 'no':
                if save == True:
                    self.portf_write_on_disk()
                    print('\nNew position saved on disk.')
                    break
            
        
    
    
    #Show the current situation (updated prices and returns)
    def update_prices(self,database):
        """Update the Current price column"""
        names = list(database['Asset name'])
        updated_prices = []
        for name in names:
            #Get via API the Current Price
            try:
                url = f'https://api.coingecko.com/api/v3/simple/price?ids={name}&vs_currencies=usd'
                response = requests.get(url)
            except ValueError:
                print(f'For {name} is not possible to get an updated price.')
                continue
            else: 
                print(f'Respnse status: {response.status_code}')
                content = response.json()
                updated_price = content[f'{name}']['usd'] 
                updated_prices.append(updated_price)
                
        database['Current price'] = updated_price
    
    
    
    
    #From the transactions database create a portfolio
    
    
    """def currrent_portfolio(self):
        Show the current portfolio.
        for i in self.uniq_asset_names:
            d = self.data[self.data['Asset name']== i]
            d = d.select_dtypes(include = 'number').sum()"""