
"""
#REST API


#WITH requests

#API call + response object


#Crypto from coingeko API
import requests
url = 'https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd'
response = requests.get(url)
print(response.json())

#custom
name = 'bitcoin'
url = f'https://api.coingecko.com/api/v3/simple/price?ids={name}&vs_currencies=usd'
response = requests.get(url)
print(response.json())


#STOCKS from exchange
import requests
import time

while True:
    response = requests.get("https://api.exchange.com/v1/price?symbol=AAPL")
    print(response.json())
    time.sleep(5)  # wait 5 seconds before asking again
    

import requests

url = "https://api.binance.com/api/v3/ticker/price"
params = {"symbol": "BTCUSDT"}

response = requests.get(url, params=params)
print(response.json())



#BINANCE API REST
#documentation   'https://developers.binance.com/docs/binance-spot-api-docs/rest-api/general-api-information'
url = 'https://api.binance.com/'





    
#WITH Websockets

import asyncio
import websockets

async def stream_price():
    uri = "wss://api.exchange.com/price_stream?symbol=AAPL"
    async with websockets.connect(uri) as ws:
        while True:
            data = await ws.recv()
            print(data)  # gets new price in real-time

asyncio.run(stream_price())"""



"""#STOCKS from exchange
#https://www.alphavantage.co/documentation/
#https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=IBM&apikey=demo
import requests
import time

while True:
    url = 'https://www.alphavantage.co/query'
    params = {'function':'GLOBAL_QUOTE','symbol':'IBM','apikey':'BYFW5SP86OWCEKKB'}
    response = requests.get(url,params=params)
    #response = requests.get("https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=IBM&apikey=BYFW5SP86OWCEKKB")
    content = response.json()
    price = content['Global Quote']['05. price']
    print(price)
    time.sleep(5)  # wait 5 seconds before asking again"""
    

"""import requests
import time

string = 'apple'

url = 'https://yahoo-finance15.p.rapidapi.com/api/v1/markets/search'
querystring = {"search":string}

headers = {
	"x-rapidapi-key": "6487ea259dmsh1cf031f35e0b4e1p17e5a5jsn1a70867302f5",
	"x-rapidapi-host": "yahoo-finance15.p.rapidapi.com"
}
response = requests.get(url,headers=headers,params=querystring)
content = response.json()
print(content['body'][0]['symbol'])
"""


import requests
import time


def _get_price(ticker):
        """Get the price via API"""
        ticker = ticker
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
            print(f'{ticker} is not a valid ticker.')
            return None
        else:
            content = response.json()
            curr_p = content['body']['primaryData']['lastSalePrice']
            return float(curr_p[1:])


def _search_ISIN(string):
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



print(_get_price('AAPL'))