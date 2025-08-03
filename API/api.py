
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



