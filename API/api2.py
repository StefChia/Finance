
import requests
import time
import json


#BINANCE API REST
#documentation   'https://developers.binance.com/docs/binance-spot-api-docs/rest-api/general-api-information'

url_base = 'https://api.binance.com/'

#Market Day data
endpoint = '/api/v3/ticker/tradingDay'
#params = {'symbol':'BTCUSDT'}
params = {'symbols': json.dumps(['BTCUSDT', 'BNBUSDT'])}


url = url_base+endpoint
while True:
    response = requests.get(url,params=params)
    data = response.json()
    print(data)
    #print(f"\nSymbol: {data['symbol']}")
    #print(f'Last price: {data['lastPrice']}')
    #print(f'Volume: {data['volume']}')
    time.sleep(5)
    
"""#Price data
endpoint_price = '/api/v3/ticker/price'"""